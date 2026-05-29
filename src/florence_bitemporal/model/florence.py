from typing import Optional

import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor

from ..config import CFG, Config
from ..utils import log
from .fusion import TemporalFusionModule


class Florence2BiTemporal(nn.Module):
    def __init__(self, florence_path, lora_cfg, cfg: Config = CFG):
        super().__init__()
        self.cfg = cfg
        log(f"Loading Florence-2 from {florence_path}")

        self.processor = AutoProcessor.from_pretrained(florence_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            florence_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        new_tokens = list(cfg.TASK_TOKENS)
        added = self.processor.tokenizer.add_tokens(new_tokens, special_tokens=True)
        if added > 0:
            base.resize_token_embeddings(len(self.processor.tokenizer))
            log(f"Added {added} special tokens; vocab now {len(self.processor.tokenizer)}.")
            self._init_new_task_tokens(base)

        try:
            base.language_model = get_peft_model(base.language_model, lora_cfg)
            log("Applied LoRA to language_model.")
        except Exception as exc:
            log(f"[WARN] LoRA failed: {exc}; training full LM.")

        self.florence = base
        self.tfm: Optional[TemporalFusionModule] = None
        self._pos_check_done = False

    def _init_new_task_tokens(self, base):
        """
        Initialize newly-added task tokens from semantically related Florence-2
        native task tokens instead of random noise.
        """
        tokenizer = self.processor.tokenizer
        init_map = {
            "<CHANGE_DETECTION>": ["<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>"],
            "<CHANGE_SEGMENTATION>": [
                "<REGION_TO_SEGMENTATION>",
                "<REFERRING_EXPRESSION_SEGMENTATION>",
            ],
            "<CHANGE_VQA>": ["<CAPTION>", "<MORE_DETAILED_CAPTION>", "<DETAILED_CAPTION>"],
        }

        try:
            embed = base.language_model.get_input_embeddings()
        except Exception:
            log("  [FIX 27] Could not access language_model.get_input_embeddings(); skipping init.")
            return

        initialized = 0
        with torch.no_grad():
            for new_token, source_candidates in init_map.items():
                new_id = tokenizer.convert_tokens_to_ids(new_token)
                if new_id is None or new_id == tokenizer.unk_token_id:
                    continue

                source_used = None
                source_embedding = None
                for source_token in source_candidates:
                    source_id = tokenizer.convert_tokens_to_ids(source_token)
                    if (
                        source_id is not None
                        and source_id != tokenizer.unk_token_id
                        and source_id < embed.weight.shape[0]
                    ):
                        source_embedding = embed.weight[source_id].detach().clone()
                        source_used = f"{source_token} (single token)"
                        break

                    source_ids = tokenizer.encode(source_token, add_special_tokens=False)
                    valid_ids = [
                        item
                        for item in source_ids
                        if item != tokenizer.unk_token_id and item < embed.weight.shape[0]
                    ]
                    if valid_ids:
                        source_embedding = embed.weight[valid_ids].detach().mean(dim=0)
                        source_used = f"{source_token} (avg of {len(valid_ids)} subtokens)"
                        break

                if source_embedding is None:
                    log(f"  [FIX 27] No source found for {new_token}; leaving random.")
                    continue

                embed.weight[new_id].copy_(source_embedding.to(embed.weight.dtype))
                initialized += 1
                log(f"  [FIX 27] {new_token:25s}  <-  {source_used}")

        log(f"  [FIX 27] Initialized {initialized}/{len(init_map)} new task tokens.")

    def _encode_vision(self, pixel_values):
        if hasattr(self.florence, "_encode_image"):
            return self.florence._encode_image(pixel_values)
        vision_features = self.florence.vision_tower(pixel_values)
        return self.florence.image_projection(vision_features)

    def _ensure_tfm(self, dim, device, dtype):
        if self.tfm is None:
            log(f"Building TFM dim={dim}")
            self.tfm = TemporalFusionModule(dim=dim).to(device=device, dtype=torch.float32)

    def _check_position_budget(self, total_len):
        if self._pos_check_done:
            return

        self._pos_check_done = True
        config = getattr(self.florence, "config", None)
        text_config = getattr(config, "text_config", config) if config is not None else None
        max_position = (
            getattr(text_config, "max_position_embeddings", 1024)
            if text_config is not None
            else 1024
        )
        log(f"  Encoder input length: {total_len}  (max_position_embeddings={max_position})")
        if total_len > max_position:
            log(
                f"  *** WARNING: encoder input ({total_len}) exceeds "
                f"max_position_embeddings ({max_position}). ***"
            )

    def _fused_inputs_embeds(self, pixel_values_t1, pixel_values_t2, input_ids):
        features_t1 = self._encode_vision(pixel_values_t1)
        features_t2 = self._encode_vision(pixel_values_t2)
        self._ensure_tfm(features_t1.shape[-1], features_t1.device, features_t1.dtype)
        visual_features = self.tfm(features_t1, features_t2)

        language_model = self.florence.language_model
        try:
            embed = language_model.get_input_embeddings()
        except Exception:
            embed = language_model.base_model.model.get_input_embeddings()

        text_embeddings = embed(input_ids)
        visual_features = visual_features.to(text_embeddings.dtype)
        fused = torch.cat([visual_features, text_embeddings], dim=1)
        self._check_position_budget(fused.shape[1])
        return fused, visual_features.shape[1]

    def forward(self, pixel_values_t1, pixel_values_t2, input_ids, attention_mask, labels=None):
        inputs_embeds, num_visual_tokens = self._fused_inputs_embeds(
            pixel_values_t1,
            pixel_values_t2,
            input_ids,
        )
        batch_size = input_ids.shape[0]
        visual_mask = torch.ones(
            batch_size,
            num_visual_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        encoder_attention = torch.cat([visual_mask, attention_mask], dim=1)
        return self.florence.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_attention,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values_t1,
        pixel_values_t2,
        input_ids,
        attention_mask,
        max_new_tokens=256,
        task_type="vqa",
        **gen_kwargs,
    ):
        inputs_embeds, num_visual_tokens = self._fused_inputs_embeds(
            pixel_values_t1,
            pixel_values_t2,
            input_ids,
        )
        batch_size = input_ids.shape[0]
        visual_mask = torch.ones(
            batch_size,
            num_visual_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        encoder_attention = torch.cat([visual_mask, attention_mask], dim=1)

        if task_type in ("detection", "segmentation"):
            gen_args = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "num_beams": self.cfg.GEN_NUM_BEAMS_SPATIAL,
                "repetition_penalty": self.cfg.GEN_REPETITION_PENALTY,
                "length_penalty": self.cfg.GEN_LENGTH_PENALTY,
                "no_repeat_ngram_size": self.cfg.GEN_NO_REPEAT_NGRAM,
                "early_stopping": True,
            }
        else:
            gen_args = {
                "max_new_tokens": 16,
                "do_sample": False,
                "num_beams": self.cfg.GEN_NUM_BEAMS_VQA,
                "length_penalty": 1.0,
                "early_stopping": True,
            }
        gen_args.update(gen_kwargs)

        return self.florence.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_attention,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **gen_args,
        )

    def trainable_parameters(self):
        trainable = 0
        total = 0
        for param in self.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        return trainable, total
