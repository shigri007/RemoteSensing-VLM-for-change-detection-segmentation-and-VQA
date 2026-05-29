import re

import torch

from ..config import CFG, Config
from ..utils import log


def build_optimizer(model, cfg: Config = CFG):
    tfm_params = []
    proj_params = []
    vision_params = []
    embed_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "tfm" in name:
            tfm_params.append(param)
        elif any(key in name for key in ("embed_tokens", "lm_head", "shared")):
            embed_params.append(param)
        elif (
            "image_projection" in name
            or "image_proj" in name
            or "visual_temporal_embed" in name
            or "image_pos_embed" in name
        ):
            proj_params.append(param)
        elif "vision_tower" in name:
            vision_params.append(param)
        else:
            other_params.append(param)

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": cfg.LR, "name": "other"})
    if tfm_params:
        groups.append({"params": tfm_params, "lr": cfg.LR_TFM, "name": "tfm"})
    if proj_params:
        groups.append({"params": proj_params, "lr": cfg.LR_PROJ, "name": "proj"})
    if vision_params:
        groups.append({"params": vision_params, "lr": cfg.LR_VISION, "name": "vision"})
    if embed_params:
        groups.append({"params": embed_params, "lr": cfg.LR_EMBED, "name": "embed"})

    log(
        f"  Optimizer groups: other={len(other_params)} tfm={len(tfm_params)} "
        f"proj={len(proj_params)} vision={len(vision_params)} embed={len(embed_params)}"
    )
    return torch.optim.AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)


def freeze_strategy(model, cfg: Config = CFG):
    for _, param in model.named_parameters():
        param.requires_grad = False

    if model.tfm is not None:
        for param in model.tfm.parameters():
            param.requires_grad = True

    for name, param in model.florence.language_model.named_parameters():
        if "lora_" in name.lower():
            param.requires_grad = True
        if any(key in name for key in ["embed_tokens", "lm_head", "shared"]):
            param.requires_grad = True

    projection_keys = (
        "image_projection",
        "image_proj",
        "visual_temporal_embed",
        "image_pos_embed",
        "image_proj_norm",
    )
    num_projection_unfrozen = 0
    for name, param in model.florence.named_parameters():
        if any(key in name for key in projection_keys):
            param.requires_grad = True
            num_projection_unfrozen += 1
    log(f"  Unfroze {num_projection_unfrozen} vision-projection tensors.")

    if cfg.UNFREEZE_VISION_LAST_BLOCK:
        vision_tower = getattr(model.florence, "vision_tower", None)
        if vision_tower is None:
            log("  [WARN] vision_tower not found; skipping last-block unfreeze.")
        else:
            vision_names = [name for name, _ in vision_tower.named_parameters()]
            stage_ids = set()
            for name in vision_names:
                match = re.search(r"(?:stages?|layers?|blocks?)[._](\d+)", name)
                if match:
                    stage_ids.add(int(match.group(1)))

            num_vision_unfrozen = 0
            if stage_ids:
                last_id = max(stage_ids)
                for name, param in vision_tower.named_parameters():
                    if re.search(rf"(?:stages?|layers?|blocks?)[._]{last_id}\b", name):
                        param.requires_grad = True
                        num_vision_unfrozen += param.numel()
                log(
                    f"  Unfroze vision_tower last stage/block id={last_id} "
                    f"({num_vision_unfrozen / 1e6:.2f}M params)."
                )
            else:
                params = list(vision_tower.named_parameters())
                cutoff = int(0.95 * len(params))
                for _, param in params[cutoff:]:
                    param.requires_grad = True
                    num_vision_unfrozen += param.numel()
                log(
                    f"  Unfroze tail 5% of vision_tower "
                    f"({num_vision_unfrozen / 1e6:.2f}M params)."
                )
