import argparse
import json
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from peft import LoraConfig, TaskType
from PIL import Image
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .config import CFG
from .data import MultiTaskDataset, build_all_jsons, make_balanced_sampler, make_collate
from .diagnostics import run_diagnostic
from .evaluation import evaluate
from .model import Florence2BiTemporal
from .training import build_optimizer, freeze_strategy, train_one_epoch
from .utils import ensure_dir, log, set_seed, steps_per_epoch


def build_parser(default_mode="diagnostic"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["diagnostic", "build_json", "sanity", "train", "eval", "infer"],
        default=default_mode,
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--infer_t1", type=str, default=None)
    parser.add_argument("--infer_t2", type=str, default=None)
    parser.add_argument(
        "--infer_q",
        type=str,
        default="What is the change between these two images?",
    )
    parser.add_argument(
        "--infer_task",
        choices=["CHANGE_VQA", "CHANGE_DETECTION", "CHANGE_SEGMENTATION"],
        default="CHANGE_VQA",
    )
    parser.add_argument(
        "--eval_split",
        choices=["val", "cdvqa_test", "cdvqa_test2", "levir_test"],
        default="val",
    )
    parser.add_argument("--curriculum", choices=["none", "levir_first"], default="none")
    parser.add_argument("--resume", action="store_true")
    return parser


def create_lora_config():
    return LoraConfig(
        r=CFG.LORA_RANK,
        lora_alpha=CFG.LORA_ALPHA,
        lora_dropout=CFG.LORA_DROPOUT,
        target_modules=list(CFG.LORA_TARGETS),
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )


def warmup_tfm(model, processor, device):
    log("Warming up TFM ...")
    dummy = Image.new("RGB", (CFG.IMG_SIZE, CFG.IMG_SIZE))
    dummy_pixels = processor.image_processor(images=[dummy], return_tensors="pt")[
        "pixel_values"
    ].to(device, dtype=torch.bfloat16)
    dummy_ids = processor.tokenizer(["<CHANGE_VQA> hi"], return_tensors="pt").input_ids.to(
        device
    )
    dummy_attention = torch.ones_like(dummy_ids)
    dummy_labels = processor.tokenizer(["test"], return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        _ = model(dummy_pixels, dummy_pixels, dummy_ids, dummy_attention, labels=dummy_labels)


def maybe_build_jsons(json_dir):
    if not (json_dir / "train.json").exists():
        log("Train JSON not found - running diagnostic + build_json ...")
        ok = run_diagnostic(CFG)
        if not ok:
            log("Diagnostic failed. Exit.")
            return False
        build_all_jsons(CFG)
    return True


def prepare_paths():
    ensure_dir(CFG.OUTPUT_DIR)
    json_dir = Path(CFG.OUTPUT_DIR) / "json"
    ckpt_dir = Path(CFG.OUTPUT_DIR) / "ckpt"
    vis_dir = Path(CFG.OUTPUT_DIR) / "vis"
    ensure_dir(json_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(vis_dir)
    return json_dir, ckpt_dir, vis_dir


def load_checkpoint_if_needed(model, args, ckpt_dir):
    resume_path = ckpt_dir / "last.pt"
    if args.ckpt and Path(args.ckpt).exists():
        log(f"Loading checkpoint: {args.ckpt}")
        state_dict = torch.load(args.ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log(f"  Missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    elif args.resume and resume_path.exists():
        log(f"Auto-resume from: {resume_path}")
        state_dict = torch.load(resume_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log(f"  Missing keys: {len(missing)}, unexpected: {len(unexpected)}")


def run_inference(model, processor, args, device):
    assert args.infer_t1 and args.infer_t2

    image_t1 = Image.open(args.infer_t1).convert("RGB").resize((CFG.IMG_SIZE,) * 2)
    image_t2 = Image.open(args.infer_t2).convert("RGB").resize((CFG.IMG_SIZE,) * 2)

    pixel_values_t1 = processor.image_processor(images=[image_t1], return_tensors="pt")[
        "pixel_values"
    ].to(device, dtype=torch.bfloat16)
    pixel_values_t2 = processor.image_processor(images=[image_t2], return_tensors="pt")[
        "pixel_values"
    ].to(device, dtype=torch.bfloat16)

    instruction = f"<{args.infer_task}> {args.infer_q}"
    tokens = processor.tokenizer([instruction], return_tensors="pt")
    if args.infer_task == "CHANGE_VQA":
        kind = "vqa"
    elif args.infer_task == "CHANGE_DETECTION":
        kind = "detection"
    else:
        kind = "segmentation"

    generated = model.generate(
        pixel_values_t1,
        pixel_values_t2,
        tokens.input_ids.to(device),
        tokens.attention_mask.to(device),
        task_type=kind,
        max_new_tokens=CFG.MAX_TEXT_LEN,
    )
    output = processor.tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
    print("=" * 60)
    print("Q:", instruction)
    print("A:", output)
    print("=" * 60)
    return 0


def choose_train_val_paths(args, json_dir):
    if args.mode == "sanity":
        samples_per_task = CFG.SANITY_SAMPLES_PER_TASK
        log(
            f"=== SANITY: {3 * samples_per_task} samples ({samples_per_task}/task), "
            f"{CFG.SANITY_EPOCHS} epochs (overfit test) ==="
        )
        with open(json_dir / "train.json") as file:
            data = json.load(file)

        levir_det = [
            sample for sample in data if sample["task"] == "CHANGE_DETECTION"
        ][:samples_per_task]
        levir_seg = [
            sample for sample in data if sample["task"] == "CHANGE_SEGMENTATION"
        ][:samples_per_task]
        cdvqa = [sample for sample in data if sample["task"] == "CHANGE_VQA"][
            :samples_per_task
        ]
        small = levir_det + levir_seg + cdvqa
        sanity_path = json_dir / "sanity.json"
        with open(sanity_path, "w") as file:
            json.dump(small, file, indent=2)
        return sanity_path, sanity_path, CFG.SANITY_EPOCHS, 1

    if args.mode == "eval":
        return (
            json_dir / "train.json",
            json_dir / f"{args.eval_split}.json",
            CFG.NUM_EPOCHS,
            None,
        )

    return json_dir / "train.json", json_dir / "val.json", CFG.NUM_EPOCHS, None


def build_dataloaders(args, processor, train_path, val_path, accum_override):
    train_ds = MultiTaskDataset(str(train_path), processor, CFG.IMG_SIZE)
    val_ds = MultiTaskDataset(str(val_path), processor, CFG.IMG_SIZE)
    collate = make_collate(processor, CFG.MAX_TEXT_LEN)

    if args.mode == "sanity":
        train_loader = DataLoader(
            train_ds,
            batch_size=CFG.BATCH_SIZE,
            shuffle=True,
            num_workers=CFG.NUM_WORKERS,
            collate_fn=collate,
            pin_memory=True,
        )
    else:
        sampler = make_balanced_sampler(train_ds.samples)
        train_loader = DataLoader(
            train_ds,
            batch_size=CFG.BATCH_SIZE,
            sampler=sampler,
            num_workers=CFG.NUM_WORKERS,
            collate_fn=collate,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=collate,
        pin_memory=True,
    )
    log(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    levir_loader = None
    if args.curriculum == "levir_first" and args.mode == "train":
        log("=== CURRICULUM: epoch 1 will be LEVIR-only ===")
        levir_only = [sample for sample in train_ds.samples if sample["dataset"] == "levir"]
        levir_ds = MultiTaskDataset(str(train_path), processor, CFG.IMG_SIZE)
        levir_ds.samples = levir_only
        levir_sampler = make_balanced_sampler(levir_only)
        levir_loader = DataLoader(
            levir_ds,
            batch_size=CFG.BATCH_SIZE,
            sampler=levir_sampler,
            num_workers=CFG.NUM_WORKERS,
            collate_fn=collate,
            pin_memory=True,
        )

    return train_ds, val_ds, train_loader, val_loader, levir_loader


def run_training(args, model, processor, train_loader, val_loader, levir_loader, epochs, accum_override, ckpt_dir, vis_dir):
    optimizer = build_optimizer(model, CFG)

    accum_for_sched = accum_override or CFG.GRAD_ACCUM_STEPS
    if args.curriculum == "levir_first" and args.mode == "train":
        total_steps = steps_per_epoch(len(levir_loader), accum_for_sched) + steps_per_epoch(
            len(train_loader),
            accum_for_sched,
        ) * (epochs - 1)
    else:
        total_steps = steps_per_epoch(len(train_loader), accum_for_sched) * epochs

    total_steps = max(1, total_steps)
    warmup = max(1, int(total_steps * CFG.WARMUP_RATIO))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    log(f"  Scheduler: total_steps={total_steps}, warmup={warmup}, accum={accum_for_sched}")

    best = -1.0
    history = []
    device = next(model.parameters()).device

    for epoch in range(1, epochs + 1):
        log(f"=== Epoch {epoch}/{epochs} ===")
        if args.curriculum == "levir_first" and epoch == 1 and args.mode == "train":
            log("  Using LEVIR-only loader for curriculum warmup.")
            this_loader = levir_loader
        else:
            this_loader = train_loader

        train_loss = train_one_epoch(
            model,
            this_loader,
            optimizer,
            scheduler,
            epoch,
            CFG,
            device,
            accum_steps_override=accum_override,
        )
        log(f"Epoch {epoch} train_loss={train_loss:.4f}")

        if CFG.EVAL_EVERY_EPOCH or args.mode == "sanity":
            metrics = evaluate(
                model,
                val_loader,
                processor,
                CFG,
                device,
                str(vis_dir / f"epoch_{epoch}"),
                tag=f"val_e{epoch}",
            )
            score = (
                metrics.get("det/F1", 0)
                + metrics.get("seg/F1", 0)
                + metrics.get("vqa/accuracy", 0)
            )
            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
            with open(Path(CFG.OUTPUT_DIR) / "history.json", "w") as file:
                json.dump(history, file, indent=2)

            if score > best:
                best = score
                best_path = ckpt_dir / "best.pt"
                torch.save(model.state_dict(), best_path)
                log(f"Saved best -> {best_path}  (score={score:.4f})")

        torch.save(model.state_dict(), ckpt_dir / "last.pt")

    log("Training complete.")
    return 0


def main(default_mode="diagnostic", argv=None):
    parser = build_parser(default_mode=default_mode)
    args = parser.parse_args(argv)

    set_seed(CFG.SEED)
    json_dir, ckpt_dir, vis_dir = prepare_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "diagnostic":
        ok = run_diagnostic(CFG)
        return 0 if ok else 1

    if args.mode == "build_json":
        run_diagnostic(CFG)
        log("\n=== Building JSONs ===")
        build_all_jsons(CFG)
        return 0

    if not maybe_build_jsons(json_dir):
        return 1

    train_path, val_path, epochs, accum_override = choose_train_val_paths(args, json_dir)

    lora_cfg = create_lora_config()
    model = Florence2BiTemporal(CFG.FLORENCE_PATH, lora_cfg, CFG).to(device)
    processor = model.processor

    warmup_tfm(model, processor, device)
    freeze_strategy(model, CFG)
    trainable, total = model.trainable_parameters()
    log(f"Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M  ({100 * trainable / total:.2f}%)")

    load_checkpoint_if_needed(model, args, ckpt_dir)

    if args.mode == "infer":
        return run_inference(model, processor, args, device)

    train_ds, val_ds, train_loader, val_loader, levir_loader = build_dataloaders(
        args,
        processor,
        train_path,
        val_path,
        accum_override,
    )

    if args.mode == "eval":
        evaluate(
            model,
            val_loader,
            processor,
            CFG,
            device,
            str(vis_dir / f"eval_{args.eval_split}"),
            tag=args.eval_split,
        )
        return 0

    return run_training(
        args,
        model,
        processor,
        train_loader,
        val_loader,
        levir_loader,
        epochs,
        accum_override,
        ckpt_dir,
        vis_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
