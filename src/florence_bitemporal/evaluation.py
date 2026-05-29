import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .metrics import (
    boxes_to_mask,
    cdvqa_exact_match,
    mask_metrics,
    parse_loc_boxes,
    parse_polygons,
    polys_to_mask,
)
from .utils import ensure_dir, log
from .visualization import save_comparison


def gen_task_type(task):
    if task == "CHANGE_DETECTION":
        return "detection"
    if task == "CHANGE_SEGMENTATION":
        return "segmentation"
    return "vqa"


def truncate_for_display(text, limit=120):
    text = text.replace("\n", " ")
    if len(text) <= limit:
        return text
    half = (limit - 5) // 2
    return text[:half] + " ... " + text[-half:]


@torch.no_grad()
def evaluate(model, loader, processor, cfg, device, out_dir, tag="val"):
    model.eval()
    ensure_dir(out_dir)

    det_iou, det_f1, det_precision, det_recall, det_oa = [], [], [], [], []
    seg_iou, seg_f1, seg_oa = [], [], []
    vqa_correct = 0
    vqa_total = 0
    vqa_by_type = defaultdict(lambda: [0, 0])

    all_pred_texts = []
    vis_count = 0
    log_records = []
    samples_by_task = defaultdict(list)

    for batch in loader:
        pixel_values_t1 = batch["pixel_values_t1"].to(device, dtype=torch.bfloat16)
        pixel_values_t2 = batch["pixel_values_t2"].to(device, dtype=torch.bfloat16)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        tasks_in_batch = batch["tasks"]
        majority_task = Counter(tasks_in_batch).most_common(1)[0][0]
        generation_kind = gen_task_type(majority_task)

        generated = model.generate(
            pixel_values_t1,
            pixel_values_t2,
            input_ids,
            attention_mask,
            max_new_tokens=cfg.MAX_TEXT_LEN,
            task_type=generation_kind,
        )
        texts = processor.tokenizer.batch_decode(generated, skip_special_tokens=False)
        texts = [
            text.replace("<pad>", "").replace("</s>", "").replace("<s>", "").strip()
            for text in texts
        ]
        all_pred_texts.extend(texts)

        for index, task in enumerate(tasks_in_batch):
            sample_id = batch["ids"][index]
            instruction = batch["instructions"][index]
            gt = batch["responses"][index]
            pred = texts[index]
            mask_path = batch["mask_paths"][index]
            dataset_tag = batch["datasets"][index]
            question_type = batch["qtypes"][index]

            record = {
                "id": sample_id,
                "task": task,
                "dataset": dataset_tag,
                "qtype": question_type,
                "instruction": instruction,
                "pred": pred,
                "gt": gt,
            }

            pred_mask_for_vis = None

            if task == "CHANGE_VQA":
                correct = cdvqa_exact_match(pred, gt)
                vqa_correct += int(correct)
                vqa_total += 1
                vqa_by_type[question_type][1] += 1
                if correct:
                    vqa_by_type[question_type][0] += 1
                record["correct"] = bool(correct)

            elif task in ("CHANGE_DETECTION", "CHANGE_SEGMENTATION"):
                if mask_path and Path(mask_path).exists():
                    gt_mask = np.array(Image.open(mask_path).convert("L"))
                    height, width = gt_mask.shape
                    if task == "CHANGE_DETECTION":
                        pred_mask = boxes_to_mask(parse_loc_boxes(pred), height, width)
                    else:
                        pred_mask = polys_to_mask(parse_polygons(pred), height, width)

                    pred_mask_for_vis = pred_mask
                    metrics = mask_metrics(pred_mask, gt_mask)
                    record.update(metrics)

                    if task == "CHANGE_DETECTION":
                        det_iou.append(metrics["IoU"])
                        det_f1.append(metrics["F1"])
                        det_precision.append(metrics["Precision"])
                        det_recall.append(metrics["Recall"])
                        det_oa.append(metrics["OA"])
                    else:
                        seg_iou.append(metrics["IoU"])
                        seg_f1.append(metrics["F1"])
                        seg_oa.append(metrics["OA"])

            log_records.append(record)

            if len(samples_by_task[task]) < cfg.PRINT_SAMPLES_PER_TASK:
                samples_by_task[task].append((pred, gt))

            if vis_count < cfg.SAVE_VIS_N:
                sample = next(
                    (item for item in loader.dataset.samples if item["id"] == sample_id),
                    None,
                )
                if sample is not None:
                    out_path = os.path.join(out_dir, f"{tag}_{vis_count:03d}_{task}.png")
                    save_comparison(
                        sample["image_t1"],
                        sample["image_t2"],
                        sample.get("mask", ""),
                        pred_mask_for_vis,
                        pred,
                        gt,
                        instruction,
                        out_path,
                        dataset_tag=dataset_tag,
                    )
                    vis_count += 1

    metrics = {}
    if det_iou:
        metrics["det/IoU"] = float(np.mean(det_iou))
        metrics["det/F1"] = float(np.mean(det_f1))
        metrics["det/Prec"] = float(np.mean(det_precision))
        metrics["det/Rec"] = float(np.mean(det_recall))
        metrics["det/OA"] = float(np.mean(det_oa))
        metrics["det/n"] = len(det_iou)
    if seg_iou:
        metrics["seg/IoU"] = float(np.mean(seg_iou))
        metrics["seg/F1"] = float(np.mean(seg_f1))
        metrics["seg/OA"] = float(np.mean(seg_oa))
        metrics["seg/n"] = len(seg_iou)
    if vqa_total > 0:
        metrics["vqa/accuracy"] = vqa_correct / vqa_total
        metrics["vqa/n"] = vqa_total
        for question_type, (correct, total) in vqa_by_type.items():
            if total > 0:
                metrics[f"vqa/{question_type}_acc"] = correct / total
                metrics[f"vqa/{question_type}_n"] = total

    if all_pred_texts:
        unique_preds = len(set(all_pred_texts))
        diversity_ratio = unique_preds / len(all_pred_texts)
        metrics["diag/unique_preds"] = unique_preds
        metrics["diag/total_preds"] = len(all_pred_texts)
        metrics["diag/diversity_ratio"] = diversity_ratio
        if diversity_ratio < 0.50:
            log(f"  *** WARNING: low prediction diversity ({diversity_ratio:.1%}). ***")

    with open(os.path.join(out_dir, f"{tag}_metrics.json"), "w") as file:
        json.dump(metrics, file, indent=2)
    with open(os.path.join(out_dir, f"{tag}_predictions.json"), "w") as file:
        json.dump(log_records, file, indent=2)

    log(
        f"[{tag}] "
        + "  ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in metrics.items()
        )
    )

    log(f"  Sample predictions ({tag}):")
    for task in ("CHANGE_DETECTION", "CHANGE_SEGMENTATION", "CHANGE_VQA"):
        for index, (pred, gt) in enumerate(samples_by_task.get(task, [])):
            log(f"    [{task} #{index}]")
            log(f"      gt  : {truncate_for_display(gt)}")
            log(f"      pred: {truncate_for_display(pred)}")

    return metrics
