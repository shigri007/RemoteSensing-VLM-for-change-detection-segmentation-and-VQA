import re

import cv2
import numpy as np

LOC_RE = re.compile(r"<loc_(\d+)>")


def parse_loc_boxes(text):
    nums = [int(match.group(1)) for match in LOC_RE.finditer(text)]
    return [nums[index : index + 4] for index in range(0, len(nums) - 3, 4)]


def boxes_to_mask(boxes_999, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes_999:
        x1 = int(box[0] / 999 * width)
        y1 = int(box[1] / 999 * height)
        x2 = int(box[2] / 999 * width)
        y2 = int(box[3] / 999 * height)
        x1, x2 = sorted([max(0, x1), min(width, x2)])
        y1, y2 = sorted([max(0, y1), min(height, y2)])
        mask[y1:y2, x1:x2] = 1
    return mask


def parse_polygons(text):
    polygons = []
    for match in re.finditer(r"<poly>(.*?)</poly>", text, flags=re.DOTALL):
        nums = [int(item.group(1)) for item in re.finditer(r"<loc_(\d+)>", match.group(1))]
        points = [(nums[index], nums[index + 1]) for index in range(0, len(nums) - 1, 2)]
        if len(points) >= 3:
            polygons.append(points)

    if polygons:
        return polygons

    if "<poly>" in text:
        chunks = text.split("<poly>")[1:]
        for chunk in chunks:
            chunk = chunk.split("<poly>")[0]
            chunk = chunk.split("</poly>")[0]
            nums = [int(item.group(1)) for item in re.finditer(r"<loc_(\d+)>", chunk)]
            points = [(nums[index], nums[index + 1]) for index in range(0, len(nums) - 1, 2)]
            if len(points) >= 3:
                polygons.append(points)
    return polygons


def polys_to_mask(polys_999, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polys_999:
        points = np.array(
            [[int(x / 999 * width), int(y / 999 * height)] for x, y in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 1)
    return mask


def mask_metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    true_positive = float(((pred == 1) & (gt == 1)).sum())
    false_positive = float(((pred == 1) & (gt == 0)).sum())
    false_negative = float(((pred == 0) & (gt == 1)).sum())
    true_negative = float(((pred == 0) & (gt == 0)).sum())

    iou = true_positive / (true_positive + false_positive + false_negative + 1e-9)
    precision = true_positive / (true_positive + false_positive + 1e-9)
    recall = true_positive / (true_positive + false_negative + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    oa = (true_positive + true_negative) / (
        true_positive + true_negative + false_positive + false_negative + 1e-9
    )
    return {
        "IoU": iou,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "OA": oa,
    }


def cdvqa_exact_match(pred, gt):
    pred_norm = pred.strip().lower().replace(" ", "_").rstrip(".,!?\"'")
    gt_norm = gt.strip().lower()
    if pred_norm == gt_norm:
        return True

    pred_clean = re.sub(r"[^a-z0-9_]", "", pred_norm)
    gt_clean = re.sub(r"[^a-z0-9_]", "", gt_norm)
    return pred_clean == gt_clean
