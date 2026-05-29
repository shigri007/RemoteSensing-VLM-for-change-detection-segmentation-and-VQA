from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..config import CFG, Config


def analyze_change_mask(mask):
    if mask.max() > 1:
        normalized = (mask > 127).astype(np.uint8)
    else:
        normalized = mask.astype(np.uint8)

    height, width = normalized.shape
    info = {
        "H": height,
        "W": width,
        "has_change": bool(normalized.sum() > 0),
    }
    info["change_ratio"] = float(normalized.sum()) / (height * width)

    num, _, stats, _ = cv2.connectedComponentsWithStats(normalized, connectivity=8)
    components = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 50:
            continue
        components.append(
            {
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(area),
            }
        )
    components.sort(key=lambda item: -item["area"])
    info["components"] = components
    info["num_components"] = len(components)
    return info


def normalize_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(999, int(value / dim * 999)))
        for value, dim in zip([x1, y1, x2, y2], [width, height, width, height])
    ]


def build_detection_response(info, width, height):
    if not info["has_change"]:
        return "no_change"

    parts = []
    for component in info["components"][:20]:
        bbox = normalize_bbox(component["bbox"], width, height)
        parts.append("changed_region" + "".join(f"<loc_{value}>" for value in bbox))
    return "".join(parts)


def build_segmentation_response(mask, info):
    if not info["has_change"]:
        return "no_change"

    normalized = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)
    height, width = normalized.shape
    contours, _ = cv2.findContours(normalized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    parts = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue

        points = []
        for x, y in approx:
            points.append(max(0, min(999, int(x / width * 999))))
            points.append(max(0, min(999, int(y / height * 999))))
        parts.append("<poly>" + "".join(f"<loc_{value}>" for value in points) + "</poly>")

    return "".join(parts) if parts else "no_change"


def build_levir_samples(split: str, cfg: Config = CFG):
    a_dir = Path(cfg.LEVIR_ROOT) / split / "A"
    b_dir = Path(cfg.LEVIR_ROOT) / split / "B"
    label_dir = Path(cfg.LEVIR_ROOT) / split / "label"
    files = sorted([file.name for file in a_dir.glob("*.png")])

    samples = []
    for filename in files:
        image_t1 = a_dir / filename
        image_t2 = b_dir / filename
        label = label_dir / filename
        if not (image_t2.exists() and label.exists()):
            continue

        mask = np.array(Image.open(label).convert("L"))
        info = analyze_change_mask(mask)
        height, width = info["H"], info["W"]

        samples.append(
            {
                "id": f"levir_{split}_{filename}_det",
                "dataset": "levir",
                "image_t1": str(image_t1),
                "image_t2": str(image_t2),
                "mask": str(label),
                "task": "CHANGE_DETECTION",
                "instruction": "<CHANGE_DETECTION> Locate all changed regions.",
                "response": build_detection_response(info, width, height),
            }
        )
        samples.append(
            {
                "id": f"levir_{split}_{filename}_seg",
                "dataset": "levir",
                "image_t1": str(image_t1),
                "image_t2": str(image_t2),
                "mask": str(label),
                "task": "CHANGE_SEGMENTATION",
                "instruction": "<CHANGE_SEGMENTATION> Segment all changed regions.",
                "response": build_segmentation_response(mask, info),
            }
        )
    return samples
