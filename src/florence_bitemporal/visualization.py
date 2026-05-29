from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_comparison(
    t1_path,
    t2_path,
    gt_mask_path,
    pred_mask,
    pred_text,
    gt_text,
    instruction,
    out_path,
    dataset_tag="",
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    try:
        image_t1 = np.array(Image.open(t1_path).convert("RGB"))
        image_t2 = np.array(Image.open(t2_path).convert("RGB"))
        axes[0, 0].imshow(image_t1)
        axes[0, 0].set_title("T1 (before)")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(image_t2)
        axes[0, 1].set_title("T2 (after)")
        axes[0, 1].axis("off")
    except Exception as exc:
        for axis in axes[0]:
            axis.text(0.5, 0.5, f"err: {exc}", ha="center")
            axis.axis("off")

    if gt_mask_path and Path(gt_mask_path).exists():
        try:
            gt_mask = np.array(Image.open(gt_mask_path).convert("L"))
            axes[0, 2].imshow(gt_mask, cmap="gray")
            axes[0, 2].set_title("GT mask")
        except Exception:
            axes[0, 2].text(0.5, 0.5, "no mask", ha="center")
    else:
        axes[0, 2].text(0.5, 0.5, "(VQA - no mask)", ha="center")
    axes[0, 2].axis("off")

    if pred_mask is not None:
        axes[1, 0].imshow(pred_mask, cmap="gray")
        axes[1, 0].set_title("Pred mask")
        axes[1, 0].axis("off")
        try:
            image_t2 = np.array(Image.open(t2_path).convert("RGB"))
            overlay = image_t2.copy()
            red = np.zeros_like(overlay)
            red[..., 0] = 255
            mask_3d = np.repeat(pred_mask[..., None], 3, axis=-1).astype(bool)
            overlay = np.where(mask_3d, (0.5 * overlay + 0.5 * red).astype(np.uint8), overlay)
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title("Overlay")
            axes[1, 1].axis("off")
        except Exception:
            axes[1, 1].axis("off")
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    axes[1, 2].axis("off")
    text = (
        f"[{dataset_tag}]\nQ: {instruction}\n\n"
        f"PRED: {pred_text[:300]}\n\n"
        f"GT:   {gt_text[:300]}"
    )
    axes[1, 2].text(
        0.0,
        1.0,
        text,
        fontsize=9,
        va="top",
        family="monospace",
        wrap=True,
    )

    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
