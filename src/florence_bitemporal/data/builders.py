import json
import random
from pathlib import Path

from ..config import CFG, Config
from ..utils import ensure_dir, log
from .cdvqa import build_cdvqa_samples
from .levir import build_levir_samples


def build_all_jsons(cfg: Config = CFG):
    json_dir = Path(cfg.OUTPUT_DIR) / "json"
    ensure_dir(json_dir)

    log("=== Building LEVIR-CD samples ===")
    levir_train = build_levir_samples("train", cfg)
    levir_val = build_levir_samples("val", cfg)
    levir_test = build_levir_samples("test", cfg)
    log(f"  LEVIR train={len(levir_train)} val={len(levir_val)} test={len(levir_test)}")

    log("=== Building CDVQA samples ===")
    cdvqa_train = build_cdvqa_samples("train", cfg)
    cdvqa_val = build_cdvqa_samples("val", cfg)
    cdvqa_test = build_cdvqa_samples("test", cfg)
    cdvqa_test2 = build_cdvqa_samples("test2", cfg)

    train_all = levir_train + cdvqa_train
    random.shuffle(train_all)
    val_all = levir_val + cdvqa_val

    splits = [
        ("train.json", train_all),
        ("val.json", val_all),
        ("levir_train.json", levir_train),
        ("levir_val.json", levir_val),
        ("levir_test.json", levir_test),
        ("cdvqa_train.json", cdvqa_train),
        ("cdvqa_val.json", cdvqa_val),
        ("cdvqa_test.json", cdvqa_test),
        ("cdvqa_test2.json", cdvqa_test2),
    ]
    for name, data in splits:
        with open(json_dir / name, "w") as file:
            json.dump(data, file, indent=2)
        log(f"  Wrote {name}: {len(data)}")

    return json_dir
