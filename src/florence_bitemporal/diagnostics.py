import json
from collections import Counter
from pathlib import Path

from .config import CFG, Config
from .data.cdvqa import unwrap_cdvqa
from .utils import ensure_dir, log


def run_diagnostic(cfg: Config = CFG):
    log("=" * 70)
    log("                DIAGNOSTIC MODE")
    log("=" * 70)

    issues = []

    log("\n[1] Checking LEVIR-CD ...")
    for split in ("train", "val", "test"):
        for subdir in ("A", "B", "label"):
            path = Path(cfg.LEVIR_ROOT) / split / subdir
            if not path.exists():
                issues.append(f"LEVIR missing: {path}")
                log(f"  X {path}")
            else:
                count = len(list(path.glob("*.png")))
                log(f"  + {path}  ({count} png files)")

    log("\n[2] Checking CDVQA JSONs ...")
    for split in ("Train", "Val", "Test", "Test2"):
        for kind in ("questions", "answers", "images"):
            path = Path(cfg.CDVQA_ROOT) / f"{split}_{kind}.json"
            if not path.exists():
                issues.append(f"CDVQA missing: {path}")
                log(f"  X {path}")
            else:
                with open(path) as file:
                    data = json.load(file)
                inner = unwrap_cdvqa(data, kind)
                log(f"  + {path.name}  (items={len(inner)})")

    log("\n[3] Inspecting CDVQA JSON keys ...")
    try:
        with open(Path(cfg.CDVQA_ROOT) / "Train_questions.json") as file:
            questions_raw = json.load(file)
        questions = unwrap_cdvqa(questions_raw, "questions")
        log(f"  Train_questions[0]: {questions[0] if questions else 'EMPTY'}")

        with open(Path(cfg.CDVQA_ROOT) / "Train_answers.json") as file:
            answers_raw = json.load(file)
        answers = unwrap_cdvqa(answers_raw, "answers")
        log(f"  Train_answers[0]: {answers[0] if answers else 'EMPTY'}")

        answer_values = [item.get("answer", "") for item in answers if isinstance(item, dict)]
        counts = Counter(answer_values)
        log(f"  Unique answers: {len(counts)}")
        log(f"  Top 20: {counts.most_common(20)}")

        unknown = [answer for answer in counts if answer not in cfg.CDVQA_ANSWERS]
        if unknown:
            log(f"  WARN: Answers not in CFG.CDVQA_ANSWERS: {unknown}")
    except Exception as exc:
        issues.append(f"CDVQA JSON parse error: {exc}")
        log(f"  X Error: {exc}")

    log("\n[4] Checking SECOND folders ...")
    for subdir in ("im1", "im2", "label1", "label2"):
        for base, name in [(cfg.SECOND_TRAIN, "train"), (cfg.SECOND_TEST, "test")]:
            path = Path(base) / subdir
            if not path.exists():
                issues.append(f"SECOND missing: {path}")
                log(f"  X {path}")
            else:
                count = len(list(path.glob("*.png")))
                log(f"  + {path}  ({count} png files)")

    log("\n[5] Cross-checking CDVQA filenames in SECOND ...")
    split_files = [
        ("Train", "Train_images.json"),
        ("Val", "Val_images.json"),
        ("Test", "Test_images.json"),
        ("Test2", "Test2_images.json"),
    ]
    for split, json_name in split_files:
        try:
            with open(Path(cfg.CDVQA_ROOT) / json_name) as file:
                data = unwrap_cdvqa(json.load(file), "images")

            filenames = set()
            for item in data:
                if isinstance(item, dict) and "file_name" in item:
                    filenames.add(item["file_name"])

            in_train = sum(
                1 for filename in filenames if (Path(cfg.SECOND_TRAIN) / "im1" / filename).exists()
            )
            in_test = sum(
                1 for filename in filenames if (Path(cfg.SECOND_TEST) / "im1" / filename).exists()
            )
            missing = len(filenames) - in_train - in_test
            log(
                f"  {split}: {len(filenames)} unique imgs  ->  "
                f"in_train={in_train}, in_test={in_test}, missing={missing}"
            )
            if missing > 0:
                issues.append(f"{split}: {missing} CDVQA images missing")
        except Exception as exc:
            log(f"  X {split}: {exc}")

    log("\n[6] Checking Florence-2 ...")
    florence_path = Path(cfg.FLORENCE_PATH)
    if not florence_path.exists():
        issues.append(f"Florence-2 missing: {florence_path}")
        log(f"  X {florence_path}")
    else:
        log(f"  + {florence_path}")

    log("\n[7] Checking output dir ...")
    try:
        ensure_dir(cfg.OUTPUT_DIR)
        test_path = Path(cfg.OUTPUT_DIR) / ".write_test"
        test_path.write_text("ok")
        test_path.unlink()
        log(f"  + {cfg.OUTPUT_DIR} writable")
    except Exception as exc:
        issues.append(f"Output dir error: {exc}")
        log(f"  X {exc}")

    log("\n" + "=" * 70)
    if issues:
        log(f"DIAGNOSTIC FOUND {len(issues)} ISSUE(S):")
        for index, message in enumerate(issues, 1):
            log(f"  {index}. {message}")
    else:
        log("DIAGNOSTIC PASSED.")
    log("=" * 70)
    return len(issues) == 0
