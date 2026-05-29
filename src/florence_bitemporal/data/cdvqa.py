import json
from pathlib import Path

from ..config import CFG, Config
from ..utils import log


def unwrap_cdvqa(data, expected_key):
    if isinstance(data, dict):
        if expected_key in data and isinstance(data[expected_key], list):
            return data[expected_key]
        for value in data.values():
            if isinstance(value, list):
                return value
        return []
    if isinstance(data, list):
        return data
    return []


def normalize_question_text(question):
    if isinstance(question, str):
        return question
    if isinstance(question, dict):
        for key in ("question", "text", "value", "name"):
            if key in question and isinstance(question[key], str):
                return question[key]
    return str(question)


def classify_question_type(answer: str) -> str:
    normalized = answer.strip().lower()
    if normalized in ("yes", "no"):
        return "yes_no"

    land_cover = {
        "buildings",
        "trees",
        "low_vegetation",
        "nvg_surface",
        "water",
        "playgrounds",
    }
    if normalized in land_cover:
        return "land_cover"
    if "_to_" in normalized or normalized == "0":
        return "ratio"
    return "other"


def resolve_image_path(filename: str, cfg: Config = CFG):
    for base in (cfg.SECOND_TRAIN, cfg.SECOND_TEST):
        image_t1 = Path(base) / "im1" / filename
        image_t2 = Path(base) / "im2" / filename
        if image_t1.exists() and image_t2.exists():
            return str(image_t1), str(image_t2)
    return None


def build_cdvqa_samples(split: str, cfg: Config = CFG):
    base = Path(cfg.CDVQA_ROOT)
    cap = "Test2" if split == "test2" else split.capitalize()

    with open(base / f"{cap}_questions.json") as file:
        questions = unwrap_cdvqa(json.load(file), "questions")
    with open(base / f"{cap}_answers.json") as file:
        answers = unwrap_cdvqa(json.load(file), "answers")
    with open(base / f"{cap}_images.json") as file:
        images = unwrap_cdvqa(json.load(file), "images")

    questions_by_id = {}
    for question in questions:
        if isinstance(question, dict):
            question_id = question.get("question_id", question.get("id"))
            questions_by_id[question_id] = question

    answers_by_question_id = {}
    for answer in answers:
        if isinstance(answer, dict):
            question_id = answer.get("question_id")
            answers_by_question_id[question_id] = answer.get("answer", "")

    samples = []
    skipped_no_image = 0
    skipped_no_qa = 0

    for image in images:
        if not isinstance(image, dict):
            continue
        filename = image.get("file_name")
        question_ids = image.get("questions_ids", [])
        if not filename:
            continue

        paths = resolve_image_path(filename, cfg)
        if paths is None:
            skipped_no_image += 1
            continue
        image_t1_path, image_t2_path = paths

        for question_id in question_ids:
            question = questions_by_id.get(question_id)
            answer = answers_by_question_id.get(question_id)
            if question is None or not answer:
                skipped_no_qa += 1
                continue

            question_text = normalize_question_text(question)
            question_type = classify_question_type(answer)

            samples.append(
                {
                    "id": f"cdvqa_{split}_{question_id}",
                    "dataset": "cdvqa",
                    "image_t1": image_t1_path,
                    "image_t2": image_t2_path,
                    "mask": "",
                    "task": "CHANGE_VQA",
                    "instruction": f"<CHANGE_VQA> {question_text}",
                    "response": answer,
                    "qtype": question_type,
                    "filename": filename,
                }
            )

    log(
        f"  CDVQA {split}: {len(samples)} samples  "
        f"(skipped {skipped_no_image} no-image, {skipped_no_qa} no-QA)"
    )
    return samples
