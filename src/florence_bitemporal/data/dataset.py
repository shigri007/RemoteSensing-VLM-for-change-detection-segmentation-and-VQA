import json
from collections import Counter

from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from ..utils import log


class MultiTaskDataset(Dataset):
    def __init__(self, json_path, processor, img_size):
        with open(json_path) as file:
            self.samples = json.load(file)
        self.proc = processor
        self.S = img_size

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        image = Image.open(path).convert("RGB")
        if image.size != (self.S, self.S):
            image = image.resize((self.S, self.S), Image.BILINEAR)
        return image

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "img_t1": self._load(sample["image_t1"]),
            "img_t2": self._load(sample["image_t2"]),
            "instruction": sample["instruction"],
            "response": sample["response"],
            "task": sample["task"],
            "id": sample["id"],
            "mask_path": sample.get("mask", ""),
            "dataset": sample.get("dataset", ""),
            "qtype": sample.get("qtype", ""),
        }


def make_collate(processor, max_text_len=512):
    def collate(batch):
        t1_inputs = processor.image_processor(
            images=[item["img_t1"] for item in batch],
            return_tensors="pt",
        )
        t2_inputs = processor.image_processor(
            images=[item["img_t2"] for item in batch],
            return_tensors="pt",
        )
        instructions = [item["instruction"] for item in batch]
        responses = [item["response"] for item in batch]

        input_tokens = processor.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        label_tokens = processor.tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )

        labels = label_tokens.input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values_t1": t1_inputs["pixel_values"],
            "pixel_values_t2": t2_inputs["pixel_values"],
            "input_ids": input_tokens.input_ids,
            "attention_mask": input_tokens.attention_mask,
            "labels": labels,
            "tasks": [item["task"] for item in batch],
            "ids": [item["id"] for item in batch],
            "instructions": instructions,
            "responses": responses,
            "mask_paths": [item["mask_path"] for item in batch],
            "datasets": [item["dataset"] for item in batch],
            "qtypes": [item["qtype"] for item in batch],
        }

    return collate


def make_balanced_sampler(samples):
    task_counts = Counter(sample["task"] for sample in samples)
    log(f"  Task distribution: {dict(task_counts)}")
    weights = []
    for sample in samples:
        weights.append(1.0 / max(task_counts[sample["task"]], 1))
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
