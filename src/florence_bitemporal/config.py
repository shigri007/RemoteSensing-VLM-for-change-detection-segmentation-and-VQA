from dataclasses import dataclass


@dataclass
class Config:
    LEVIR_ROOT: str = "/2023220040/vqa_vlm/levir"
    CDVQA_ROOT: str = "/2023220040/vqa_vlm/levir/cdvqa/CDVQA-main"
    SECOND_TRAIN: str = "/2023220040/vqa_vlm/second-ds/SECOND_train_set"
    SECOND_TEST: str = "/2023220040/vqa_vlm/second-ds/SECOND_total_test/test"
    FLORENCE_PATH: str = "/2023220040/vlm_ungli/contribution2/florence/florence"
    OUTPUT_DIR: str = "/2023220040/vqa_vlm/levir/outputs"

    IMG_SIZE: int = 768
    MAX_TEXT_LEN: int = 512

    BATCH_SIZE: int = 4
    GRAD_ACCUM_STEPS: int = 4
    NUM_EPOCHS: int = 8
    LR: float = 2e-5
    LR_TFM: float = 1e-4
    LR_PROJ: float = 5e-5
    LR_VISION: float = 4e-6
    LR_EMBED: float = 1e-4
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.05
    NUM_WORKERS: int = 4
    SEED: int = 42

    LOSS_W_DET: float = 1.0
    LOSS_W_SEG: float = 1.0
    LOSS_W_VQA: float = 0.5

    LORA_RANK: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    LORA_TARGETS: tuple = ("q_proj", "k_proj", "v_proj", "out_proj")

    UNFREEZE_VISION_LAST_BLOCK: bool = True

    SANITY_SAMPLES_PER_TASK: int = 4
    SANITY_EPOCHS: int = 30

    EVAL_EVERY_EPOCH: bool = True
    SAVE_VIS_N: int = 30
    GEN_NUM_BEAMS_SPATIAL: int = 4
    GEN_NUM_BEAMS_VQA: int = 2
    GEN_REPETITION_PENALTY: float = 1.15
    GEN_LENGTH_PENALTY: float = 1.0
    GEN_NO_REPEAT_NGRAM: int = 4

    PRINT_SAMPLES_PER_TASK: int = 2

    TASK_TOKENS: tuple = (
        "<CHANGE_VQA>",
        "<CHANGE_DETECTION>",
        "<CHANGE_SEGMENTATION>",
        "<poly>",
        "</poly>",
    )

    CDVQA_ANSWERS: tuple = (
        "yes",
        "no",
        "buildings",
        "trees",
        "low_vegetation",
        "NVG_surface",
        "water",
        "playgrounds",
        "0",
        "0_to_10",
        "10_to_20",
        "20_to_30",
        "30_to_40",
        "40_to_50",
        "50_to_60",
        "60_to_70",
        "70_to_80",
        "80_to_90",
        "90_to_100",
    )


CFG = Config()
