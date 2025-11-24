WANDB_ENTITY = "entity"
WANDB_PROJECT = "pretrain"

IN_RGB_MEAN = [0.485, 0.456, 0.406]
IN_RGB_STD = [0.229, 0.224, 0.225]

IN_RESENET_CHANNEL_MAP = {"red": 0, "green": 1, "blue": 2}

PROBING = "probing"
FINETUNING = "finetuning"
TRAIN_METHODS = [PROBING, FINETUNING]
