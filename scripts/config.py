from pathlib import Path

## Project's directions

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
TRAIN_DIR = PROJECT_ROOT / "dataset" / "processed" / "train"
VAL_DIR = PROJECT_ROOT / "dataset" / "processed" / "val"
TEST_DIR = PROJECT_ROOT / "dataset" / "processed" / "test"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok= True, parents= True)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok= True, parents= True)

# Predict
DEMO_DIR = PROJECT_ROOT / "demo"

PREDICTION_DIR = OUTPUT_DIR / "predictions"
PREDICTION_DIR.mkdir(exist_ok=True, parents=True)

# Preprocess
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


## Independent experiments

EXPERIMENTS = ["no_augmentation", "augmentation"]

def transfer_model_path(experiment_name):
    return MODEL_DIR / f"mobilenetv2_{experiment_name}_transfer.keras"
 
 
def finetune_model_path(experiment_name):
    return MODEL_DIR / f"mobilenetv2_{experiment_name}_finetune.keras"
 
 
def history_csv_path(experiment_name):
    return OUTPUT_DIR / f"history_{experiment_name}.csv"

## Image and Training parameters

IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 16

## Classes in STB
CLASS_NAMES = ["Metal", "Paper", "Plastic"]

NUM_CLASSES = len(CLASS_NAMES)

## Training parameters

INITIAL_EPOCHS = 30
FINETUNE_EPOCHS = 20
FINETUNE_AT_LAYER = -15

LEARNING_RATE = 1e-4
FINETUNE_LEARNING_RATE = 1e-5