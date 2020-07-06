from pathlib import Path

PROJECT_DIR = Path("/data4/selfsupervision/")
PROJECT_DATA_DIR = PROJECT_DIR / "chexpert"
CHEXPERT_DATA_DIR = PROJECT_DATA_DIR / "CheXpert" 
CHEXPERT_V1_DIR = CHEXPERT_DATA_DIR / "CheXpert-v1.0"

CHEXPERT_TRAIN_CSV = CHEXPERT_V1_DIR / "train.csv"
CHEXPERT_VALID_CSV = CHEXPERT_V1_DIR / "valid.csv"

CHEXPERT_TRAIN_DATA = CHEXPERT_V1_DIR / "train"
CHEXPERT_VALID_DATA = CHEXPERT_V1_DIR / "valid"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CHEXPERT_TASKS = ["No Finding",
                  "Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Pneumonia",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Pleural Other",
                  "Fracture",
                  "Support Devices"]
                  
CHEXPERT_COMPETITION_TASKS = ["Atelectasis",
                              "Cardiomegaly",
                              "Consolidation",
                              "Edema",
                              "Pleural Effusion"]