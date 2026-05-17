import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter

# Paths
CELEBA_ROOT = "/home/solodennikov.d/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2"
IMG_DIR = os.path.join(CELEBA_ROOT, "img_align_celeba/img_align_celeba")
ATTR_PATH = os.path.join(CELEBA_ROOT, "list_attr_celeba.csv")
OUT_ROOT = "/tmp/vkr_dataset"

# Read attributes
attr_df = pd.read_csv(ATTR_PATH)
attr_df.columns = [c.strip() for c in attr_df.columns]

# Mapping of classes to attribute conditions (score based)
# Each class gets images where the score is highest
class_rules = {
    "extraversion": {"Smiling": 1, "Attractive": 1, "Mouth_Slightly_Open": 1},
    "introversion": {"Smiling": -1, "Attractive": -1, "Mouth_Slightly_Open": -1},
    "agreeableness": {"Smiling": 1, "Attractive": 1, "Young": 1},
    "conscientiousness": {"Male": 1, "Eyeglasses": 1, "Straight_Hair": 1},
    "openness": {"Wearing_Hat": 1, "Wearing_Earrings": 1, "Wavy_Hair": 1},
    "emotional_stability": {"No_Beard": 1, "Young": 1, "Oval_Face": 1},
    "trust": {"Smiling": 1, "Attractive": 1, "No_Beard": 1},
    "charisma": {"Attractive": 1, "Smiling": 1, "High_Cheekbones": 1},
    "self_confidence": {"Male": 1, "Attractive": 1, "High_Cheekbones": 1, "Wearing_Necktie": 1},
    "fashion_style": {"Heavy_Makeup": 1, "Wearing_Lipstick": 1, "Wearing_Earrings": 1, "Wearing_Necklace": 1},
    "hairstyle_pose": {"Bangs": 1, "Wavy_Hair": 1, "Straight_Hair": 1, "Receding_Hairline": 1},
}

def compute_score(row, rules):
    score = 0
    for attr, val in rules.items():
        if attr in row:
            score += 1 if row[attr] == val else -1
    return score

# Compute scores for each image per class
scores = {cls: [] for cls in class_rules}
for _, row in attr_df.iterrows():
    img_id = row["image_id"]
    for cls, rules in class_rules.items():
        scores[cls].append((img_id, compute_score(row, rules)))

# For each class, pick top-N images with highest score, ensuring minimal overlap
TARGET_PER_CLASS = 2500  # 2500 train + some for test
TRAIN_RATIO = 0.8
IMG_SIZE = (224, 224)

used_images = set()
class_images = {}

for cls in class_rules:
    sorted_imgs = sorted(scores[cls], key=lambda x: x[1], reverse=True)
    selected = []
    for img_id, sc in sorted_imgs:
        if img_id in used_images:
            continue
        if len(selected) < TARGET_PER_CLASS:
            selected.append(img_id)
            used_images.add(img_id)
    class_images[cls] = selected
    print(f"Class {cls}: selected {len(selected)} images, max score {sorted_imgs[0][1]}")

# Copy and resize images
for cls, imgs in class_images.items():
    np.random.seed(42)
    np.random.shuffle(imgs)
    split = int(len(imgs) * TRAIN_RATIO)
    train_imgs = imgs[:split]
    test_imgs = imgs[split:]
    
    for split_name, split_imgs in [("train", train_imgs), ("test", test_imgs)]:
        out_dir = os.path.join(OUT_ROOT, split_name, cls)
        os.makedirs(out_dir, exist_ok=True)
        for img_id in split_imgs:
            src = os.path.join(IMG_DIR, img_id)
            dst = os.path.join(out_dir, img_id)
            if os.path.exists(src):
                # Resize to target size
                img = Image.open(src).convert("RGB")
                img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                img.save(dst)

print("Dataset preparation complete!")
print("Image size:", IMG_SIZE)
for cls in class_rules:
    train_count = len(os.listdir(os.path.join(OUT_ROOT, "train", cls)))
    test_count = len(os.listdir(os.path.join(OUT_ROOT, "test", cls)))
    print(f"{cls}: train={train_count}, test={test_count}")
