import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

CLASS_NAMES = ["extraversion", "introversion", "agreeableness", "conscientiousness",
               "openness", "emotional_stability", "trust", "charisma",
               "self_confidence", "fashion_style", "hairstyle_pose"]

src = "/tmp/cross_datasets/jaffe_raw/jaffe"
dst = "/tmp/cross_datasets/jaffe_224"
os.makedirs(dst, exist_ok=True)
files = [f for f in os.listdir(src) if f.lower().endswith(('.tiff','.tif','.png','.jpg'))]
for f in files:
    try:
        img = Image.open(os.path.join(src, f)).convert('RGB')
        img = img.resize((224,224), Image.Resampling.LANCZOS)
        img.save(os.path.join(dst, os.path.splitext(f)[0]+".jpg"))
    except Exception as e:
        print("skip", f, e)
print(f"JAFFE prepared: {len([x for x in os.listdir(dst) if x.endswith('.jpg')])} images")

results = []
for model_name, path in [("EfficientNetB0", "best_efficientnetb0_finetune.keras"), ("ResNet50", "best_resnet50_finetune.keras")]:
    print(f"Evaluating {model_name} on JAFFE...")
    model = keras.models.load_model(path)
    imgs = []
    for f in sorted(os.listdir(dst)):
        if not f.endswith('.jpg'): continue
        img = Image.open(os.path.join(dst, f)).convert('RGB')
        imgs.append(np.array(img))
    batch = np.stack(imgs)
    preds = model.predict(batch, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    dist = {c: float(np.mean(pred_classes == i)) for i, c in enumerate(CLASS_NAMES)}
    mean_conf = float(np.mean(confidences))
    results.append({"Dataset": "JAFFE", "Model": model_name, "Images": len(imgs), "MeanConfidence": mean_conf, **dist})
    plt.figure(figsize=(8,4))
    plt.bar(range(len(CLASS_NAMES)), [dist[c] for c in CLASS_NAMES])
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha='right')
    plt.title(f"Распределение предсказаний на JAFFE — {model_name}")
    plt.ylabel("Доля предсказаний")
    plt.tight_layout()
    plt.savefig(f"jaffe_dist_{model_name}.png", dpi=150)
    plt.close()
    del model
    keras.backend.clear_session()

pd.DataFrame(results).to_csv("cross_dataset_results.csv", index=False)
print("Cross-dataset evaluation done.")
