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
NUM_CLASSES = len(CLASS_NAMES)

def load_images_from_dir(directory, size=(224,224), rescale=True):
    imgs = []
    paths = []
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.pgm', '.ppm', '.tif', '.tiff')):
            p = os.path.join(directory, f)
            try:
                img = Image.open(p).convert('RGB').resize(size, Image.Resampling.LANCZOS)
                arr = np.array(img)
                if rescale:
                    arr = arr / 255.0
                imgs.append(arr)
                paths.append(p)
            except Exception as e:
                print(f"Skip {p}: {e}")
    return np.array(imgs, dtype=np.float32), paths

def entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)

DATASETS = {
    "JAFFE": "/tmp/cross_datasets_preprocessed/jaffe_flat/images",
    "ORL": "/tmp/cross_datasets_preprocessed/orl_flat/images",
}

# Fetch LFW via sklearn and save as images
LFW_DIR = "/tmp/cross_datasets_preprocessed/lfw_flat/images"
os.makedirs(LFW_DIR, exist_ok=True)
if len([f for f in os.listdir(LFW_DIR) if f.endswith('.jpg')]) == 0:
    print("Fetching LFW dataset via sklearn...")
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(min_faces_per_person=1, resize=1.0, color=True, download_if_missing=True)
    # lfw.images shape: (n_samples, h, w, 3) or (n_samples, h, w) depending on color param
    # Actually fetch_lfw_people color=True returns images in shape (n_samples, 62, 47, 3)
    for i in range(lfw.images.shape[0]):
        img_arr = lfw.images[i]
        # Ensure uint8
        if img_arr.max() <= 1.0:
            img_arr = (img_arr * 255).astype(np.uint8)
        else:
            img_arr = img_arr.astype(np.uint8)
        img = Image.fromarray(img_arr).resize((224,224), Image.Resampling.LANCZOS)
        img.save(os.path.join(LFW_DIR, f"lfw_{i:06d}.jpg"))
    print(f"Saved {lfw.images.shape[0]} LFW images to {LFW_DIR}")
DATASETS["LFW"] = LFW_DIR

results = []
predictions = []

for model_name, path in [("EfficientNetB0", "models/best_efficientnetb0_finetune.keras"), ("ResNet50", "models/best_resnet50_finetune.keras")]:
    print(f"\nLoading {model_name}...")
    model = keras.models.load_model(path)
    rescale = False  # preprocess_input is baked into the model graph
    for ds_name, ds_path in DATASETS.items():
        imgs_arr, paths_list = load_images_from_dir(ds_path, rescale=rescale)
        if len(imgs_arr) == 0:
            print(f"  {ds_name}: no images found")
            continue
        print(f"  Evaluating on {ds_name} ({len(imgs_arr)} images)...")
        preds = model.predict(imgs_arr, verbose=0)
        confs = np.max(preds, axis=1)
        ents = entropy(preds)
        pred_classes = np.argmax(preds, axis=1)
        dist = {c: float(np.mean(pred_classes == i)) for i, c in enumerate(CLASS_NAMES)}
        mean_conf = float(np.mean(confs))
        results.append({"Dataset": ds_name, "Model": model_name, "Images": len(imgs_arr), "MeanConfidence": mean_conf, **dist})
        for p, pc, conf, ent in zip(paths_list, pred_classes, confs, ents):
            predictions.append({
                "dataset": ds_name,
                "model": model_name,
                "image_path": p,
                "pred_class": CLASS_NAMES[pc],
                "confidence": float(conf),
                "entropy": float(ent)
            })
        # Plot distribution
        plt.figure(figsize=(10,5))
        plt.bar(range(len(CLASS_NAMES)), [dist[c] for c in CLASS_NAMES], color='steelblue')
        plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha='right')
        plt.title(f"Распределение предсказаний — {ds_name} ({model_name})")
        plt.ylabel("Доля предсказаний")
        plt.ylim(0, 1)
        plt.tight_layout()
        out_png = f"/home/solodennikov.d/vadim_project/Master_VKR/figures/cross_dataset/cd_dist_{ds_name}_{model_name}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"    Saved plot: {out_png}")
    del model
    keras.backend.clear_session()

# Save results
if results:
    res_df = pd.DataFrame(results)
    res_df.to_csv("/home/solodennikov.d/vadim_project/Master_VKR/data/results/cross_dataset_results_orl_jaffe.csv", index=False)
    print("\nSaved cross_dataset_results_orl_jaffe.csv")

if predictions:
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv("/home/solodennikov.d/vadim_project/Master_VKR/data/results/cross_dataset_predictions_orl_jaffe.csv", index=False)
    print("Saved cross_dataset_predictions_orl_jaffe.csv")

print("Done.")
