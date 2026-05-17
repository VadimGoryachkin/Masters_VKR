import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

CLASS_NAMES_EN = ["extraversion", "introversion", "agreeableness", "conscientiousness",
               "openness", "emotional_stability", "trust", "charisma",
               "self_confidence", "fashion_style", "hairstyle_pose"]
CLASS_NAMES_RU = ["Экстраверсия", "Интроверсия", "Доброжелательность", "Ответственность",
               "Открытость", "Эмоц. стабильность", "Доверие", "Харизма",
               "Самоуверенность", "Стиль/мода", "Прическа/поза"]
NUM_CLASSES = len(CLASS_NAMES_EN)
TEST_DIR = "/tmp/vkr_dataset/test"

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        roc = roc_auc_score(label_binarize(y_true, classes=np.arange(NUM_CLASSES)), y_prob, average='macro', multi_class='ovr')
    except:
        roc = np.nan
    return acc, prec, rec, f1m, f1w, roc

def plot_cm(y_true, y_pred, title, fname):
    plt.figure(figsize=(12,10))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES_RU, yticklabels=CLASS_NAMES_RU)
    plt.title(title)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

results = []

# 1. Custom CNN with landmarks
print("Evaluating Custom CNN...")
class LandmarkImageDataGenerator(keras.utils.Sequence):
    def __init__(self, directory, batch_size, target_size, shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.samples = []
        self.labels = []
        self.landmarks = []
        for cls_idx, cls in enumerate(CLASS_NAMES_EN):
            cls_dir = os.path.join(directory, cls)
            if not os.path.isdir(cls_dir): continue
            names_path = os.path.join(cls_dir, 'image_names.txt')
            land_path = os.path.join(cls_dir, 'landmarks.npy')
            if not os.path.exists(names_path) or not os.path.exists(land_path): continue
            with open(names_path, 'r') as f:
                names = [l.strip() for l in f if l.strip()]
            lands = np.load(land_path)
            for n, l in zip(names, lands):
                self.samples.append(os.path.join(cls_dir, n))
                self.labels.append(cls_idx)
                l_norm = l.copy()
                l_norm[0::2] = l_norm[0::2] / 178.0
                l_norm[1::2] = l_norm[1::2] / 218.0
                self.landmarks.append(l_norm)
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        self.landmarks = np.array(self.landmarks, dtype=np.float32)
        self.indexes = np.arange(len(self.samples))
        if shuffle: np.random.shuffle(self.indexes)
    def __len__(self): return int(np.ceil(len(self.samples) / self.batch_size))
    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs = []
        for i in batch_idx:
            img = keras.preprocessing.image.load_img(self.samples[i], target_size=self.target_size)
            img = keras.preprocessing.image.img_to_array(img) / 255.0
            batch_imgs.append(img)
        batch_imgs = np.array(batch_imgs)
        batch_labels = self.labels[batch_idx]
        batch_landmarks = self.landmarks[batch_idx]
        batch_labels_cat = keras.utils.to_categorical(batch_labels, NUM_CLASSES)
        return (batch_imgs, batch_landmarks), batch_labels_cat

val_gen = LandmarkImageDataGenerator(TEST_DIR, 32, (224,224), shuffle=False)
custom_model = keras.models.load_model('best_custom_cnn.keras')
all_probs = []
all_trues = []
for i in range(len(val_gen)):
    (imgs, lands), labels = val_gen[i]
    probs = custom_model.predict((imgs, lands), verbose=0)
    all_probs.append(probs)
    all_trues.append(labels)
y_prob = np.concatenate(all_probs)
y_true_cat = np.concatenate(all_trues)
y_pred = np.argmax(y_prob, axis=1)
y_true = np.argmax(y_true_cat, axis=1)
acc, prec, rec, f1m, f1w, roc = compute_metrics(y_true, y_pred, y_prob)
print(f"Custom CNN: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1m={f1m:.4f}, F1w={f1w:.4f}, ROC={roc:.4f}")
plot_cm(y_true, y_pred, 'Матрица ошибок — Кастомная SE-ResNet + Landmarks', 'cm_Custom_SE-ResNet_Landmarks_ru.png')
results.append({"Model": "Custom SE-ResNet + Landmarks", "Accuracy": acc, "Macro F1-score": f1m, "Weighted F1-score": f1w, "ROC-AUC": roc})
keras.backend.clear_session()

# 2. EfficientNetB0
print("Evaluating EfficientNetB0...")
val_eff = ImageDataGenerator().flow_from_directory(TEST_DIR, target_size=(224,224), batch_size=32, class_mode='categorical', shuffle=False)
eff_model = keras.models.load_model('best_efficientnetb0_finetune.keras')
y_prob_eff = eff_model.predict(val_eff, verbose=1)
y_pred_eff = np.argmax(y_prob_eff, axis=1)
y_true_eff = val_eff.classes
acc, prec, rec, f1m, f1w, roc = compute_metrics(y_true_eff, y_pred_eff, y_prob_eff)
print(f"EfficientNetB0: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1m={f1m:.4f}, F1w={f1w:.4f}, ROC={roc:.4f}")
plot_cm(y_true_eff, y_pred_eff, 'Матрица ошибок — EfficientNetB0', 'cm_EfficientNetB0_ru.png')
results.append({"Model": "EfficientNetB0", "Accuracy": acc, "Macro F1-score": f1m, "Weighted F1-score": f1w, "ROC-AUC": roc})
keras.backend.clear_session()

# 3. ResNet50
print("Evaluating ResNet50...")
val_res = ImageDataGenerator().flow_from_directory(TEST_DIR, target_size=(224,224), batch_size=32, class_mode='categorical', shuffle=False)
res_model = keras.models.load_model('best_resnet50_finetune.keras')
y_prob_res = res_model.predict(val_res, verbose=1)
y_pred_res = np.argmax(y_prob_res, axis=1)
y_true_res = val_res.classes
acc, prec, rec, f1m, f1w, roc = compute_metrics(y_true_res, y_pred_res, y_prob_res)
print(f"ResNet50: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1m={f1m:.4f}, F1w={f1w:.4f}, ROC={roc:.4f}")
plot_cm(y_true_res, y_pred_res, 'Матрица ошибок — ResNet50', 'cm_ResNet50_ru.png')
results.append({"Model": "ResNet50", "Accuracy": acc, "Macro F1-score": f1m, "Weighted F1-score": f1w, "ROC-AUC": roc})
keras.backend.clear_session()

# Save CSV
df = pd.DataFrame(results)
df.to_csv('model_comparison_results.csv', index=False)
print("CSV updated with all 3 models.")
print(df)
