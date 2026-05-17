import os
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_PATH = "/tmp/vkr_dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "test")
CLASS_NAMES = ["extraversion", "introversion", "agreeableness", "conscientiousness",
               "openness", "emotional_stability", "trust", "charisma",
               "self_confidence", "fashion_style", "hairstyle_pose"]
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# Custom CNN data generator with landmarks
class LandmarkGen(keras.utils.Sequence):
    def __init__(self, directory, batch_size, target_size, shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.samples = []
        self.labels = []
        self.landmarks = []
        for cls_idx, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(directory, cls)
            if not os.path.isdir(cls_dir): continue
            npy = os.path.join(cls_dir, 'landmarks.npy')
            txt = os.path.join(cls_dir, 'image_names.txt')
            if not os.path.exists(npy) or not os.path.exists(txt): continue
            with open(txt) as f:
                names = [l.strip() for l in f if l.strip()]
            lands = np.load(npy)
            for n, l in zip(names, lands):
                self.samples.append(os.path.join(cls_dir, n))
                self.labels.append(cls_idx)
                l_norm = l.copy()
                l_norm[0::2] /= 178.0
                l_norm[1::2] /= 218.0
                self.landmarks.append(l_norm)
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        self.landmarks = np.array(self.landmarks, dtype=np.float32)
        self.indexes = np.arange(len(self.samples))
        if shuffle: np.random.shuffle(self.indexes)
    def __len__(self): return int(np.ceil(len(self.samples)/self.batch_size))
    def __getitem__(self, idx):
        bi = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        imgs = []
        for i in bi:
            img = keras.preprocessing.image.load_img(self.samples[i], target_size=self.target_size)
            imgs.append(keras.preprocessing.image.img_to_array(img)/255.0)
        imgs = np.array(imgs)
        labs = self.labels[bi]
        lands = self.landmarks[bi]
        return (imgs, lands), keras.utils.to_categorical(labs, NUM_CLASSES)
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)

train_gen = LandmarkGen(TRAIN_DIR, BATCH_SIZE, IMG_SIZE, shuffle=True)
val_gen = LandmarkGen(TEST_DIR, BATCH_SIZE, IMG_SIZE, shuffle=False)

# Class weights
weights = compute_class_weight('balanced', classes=np.unique(train_gen.labels), y=train_gen.labels)
class_weights = {i: weights[i] for i in range(len(weights))}

# Callbacks helper
def get_callbacks(name):
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f'best_{name}.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

# SE block and Residual block
from tensorflow.keras import layers

def se_block(inputs, reduction=16):
    f = inputs.shape[-1]
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Dense(f//reduction, activation='relu')(se)
    se = layers.Dense(f, activation='sigmoid')(se)
    se = layers.Reshape((1,1,f))(se)
    return layers.multiply([inputs, se])

def res_block(x, filters, stride=1, dropout=0.25):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = se_block(x)
    if stride!=1 or shortcut.shape[-1]!=filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut]); x = layers.ReLU()(x); x = layers.Dropout(dropout)(x)
    return x

def build_custom():
    img = layers.Input((224,224,3), name='image_input')
    x = layers.Conv2D(64,7,strides=2,padding='same',use_bias=False)(img)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3,strides=2,padding='same')(x)
    x = res_block(x,64,dropout=0.2); x = res_block(x,64,dropout=0.2)
    x = res_block(x,128,stride=2,dropout=0.3); x = res_block(x,128,dropout=0.3)
    x = res_block(x,256,stride=2,dropout=0.35); x = res_block(x,256,dropout=0.35)
    x = res_block(x,512,stride=2,dropout=0.4); x = res_block(x,512,dropout=0.4)
    x = layers.GlobalAveragePooling2D()(x)
    land = layers.Input((10,), name='landmark_input')
    l = layers.Dense(64, activation='relu')(land); l = layers.BatchNormalization()(l)
    l = layers.Dense(32, activation='relu')(l); l = layers.BatchNormalization()(l)
    fused = layers.concatenate([x,l])
    fused = layers.Dense(512, activation='relu')(fused); fused = layers.BatchNormalization()(fused); fused = layers.Dropout(0.5)(fused)
    fused = layers.Dense(256, activation='relu')(fused); fused = layers.BatchNormalization()(fused); fused = layers.Dropout(0.4)(fused)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(fused)
    model = Model(inputs=[img,land], outputs=out)
    model.compile(Adam(1e-4), 'categorical_crossentropy', metrics=['accuracy'])
    return model

print("Training Custom CNN...")
custom = build_custom()
hist_custom = custom.fit(train_gen, validation_data=val_gen, epochs=15, class_weight=class_weights, callbacks=get_callbacks('custom_cnn'), verbose=1)
json.dump(hist_custom.history, open('history_custom.json','w'))
custom.save('final_custom_cnn_model.keras')

def plot_hist(h, name, fname):
    epochs = range(1, len(h['accuracy'])+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, h['accuracy'], 'b-', label='Train')
    plt.plot(epochs, h['val_accuracy'], 'r-', label='Val')
    plt.title(f'Accuracy — {name}'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(epochs, h['loss'], 'b-', label='Train')
    plt.plot(epochs, h['val_loss'], 'r-', label='Val')
    plt.title(f'Loss — {name}'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

plot_hist(hist_custom.history, 'Custom CNN', 'plot_custom_cnn.png')
print("Custom CNN done.")

# Transfer learning data generators WITHOUT rescale, using preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess

# EfficientNetB0
def build_eff():
    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
    base.trainable = False
    inp = layers.Input((224,224,3))
    x = eff_preprocess(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.35)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inp, out, name='EfficientNetB0')
    model.compile(Adam(1e-4), 'categorical_crossentropy', metrics=['accuracy'])
    return model, base

# Since we use preprocess_input inside model, generator should NOT rescale
train_gen_eff = ImageDataGenerator(rotation_range=25, width_shift_range=0.2, height_shift_range=0.2,
                                    zoom_range=0.2, shear_range=0.15, horizontal_flip=True).flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=SEED)
val_gen_eff = ImageDataGenerator().flow_from_directory(TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                        class_mode='categorical', shuffle=False)

print("Training EfficientNetB0 initial...")
eff, eff_base = build_eff()
hist_eff_i = eff.fit(train_gen_eff, validation_data=val_gen_eff, epochs=15, class_weight=class_weights,
                     callbacks=get_callbacks('efficientnetb0_initial'), verbose=1)
json.dump(hist_eff_i.history, open('history_eff_initial.json','w'))
plot_hist(hist_eff_i.history, 'EfficientNetB0 Initial', 'plot_eff_initial.png')

# Fine-tune EfficientNetB0
print("Fine-tuning EfficientNetB0...")
eff_base.trainable = True
for layer in eff_base.layers[:-30]: layer.trainable = False
eff.compile(Adam(1e-5), 'categorical_crossentropy', metrics=['accuracy'])
hist_eff_f = eff.fit(train_gen_eff, validation_data=val_gen_eff, epochs=10, class_weight=class_weights,
                     callbacks=get_callbacks('efficientnetb0_finetune'), verbose=1)
json.dump(hist_eff_f.history, open('history_eff_finetune.json','w'))
plot_hist(hist_eff_f.history, 'EfficientNetB0 Fine-tune', 'plot_eff_finetune.png')
eff.save('final_efficientnetb0_model.keras')
print("EfficientNetB0 done.")

# ResNet50
def build_res():
    base = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
    base.trainable = False
    inp = layers.Input((224,224,3))
    x = res_preprocess(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inp, out, name='ResNet50')
    model.compile(Adam(1e-4), 'categorical_crossentropy', metrics=['accuracy'])
    return model, base

train_gen_res = ImageDataGenerator(rotation_range=25, width_shift_range=0.2, height_shift_range=0.2,
                                    zoom_range=0.2, shear_range=0.15, horizontal_flip=True).flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=SEED)
val_gen_res = ImageDataGenerator().flow_from_directory(TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                        class_mode='categorical', shuffle=False)

print("Training ResNet50 initial...")
res, res_base = build_res()
hist_res_i = res.fit(train_gen_res, validation_data=val_gen_res, epochs=15, class_weight=class_weights,
                     callbacks=get_callbacks('resnet50_initial'), verbose=1)
json.dump(hist_res_i.history, open('history_res_initial.json','w'))
plot_hist(hist_res_i.history, 'ResNet50 Initial', 'plot_res_initial.png')

# Fine-tune ResNet50
print("Fine-tuning ResNet50...")
res_base.trainable = True
for layer in res_base.layers[:-40]: layer.trainable = False
res.compile(Adam(1e-5), 'categorical_crossentropy', metrics=['accuracy'])
hist_res_f = res.fit(train_gen_res, validation_data=val_gen_res, epochs=10, class_weight=class_weights,
                     callbacks=get_callbacks('resnet50_finetune'), verbose=1)
json.dump(hist_res_f.history, open('history_res_finetune.json','w'))
plot_hist(hist_res_f.history, 'ResNet50 Fine-tune', 'plot_res_finetune.png')
res.save('final_resnet50_model.keras')
print("ResNet50 done.")
print("All training completed successfully!")
