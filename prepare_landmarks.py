import os
import pandas as pd
import numpy as np

CELEBA_ROOT = "/home/solodennikov.d/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2"
LANDMARKS_PATH = os.path.join(CELEBA_ROOT, "list_landmarks_align_celeba.csv")
OUT_ROOT = "/tmp/vkr_dataset"

landmarks_df = pd.read_csv(LANDMARKS_PATH)
landmarks_df.columns = [c.strip() for c in landmarks_df.columns]

landmark_cols = ['lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y',
                 'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y',
                 'rightmouth_x', 'rightmouth_y']

# Normalize landmarks to [0,1] based on CelebA aligned size (178x218 usually, but aligned to 178x178)
# Actually aligned images are 178x218 originally, but we resized to 224x224
# Let's just store raw and normalize per-image during training
landmark_dict = {}
for _, row in landmarks_df.iterrows():
    img_id = row['image_id']
    vals = [float(row[c]) for c in landmark_cols]
    landmark_dict[img_id] = np.array(vals, dtype=np.float32)

# Save for each split/class
for split in ['train', 'test']:
    split_dir = os.path.join(OUT_ROOT, split)
    for cls in os.listdir(split_dir):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        all_landmarks = []
        all_names = []
        for img_name in sorted(os.listdir(cls_dir)):
            if img_name in landmark_dict:
                all_landmarks.append(landmark_dict[img_name])
                all_names.append(img_name)
        if all_landmarks:
            np.save(os.path.join(cls_dir, 'landmarks.npy'), np.stack(all_landmarks))
            with open(os.path.join(cls_dir, 'image_names.txt'), 'w') as f:
                f.write('\n'.join(all_names))
            print(f"Saved landmarks for {split}/{cls}: {len(all_landmarks)}")

print("Landmark preparation done.")
