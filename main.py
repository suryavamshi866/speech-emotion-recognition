import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings

warnings.filterwarnings('ignore')

# --- Load dataset ---
paths = []
labels = []

for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        # ✅ Skip system or non-audio files
        if not filename.lower().endswith(('.wav', '.mp3')):
            continue

        paths.append(os.path.join(dirname, filename))

        # ✅ Extract label safely
        label = filename.split('_')[-1].split('.')[0].lower()
        labels.append(label)

    # ✅ Optional limit (remove if full dataset)
    if len(paths) >= 2800:
        break

print('🎯 Dataset Loaded')
print(f"Total audio files found: {len(paths)}")
print(f"Example path: {paths[0] if paths else 'No files found'}")
print(f"Example label: {labels[0] if labels else 'No labels found'}")

if len(paths) == 0:
    print("⚠️ No dataset found. Please check your dataset path.")
else:
    print("✅ Dataset loaded successfully!")

# --- Return helper ---
def get_dataset():
    return paths, labels
