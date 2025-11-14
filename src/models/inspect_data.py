import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import get_settings

settings = get_settings()


# Paths
X_train = np.load(f"{settings.processed_data_dir}/X_train.npy")
y_train = np.load(f"{settings.processed_data_dir}/y_train.npy")
df_index = pd.read_csv(f"{settings.processed_data_dir}/train_index.csv")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Label distribution:", pd.Series(y_train).value_counts())
print("First images:")
for i in range(3):
    plt.imshow(X_train[i])
    plt.title(f"Label: {y_train[i]}")
    plt.show()
