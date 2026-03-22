"""
Task 9: Model Training for S4 Galaxy Classification
====================================================
This script implements all TODO items from train.ipynb as a standalone
Python script, following the milestone document exactly:
- 64x64 images (original GalaxyMNIST)
- conv1d-based S4D (direct convolution, not FFT)
- Adam optimizer with lr=0.0015
- Batch size 64
- At least 10 epochs

Completes:
  - TODO: Set ERP ID seed
  - TODO: Plot training loss and validation accuracy
  - TODO: Load test data
  - TODO: Plot confusion matrix
  - Per-class metrics (precision, recall, F1)
  - Sample predictions grid
  - Error analysis
  - Model checkpoint saving
  - Model parameter export
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import csv
import random
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

# Our model
from model import GalaxyClassifierS4D
from model.functions import export_model_parameters, load_data


# ==============================================================================
# CONFIGURATION — Following train.ipynb defaults
# ==============================================================================
RNG_SEED = 42           # TODO: Replace with your ERP ID
COLORED = False         # False = grayscale (1 channel), True = RGB (3 channels)
BATCH_SIZE = 64         # From notebook Section 2.2
EPOCHS = 20             # "Change this to at least 10 for meaningful training"
LEARNING_RATE = 0.0015  # From notebook Section 5: Adam lr=0.0015
WEIGHT_DECAY = 1e-4     # Mild regularization
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Unbarred Spiral"]
IMAGES_DIR = "images"
MODEL_PARAMS_DIR = "model_params"

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODEL_PARAMS_DIR, exist_ok=True)


# ==============================================================================
# SEED EVERYTHING
# ==============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(RNG_SEED)

print(f"Using RNG seed: {RNG_SEED}")
print(f"Using device: {DEVICE}")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
print("=" * 60)


# ==============================================================================
# 1. LOAD AND PREPROCESS DATA
# ==============================================================================
print("\n[1/8] Loading training data...")
X, y_onehot, y = load_data(root="./data", download=True, train=True, colored=COLORED)
NUM_CLASSES = y_onehot.shape[1]

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  y_onehot shape: {y_onehot.shape}")
print(f"  Number of classes: {NUM_CLASSES}")


# ==============================================================================
# 2. SPLIT INTO TRAIN/VALIDATION AND CREATE DATALOADERS
# ==============================================================================
print("\n[2/8] Splitting data and creating DataLoaders...")

x_train, x_val, y_train_onehot, y_val_onehot = train_test_split(
    X, y_onehot, test_size=0.2, random_state=RNG_SEED, stratify=y
)

train_ds = TensorDataset(x_train, y_train_onehot)
val_ds = TensorDataset(x_val, y_val_onehot)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

print(f"  Train: {len(x_train)} samples ({len(train_loader)} batches)")
print(f"  Val:   {len(x_val)} samples ({len(val_loader)} batches)")

# Save sample images for C/RISC-V programs
indices = random.sample(range(len(x_train)), 100)
with open("galaxy_samples.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for idx in indices:
        image = x_train[idx].squeeze().numpy()  # (64, 64)
        label = torch.argmax(y_train_onehot[idx]).item()
        row = [label] + image.flatten().tolist()
        writer.writerow(row)
print("  Saved 100 sample images to galaxy_samples.csv")


# ==============================================================================
# 3. BUILD MODEL
# ==============================================================================
print("\n[3/8] Building GalaxyClassifierS4D model...")
model = GalaxyClassifierS4D(num_classes=NUM_CLASSES, colored=COLORED).to(DEVICE)
model_sum = summary(model, input_size=(2, 1 if not COLORED else 3, 64, 64), verbose=0)
print(model_sum)


# ==============================================================================
# 4. TRAINING — From notebook Section 5
# ==============================================================================
print(f"\n[4/8] Training for {EPOCHS} epochs...")
print("  Note: With conv1d on CPU, each epoch may take several minutes.")
print("  The milestone expects training to take at least 2 hours.\n")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

history = {
    "loss": [],
    "val_accuracy": []
}

best_val_accuracy = 0.0
training_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # --- Training ---
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs, return_logits=True)
        loss = loss_fn(outputs, torch.argmax(targets, dim=1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
    
    epoch_loss = running_loss / len(train_loader)
    history["loss"].append(epoch_loss)
    
    # --- Validation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs, return_logits=True)
            predicted = torch.argmax(outputs, dim=1)
            target = torch.argmax(targets, dim=1)
            correct += (predicted == target).sum().item()
            total += targets.size(0)
    
    val_accuracy = correct / total
    history["val_accuracy"].append(val_accuracy)
    
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - training_start
    avg_epoch_time = elapsed / (epoch + 1)
    remaining = avg_epoch_time * (EPOCHS - epoch - 1)
    
    elapsed_m, elapsed_s = divmod(int(elapsed), 60)
    elapsed_h, elapsed_m = divmod(elapsed_m, 60)
    remain_m, remain_s = divmod(int(remaining), 60)
    remain_h, remain_m = divmod(remain_m, 60)
    
    print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {epoch_loss:.4f} — Val Accuracy: {val_accuracy:.4f} — Time: {epoch_time:.1f}s")
    print(f"    ⏱  Elapsed: {elapsed_h}h {elapsed_m:02d}m {elapsed_s:02d}s | ETA: {remain_h}h {remain_m:02d}m {remain_s:02d}s | Avg: {avg_epoch_time:.0f}s/epoch")
    
    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "galaxy_s4_model.pth")
        print(f"    ✓ New best model saved (Val Acc: {val_accuracy:.4f})")


# ==============================================================================
# 5. PLOT TRAINING CURVES — TODO from notebook Section 6.1
# ==============================================================================
print("\n[5/8] Plotting training curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Training Loss
ax1.plot(range(1, EPOCHS+1), history["loss"], 'b-o', linewidth=2, markersize=6)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss")
ax1.set_title("Training Loss Over Epochs")
ax1.grid(True, alpha=0.3)

# Validation Accuracy
ax2.plot(range(1, EPOCHS+1), history["val_accuracy"], 'g-o', linewidth=2, markersize=6)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Accuracy")
ax2.set_title("Validation Accuracy Over Epochs")
ax2.axhline(y=0.65, color='r', linestyle='--', alpha=0.7, label="65% Target")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "training_curves.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved training_curves.png")


# ==============================================================================
# 6. EVALUATE ON TEST SET — TODO from notebook Section 6.2
# ==============================================================================
print("\n[6/8] Evaluating on test set...")

# TODO: Load the test data (from notebook Section 6.2.1)
X_test, y_test_onehot, y_test = load_data(root="./data", download=True, train=False, colored=COLORED)

test_ds = TensorDataset(X_test, y_test_onehot)
test_loader = DataLoader(test_ds, batch_size=64)

# Load best model
model.load_state_dict(torch.load("galaxy_s4_model.pth", weights_only=True))
model.eval()

correct = 0
total = 0
all_preds = []
all_targets = []
all_probs = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        # Get probabilities (softmax)
        probs = model(imgs, return_logits=False)
        logits = model(imgs, return_logits=True)
        
        preds = torch.argmax(logits, dim=1)
        target = torch.argmax(labels, dim=1)
        
        correct += (preds == target).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

test_accuracy = correct / total
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Target: ≥0.6500 — {'✓ PASSED' if test_accuracy >= 0.65 else '✗ Below target'}")

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_probs = np.array(all_probs)


# ==============================================================================
# 7. CONFUSION MATRIX — TODO from notebook Section 6.2
# ==============================================================================
print("\n[7/8] Generating confusion matrix and per-class metrics...")

# Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(f"Confusion Matrix (Test Accuracy: {test_accuracy:.4f})")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved confusion_matrix.png")

# Per-class metrics (precision, recall, F1)
print("\n  Per-Class Metrics:")
report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4)
print(report)

# Save report to file
with open(os.path.join(IMAGES_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"RNG Seed: {RNG_SEED}\n")
    f.write(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}\n\n")
    f.write(report)


# ==============================================================================
# 8. SAMPLE PREDICTIONS & ERROR ANALYSIS — Milestone Section 10.4, 10.5
# ==============================================================================
print("\n[8/8] Generating sample predictions and error analysis...")

# 3x3 grid of sample predictions
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Select samples: mix of correct and incorrect
correct_mask = all_preds == all_targets
incorrect_mask = ~correct_mask

# Get indices
correct_indices = np.where(correct_mask)[0]
incorrect_indices = np.where(incorrect_mask)[0]

# Choose samples: try to get 6 correct + 3 incorrect
n_correct = min(6, len(correct_indices))
n_incorrect = min(3, len(incorrect_indices))
n_correct = 9 - n_incorrect  # fill remaining with correct

sample_idx = []
if len(incorrect_indices) > 0:
    sample_idx.extend(np.random.choice(incorrect_indices, size=n_incorrect, replace=False))
if len(correct_indices) > 0:
    sample_idx.extend(np.random.choice(correct_indices, size=n_correct, replace=False))

np.random.shuffle(sample_idx)

for i, idx in enumerate(sample_idx[:9]):
    ax = axes[i // 3][i % 3]
    img = X_test[idx].squeeze().numpy()  # (64, 64) for grayscale
    
    true_label = CLASS_NAMES[all_targets[idx]]
    pred_label = CLASS_NAMES[all_preds[idx]]
    confidence = all_probs[idx][all_preds[idx]]
    is_correct = all_targets[idx] == all_preds[idx]
    
    # Plot with Magma colormap (as recommended in milestone for grayscale)
    ax.imshow(img, cmap='magma')
    color = 'green' if is_correct else 'red'
    ax.set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})", 
                 fontsize=9, color=color)
    ax.axis('off')

plt.suptitle("Sample Predictions (Green=Correct, Red=Incorrect)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "sample_predictions.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved sample_predictions.png")

# Error Analysis summary
print("\n  Error Analysis:")
print(f"  Total test samples: {len(all_targets)}")
print(f"  Correct: {correct_mask.sum()} ({correct_mask.mean()*100:.1f}%)")
print(f"  Incorrect: {incorrect_mask.sum()} ({incorrect_mask.mean()*100:.1f}%)")

if len(incorrect_indices) > 0:
    print("\n  Most common misclassification patterns:")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i][j] > 0:
                print(f"    {CLASS_NAMES[i]} → {CLASS_NAMES[j]}: {cm[i][j]} samples")


# ==============================================================================
# EXPORT MODEL PARAMETERS
# ==============================================================================
print("\n" + "=" * 60)
print("Exporting model parameters...")
export_model_parameters(model, MODEL_PARAMS_DIR)

# Save model with naming convention from notebook
torch.save(model.state_dict(), f"model_params/galaxys4{'-colored' if COLORED else ''}-{RNG_SEED}.pth")
print(f"  Saved model_params/galaxys4{'-colored' if COLORED else ''}-{RNG_SEED}.pth")


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Final Test Accuracy:    {test_accuracy:.4f}")
print(f"  Best Val Accuracy:      {best_val_accuracy:.4f}")
print(f"  Target (≥65%):          {'✓ ACHIEVED' if test_accuracy >= 0.65 else '✗ NOT YET'}")
print(f"  Epochs:                 {EPOCHS}")
print(f"  Learning Rate:          {LEARNING_RATE}")
print(f"  Batch Size:             {BATCH_SIZE}")
print(f"  RNG Seed:               {RNG_SEED}")
print(f"\n  Saved artifacts:")
print(f"    - galaxy_s4_model.pth (best checkpoint)")
print(f"    - model_params/ (weights export)")
print(f"    - images/training_curves.png")
print(f"    - images/confusion_matrix.png")
print(f"    - images/sample_predictions.png")
print(f"    - images/classification_report.txt")
print(f"    - galaxy_samples.csv (100 samples for C/RISC-V)")
print("=" * 60)
