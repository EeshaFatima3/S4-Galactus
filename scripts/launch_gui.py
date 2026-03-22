"""
launch_gui.py  —  Task 4 (Bonus): GUI Launcher with Backend Selection
=======================================================================
Launches the Galaxy Explorer GUI with either Python or C backend.

Usage:
    python launch_gui.py                  # defaults to Python backend
    python launch_gui.py --backend c      # uses C backend
    python launch_gui.py --backend python # uses Python backend

Prerequisites:
    - For 'python': galaxy_s4_model.pth in colab_results/
    - For 'c':      demo executable + model_weights.bin in task9_deliverables/
"""
import os
import sys
import argparse
import csv
import numpy as np
import torch

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLAB_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "colab_results"))
TASK9_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR))

# If we're running from colab_results, adjust paths
if os.path.exists(os.path.join(SCRIPT_DIR, "model", "interface.py")):
    COLAB_DIR = SCRIPT_DIR
    TASK9_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "task9_deliverables"))

sys.path.insert(0, COLAB_DIR)


def load_validation_data(csv_path, num_samples=100):
    """Load validation data from galaxy_samples.csv."""
    images = []
    labels = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= num_samples:
                break
            label = int(row[0])
            pixels = np.array([float(v) for v in row[1:]], dtype=np.float32)
            images.append(pixels.reshape(1, 64, 64))  # (C, H, W) grayscale
            labels.append(label)

    x_val = torch.from_numpy(np.stack(images))            # (N, 1, 64, 64)
    y_val = torch.zeros(len(labels), 4)                    # one-hot labels
    for i, l in enumerate(labels):
        y_val[i, l] = 1.0

    return x_val, y_val


def main():
    parser = argparse.ArgumentParser(description="Galaxy Explorer GUI")
    parser.add_argument(
        "--backend", choices=["python", "c"], default="python",
        help="Choose inference backend: 'python' (PyTorch) or 'c' (compiled C)"
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Number of validation samples to load (default: 100)"
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Backend: {args.backend}")
    print(f"Samples: {args.samples}")

    # --- Load model via ModelInterface ---
    from model.interface import ModelInterface

    if args.backend == "python":
        model_path = os.path.join(COLAB_DIR, "galaxy_s4_model.pth")
        model = ModelInterface(
            implementation='python',
            model_path=model_path,
            num_classes=4, colored=False, device=device
        )
    elif args.backend == "c":
        model = ModelInterface(
            implementation='c',
            model_path=TASK9_DIR,
            num_classes=4, colored=False, device=device
        )

    # --- Load validation data ---
    csv_path = os.path.join(COLAB_DIR, "galaxy_samples.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(TASK9_DIR, "galaxy_samples.csv")

    print(f"Loading data from: {csv_path}")
    x_val, y_val = load_validation_data(csv_path, args.samples)
    print(f"Loaded {len(x_val)} samples")

    # --- Launch GUI ---
    from model.gui import GalaxyExplorerGUI
    gui = GalaxyExplorerGUI(model, x_val, y_val, device)
    gui.run()


if __name__ == "__main__":
    main()
