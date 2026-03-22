"""
model/interface.py  —  Unified Model Interface
================================================
Supports three backends:
  - 'python'  : PyTorch model (Milestone 1)
  - 'c'       : C implementation via subprocess (Task 4 Bonus, Milestone 2)
  - 'riscv'   : RISC-V QEMU (Milestone 3, placeholder)
"""
import os
import re
import subprocess
import tempfile
import numpy as np
import torch


class ModelInterface:
    """
    Unified interface for galaxy classification models.

    This class abstracts the implementation details (Python / C / RISC-V)
    and provides a consistent API for model inference regardless of backend.

    Parameters
    ----------
    implementation : str
        One of 'python', 'c', or 'riscv'.
    model_path : str
        Path to model weights (.pth for Python, directory containing
        model_weights.bin + demo executable for C).
    num_classes : int
        Number of output classes.
    colored : bool
        Whether model expects colored or grayscale images.
    device : torch.device
        Device for inference.
    """

    def __init__(self, implementation, model_path, num_classes, colored, device):
        """Initialize model based on implementation type."""
        self.implementation = implementation
        self.device = device
        self.num_classes = num_classes

        if implementation == 'python':
            from model import GalaxyClassifierS4D
            print(f"Loading Python model from {model_path}")
            self.model = GalaxyClassifierS4D(colored=colored).to(device)
            self.model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
            self.model.eval()

        elif implementation == 'c':
            # ---- C Backend Setup (Task 4 Bonus) ----
            # model_path should point to the task9_deliverables directory
            # that contains: demo (or demo.exe), model_weights.bin
            self.c_dir = os.path.abspath(model_path)

            # Find the demo executable
            demo_candidates = ['demo', 'demo.exe', 's4d_classifier', 's4d_classifier.exe']
            self.demo_exe = None
            for name in demo_candidates:
                path = os.path.join(self.c_dir, name)
                if os.path.isfile(path):
                    self.demo_exe = path
                    break

            if self.demo_exe is None:
                raise FileNotFoundError(
                    f"Cannot find demo executable in {self.c_dir}. "
                    f"Build it with: gcc -O2 -o demo nn.c main.c -lm"
                )

            # Verify model_weights.bin exists
            self.weights_path = os.path.join(self.c_dir, "model_weights.bin")
            if not os.path.isfile(self.weights_path):
                raise FileNotFoundError(
                    f"Cannot find model_weights.bin in {self.c_dir}"
                )

            print(f"[C Backend] Executable: {self.demo_exe}")
            print(f"[C Backend] Weights:    {self.weights_path}")
            self.model = None

        elif implementation == 'riscv':
            print("Initializing RISC-V interface")
            # TODO: Setup RISC-V communication/configuration
            self.model = None

        else:
            raise ValueError(
                f"Unknown implementation '{implementation}'. "
                f"Choose from: 'python', 'c', 'riscv'"
            )

    def __call__(self, x):
        """
        Run model inference.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (1, C, H, W) or (1, C, L).

        Returns
        -------
        torch.Tensor
            Model predictions (probabilities), shape (1, num_classes).
        """
        if self.implementation == 'python':
            return self.model(x)

        elif self.implementation == 'c':
            return self._run_c_inference(x)

        elif self.implementation == 'riscv':
            # TODO: Send x to RISC-V QEMU, receive predictions
            raise NotImplementedError("RISC-V inference not yet implemented")

    def _run_c_inference(self, x):
        """
        Run inference through the C backend via subprocess.

        Steps:
          1. Flatten input tensor to raw pixel array (4096 floats)
          2. Save as temporary .bin file
          3. Execute demo executable with the .bin file path
          4. Parse stdout to extract class probabilities
          5. Return as PyTorch tensor

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (1, C, H, W).

        Returns
        -------
        torch.Tensor
            Probability distribution, shape (1, num_classes).
        """
        # 1. Flatten tensor to raw pixel array and save to temp file
        img_np = x.squeeze().cpu().float().numpy()

        # Handle different input shapes:
        #   (C, H, W) -> flatten to (H*W*C,)
        #   (H, W)    -> flatten to (H*W,)
        img_flat = img_np.flatten().astype(np.float32)

        # 2. Write to temporary binary file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.bin')
        try:
            with os.fdopen(tmp_fd, 'wb') as f:
                img_flat.tofile(f)

            # 3. Run the demo executable
            result = subprocess.run(
                [self.demo_exe, tmp_path],
                capture_output=True,
                text=True,
                cwd=self.c_dir,   # so it can find model_weights.bin
                timeout=60,       # safety timeout for O(L^2) convolution
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"C demo failed (exit code {result.returncode}):\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            # 4. Parse probabilities from the output
            #    Expected: "Class 0 (Smooth Round): 0.1545"
            probs = re.findall(
                r'Class\s+\d+\s*\([^)]*\):\s+([\d.]+)',
                result.stdout
            )

            if len(probs) != self.num_classes:
                raise RuntimeError(
                    f"Expected {self.num_classes} probabilities, "
                    f"got {len(probs)}.\nC output:\n{result.stdout}"
                )

            prob_values = [float(p) for p in probs]

        finally:
            # 5. Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return torch.tensor(prob_values).unsqueeze(0)

    def eval(self):
        """Set model to evaluation mode (for consistency with PyTorch API)."""
        if self.implementation == 'python':
            self.model.eval()
        # C and RISC-V don't need eval mode — they're always inference-only
