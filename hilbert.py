import torch   
import torch.nn as nn


class HilbertScan(nn.Module):
    """
    Reorders pixels according to a Hilbert Curve for multi-channel images.
    
    The Hilbert curve is a space-filling curve that preserves spatial locality
    when mapping 2D coordinates to 1D sequences. This module applies the same
    Hilbert curve pattern to each channel independently, then reorganizes the
    output so the sequence dimension comes first.
    
    Supports grayscale (C=1) or RGB (C=3) images.
    
    Attributes
    ----------
    indices : torch.LongTensor
        Precomputed Hilbert curve indices for a 64×64 grid, stored as a
        non-trainable buffer.
    
    Input
    -----
    x : torch.Tensor
        Input tensor of shape (B, C, H, W), where
        B : batch size
        C : number of channels (1 for grayscale, 3 for RGB)
        H : height (64)
        W : width (64)
    
    Returns
    -------
    out : torch.Tensor
        Reordered tensor of shape (B, seq_len, C) where seq_len = H*W = 4096.
        Pixels are arranged according to the Hilbert curve traversal order.
    """
    def __init__(self):
        """Initialize HilbertScan with precomputed indices for 64x64 images."""
        super().__init__()
        indices = self.get_hilbert_indices(64)
        self.register_buffer('indices', indices)

    def _d2xy(self, n, d):

        x = 0 # Starting with initial coordinates 
        y = 0  
        t = d # d working copy
        s = 1 # Current square space, we build our way up

        while s < n:
            rx = 1 & (t // 2) # These help determine which coordinate we are in
            ry = 1 & (t ^ rx)

            # Rotate if only ry == 0
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                # Swap x and y if ry == 0 and rx ==1
                # For a continuous curve
                x, y = y, x

            x += s * rx  # Offset addition to move into the correct quadrant
            y += s * ry

            t //= 4 # Each level splits into 4 quads; process 2 bits at a time
            s *= 2 # Size of s grows

        return x, y

    def get_hilbert_indices(self, n):
        """
        Generate Hilbert curve indices for an n x n grid.
        
        Creates a lookup table that maps Hilbert curve positions to
        flattened array indices for a 2D grid.
        
        Parameters
        ----------
        n : int
            Grid size (must be a power of 2).
        
        Returns
        -------
        torch.LongTensor
            Tensor of shape (n²,) containing flattened indices following
            the Hilbert curve traversal order.
        """
        indices = []
        for d in range(n * n):
            x, y = self._d2xy(n, d)
            # GalaxyMNIST is 64x64, power of 2
            if x < 64 and y < 64:
                indices.append(y * 64 + x)
        return torch.LongTensor(indices)

    def forward(self, x):
        """
        Apply Hilbert curve reordering to input images.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (B, C, H, W).
        
        Returns
        -------
        torch.Tensor
            Reordered tensor of shape (B, seq_len, C) where seq_len = H*W,
            with pixels arranged in Hilbert curve order.
        """
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.view(B, C, -1)           # Flatten each channel: (B, C, H*W)
        x = x[:, :, self.indices]      # Reorder according to Hilbert: (B, C, H*W)
        x = x.permute(0, 2, 1)         # (B, seq_len, C) so sequence dimension is 1D
        return x

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    print("Running HilbertScan test...\n")

    hilbert = HilbertScan()

    # --- Test 1: Print first 10 coordinates for 8x8 ---
    n = 8
    print("First 10 Hilbert coordinates (8x8):")
    for d in range(10):
        x, y = hilbert._d2xy(n, d)
        print(f"d = {d} -> ({x}, {y})")

    # --- Test 2: Visualize 8x8 Hilbert curve ---
    xs, ys = [], []
    for d in range(n * n):
        x, y = hilbert._d2xy(n, d)
        xs.append(x)
        ys.append(y)

    plt.figure(figsize=(5,5))
    plt.plot(xs, ys, '-o', markersize=5)
    for i in range(len(xs)):
        plt.text(xs[i], ys[i], str(i), fontsize=8)
    plt.gca().invert_yaxis()
    plt.title("Hilbert Curve (8x8)")
    plt.show()

    # --- Test 3: Compare average consecutive distances (64x64) ---
    n = 64

    # 1) Row-major order coordinates
    row_coords = [(j, i) for i in range(n) for j in range(n)]

    row_distances = [
        np.sqrt((row_coords[i+1][0] - row_coords[i][0])**2 + 
                (row_coords[i+1][1] - row_coords[i][1])**2)
        for i in range(len(row_coords)-1)
    ]
    row_avg = np.mean(row_distances)

    # 2) Hilbert order coordinates
    hilbert_coords = [hilbert._d2xy(n, d) for d in range(n*n)]

    hilbert_distances = [
        np.sqrt((hilbert_coords[i+1][0] - hilbert_coords[i][0])**2 + 
                (hilbert_coords[i+1][1] - hilbert_coords[i][1])**2)
        for i in range(len(hilbert_coords)-1)
    ]
    hilbert_avg = np.mean(hilbert_distances)

    # --- Print results ---
    print("\nAverage consecutive distances (64x64):")
    print(f"{'Method':<12} {'Average Distance'}")
    print(f"{'-'*30}")
    print(f"{'Row-major':<12} {row_avg:.4f}")
    print(f"{'Hilbert':<12} {hilbert_avg:.4f}")

    # --- Conceptual explanation ---
    print("\nExplanation:")
    print("Hilbert curve preserves spatial locality: pixels close in 2D space are also close sequentially.")
    print("Row-major order has large jumps at row boundaries, increasing average distance.")
    print("For S4 sequence models, smaller sequential distances mean local spatial patterns are")
    print("propagated more effectively in the hidden state, improving learning of images.")
