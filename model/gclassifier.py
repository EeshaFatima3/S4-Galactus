import torch
import torch.nn as nn

from torchinfo import summary
from .hilbert import HilbertScan
from .tlts import TakeLastTimestep
from .s4d import S4D

class GalaxyClassifierS4D(nn.Module):
    """
    Galaxy classifier using Hilbert Scan and S4 sequence modeling.
    
    This model scans 2D galaxy images into a 1D Hilbert sequence, projects
    the multi-channel pixel values to a higher-dimensional feature space,
    processes the sequence with stacked S4 layers with GELU activations, 
    takes the final timestep as a summary representation, and applies a 
    linear classifier to predict galaxy types.
    
    Parameters
    ----------
    s4_state : int, optional
        Hidden state dimension for the S4 layers (default is 64).
    d_model : int, optional
        Output feature dimension of the S4 layers (default is 64).
    num_classes : int, optional
        Number of output classes (default is 4).
    colored : bool, optional
        If True, expects RGB input images (3 channels); if False, expects
        grayscale images (1 channel) (default is True).
    
    Attributes
    ----------
    seq_len : int
        Sequence length after Hilbert scan (64*64 = 4096).
    d_model : int
        Dimension of the S4 output features.
    hilbert_channels : int
        Number of input channels (1 for grayscale, 3 for RGB).
    hilbert_scan : HilbertScan
        Layer that converts 2D images into 1D sequences using a Hilbert scan.
    uproject : nn.Linear
        Linear projection mapping hilbert_channels to d_model dimensions.
    s4_1 : S4D
        First S4 layer.
    act1 : nn.GELU
        GELU activation after the first S4 layer.
    s4_2 : S4D
        Second S4 layer.
    act2 : nn.GELU
        GELU activation after the second S4 layer.
    take_last : TakeLastTimestep
        Layer that extracts the last timestep from the sequence.
    fc : nn.Linear
        Linear classifier mapping S4 features to output classes.
    softmax : nn.Softmax
        Softmax layer for output probabilities.
    """
    def __init__(self, s4_state=64, d_model=64, num_classes=4, colored=True):
        super().__init__()
        self.seq_len = 64 * 64 
        self.d_model = d_model

        # Hilbert Scan layer
        self.hilbert_scan = HilbertScan()
        self.hilbert_channels = 1 if not colored else 3

        self.uproject = nn.Linear(self.hilbert_channels, d_model)

        # S4 layers
        self.s4_1 = S4D(d_model=d_model, d_state=s4_state, transposed=False)
        self.act1 = nn.GELU()

        self.s4_2 = S4D(d_model=d_model, d_state=s4_state, transposed=False)
        self.act2 = nn.GELU()

        # Take last timestep
        self.take_last = TakeLastTimestep()

        # Classifier
        self.fc = nn.Linear(d_model, num_classes)

        # Softmax for output probabilities
        self.softmax = nn.Softmax(dim=-1)

    

    def forward(self, x, return_logits=False):
        """
        Forward pass of the PixelS4Galaxy model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, 64, 64), where B is the batch size
            and C is the number of channels (1 for grayscale, 3 for RGB).
        return_logits : bool, optional
            If True, returns raw logits instead of softmax probabilities 
            (default is False).
        
        Returns
        -------
        output : torch.Tensor
            If return_logits=True: Output logits of shape (B, num_classes),
            representing unnormalized scores for each galaxy class.
            If return_logits=False: Output probabilities of shape (B, num_classes),
            representing the softmax probability distribution over classes.
        
        """

    # x: (B, C, 64, 64)

    # 1. Hilbert scan -> (B, 4096, C)
        x = self.hilbert_scan(x)

    # 2. Linear projection -> (B, 4096, 64)
        x = self.uproject(x)

    # 3. S4 Layer 1
        x, _ = self.s4_1(x)
        x = self.act1(x)

    # 4. S4 Layer 2
        x, _ = self.s4_2(x)
        x = self.act2(x)

    # 5. Take last timestep -> (B, 64)
        x = self.take_last(x)

    # 6. Classification head -> (B, 4)
        logits = self.fc(x)

        if return_logits:
            return logits
        else:
            return self.softmax(logits)
        
        
if __name__ == "__main__":
    import torch

    print("Verifying tests. \n")

    B = 2  # Example batch size

    # Test 1: RGB Batch (B, 3, 64, 64)

    model_rgb = GalaxyClassifierS4D(colored=True)

    x_rgb = torch.randn(B, 3, 64, 64)  # (B, 3, 64, 64)
    out_rgb = model_rgb(x_rgb)         # Expected -> (B, 4)

    print("RGB Input Shape:", x_rgb.shape)      # (B, 3, 64, 64)
    print("RGB Output Shape:", out_rgb.shape)   # (B, 4)

    # Test 2: Grayscale Batch (B, 1, 64, 64)

    model_gray = GalaxyClassifierS4D(colored=False)

    x_gray = torch.randn(B, 1, 64, 64)  # (B, 1, 64, 64)
    out_gray = model_gray(x_gray)       # Expected -> (B, 4)

    print("Grayscale Input Shape:", x_gray.shape)     # (B, 1, 64, 64)
    print("Grayscale Output Shape:", out_gray.shape)  # (B, 4)

    print("\nBatch tests completed.\n")

    # Forward pass trace (Single RGB image)

    print("Running detailed forward trace: \n")

    x_rgb_single = torch.randn(1, 3, 64, 64)  # (1, 3, 64, 64)

    model = GalaxyClassifierS4D(colored=True)

    with torch.no_grad():

        print("Input:", x_rgb_single.shape)  # (1, 3, 64, 64)

        x = model.hilbert_scan(x_rgb_single)
        print("After HilbertScan:", x.shape)  # (1, 4096, 3)

        x = model.uproject(x)
        print("After Input Projection:", x.shape)  # (1, 4096, 64)

        x, _ = model.s4_1(x)
        print("After S4D Layer 1:", x.shape)  # (1, 4096, 64)

        x = model.act1(x)
        print("After GELU 1:", x.shape)  # (1, 4096, 64)

        x, _ = model.s4_2(x)
        print("After S4D Layer 2:", x.shape)  # (1, 4096, 64)

        x = model.act2(x)
        print("After GELU 2:", x.shape)  # (1, 4096, 64)

        x = model.take_last(x)
        print("After TakeLastTimestep:", x.shape)  # (1, 64)

        logits = model.fc(x)
        print("After Classification Head (logits):", logits.shape)  # (1, 4)

        probs = model.softmax(logits)
        print("After Softmax (probabilities):", probs.shape)  # (1, 4)

        print("Softmax sum:", probs.sum(dim=1))  # Should equal tensor([1.])


    # Parameter Count Verification

    print("\nParameter Count Verification (RGB model):\n")
    summary(model_rgb, input_size=(B, 3, 64, 64))

    print("\nManual parameter count (RGB):",
          sum(p.numel() for p in model_rgb.parameters()))

    print("\nParameter Count Verification (Grayscale model):\n")
    summary(model_gray, input_size=(B, 1, 64, 64))

    print("\nManual parameter count (Grayscale):",
          sum(p.numel() for p in model_gray.parameters()))
