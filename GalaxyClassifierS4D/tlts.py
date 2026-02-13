import torch.nn as nn

class TakeLastTimestep(nn.Module):
    """
    Module that extracts the last timestep from a sequence.

    This layer is used to summarize sequence outputs from recurrent 
    or sequence models by taking only the final timestep as a feature vector.

    Input
    -----
    x : torch.Tensor
        Input tensor of shape (B, L, D), where
        B : batch size,
        L : sequence length,
        D : feature dimension.

    Returns
    -------
    out : torch.Tensor
        Output tensor of shape (B, D), corresponding to the last timestep
        of each sequence in the batch.
    """
    def forward(self, x):
        # Extract the last timestep for each sequence
        return x[:, -1, :]
