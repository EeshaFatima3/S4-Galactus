import torch
import torch.nn as nn
import torch.nn.functional as F

class S4Convolutional(nn.Module):
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Parameters identical to S4Recurrent for Task 5.3 validation
        self.A = nn.Parameter(torch.eye(d_state) * -0.5 + torch.randn(d_state, d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(d_state, 1) * 0.1)
        self.C = nn.Parameter(torch.randn(1, d_state) * 0.1)
        self.D = nn.Parameter(torch.zeros(d_model))
        self.log_dt = nn.Parameter(torch.log(torch.tensor([0.01])))

    def compute_kernel(self, L):
        device = self.A.device
        dt = torch.exp(self.log_dt)
        I = torch.eye(self.d_state, device=device)
        
        A_bar = torch.matrix_exp(dt * self.A)
        B_bar = torch.linalg.solve(self.A, (A_bar - I) @ self.B)

        # Kernel Generation: K = [C@B, C@A@B, C@A^2@B...]
        kernel_elements = []
        curr_mat = B_bar # (N, 1)
        for _ in range(L):
            k_i = self.C @ curr_mat # Scalar result
            kernel_elements.append(k_i.view(-1))
            curr_mat = A_bar @ curr_mat
            
        return torch.stack(kernel_elements, dim=1) # (1, L)

    def forward(self, u):
        B, L, H = u.shape
        K = self.compute_kernel(L) # (1, L)
        u_t = u.transpose(1, 2) # (B, H, L)

        # Flip for causal conv and left-pad to align with recurrence timing
        weight = K.flip(-1).view(1, 1, -1).repeat(H, 1, 1)
        u_padded = F.pad(u_t, (L - 1, 0))
        
        # Parallel convolution with groups=H for independent features
        y = F.conv1d(u_padded, weight, groups=H)
        
        # Add skip connection Du
        y = y + (self.D.view(1, H, 1) * u_t)
        
        return y.transpose(1, 2)