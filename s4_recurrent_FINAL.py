import torch
import torch.nn as nn

class S4Recurrent(nn.Module):
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model   # H
        self.d_state = d_state   # N

        # 5.1 Requirement: A is N x N, B is N x 1, C is 1 x N
        self.A = nn.Parameter(torch.eye(d_state) * -0.5 + torch.randn(d_state, d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(d_state, 1) * 0.1)
        self.C = nn.Parameter(torch.randn(1, d_state) * 0.1)
        self.D = nn.Parameter(torch.zeros(d_model))
        
        # Log-space Delta for positivity
        self.log_dt = nn.Parameter(torch.log(torch.tensor([0.01])))

    def forward(self, u):
        B, L, H = u.shape
        N = self.d_state
        device = u.device

        # Discretization (ZOH) using torch.matrix_exp as requested
        dt = torch.exp(self.log_dt)
        I = torch.eye(N, device=device)
        A_bar = torch.matrix_exp(dt * self.A)
        B_bar = torch.linalg.solve(self.A, (A_bar - I) @ self.B)

        # Initialize state x_0 = 0 (Shape: B, H, N)
        x = torch.zeros(B, H, N, device=device)
        outputs = []

        for k in range(L):
            u_k = u[:, k, :].unsqueeze(-1) # (B, H, 1)
            
            # x_k = A_bar @ x_{k-1} + B_bar @ u_k
            # Broadcasting: (N,N) @ (B,H,N,1) + (N,1) * (B,H,1)
            x = (A_bar @ x.unsqueeze(-1)).squeeze(-1) + (B_bar @ u_k.transpose(-1, -2)).transpose(-1, -2).squeeze(-1)
            
            # y_k = C @ x_k + D * u_k
            # C @ x.unsqueeze(-1) gives (B, H, 1, 1). We squeeze it to (B, H)
            y_k = (self.C @ x.unsqueeze(-1)).view(B, H) + (u[:, k, :] * self.D)
            outputs.append(y_k)

        return torch.stack(outputs, dim=1)