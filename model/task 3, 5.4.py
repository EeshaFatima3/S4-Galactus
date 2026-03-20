def forward(self, u):
    	"""
    	Forward pass through the S4D layer using Direct Convolution.
    	
    	1. Materialize SSM parameters (dt, A, C) from log-space.
    	2. Generate discrete SSM kernel K.
    	3. Perform direct causal convolution using conv1d.
    	4. Add skip connection D.
        """
    	if not self.transposed: u = u.transpose(-1, -2)
    	L = u.size(-1)
    	# 1. Materialize Parameters
    	dt = torch.exp(self.log_dt)
    	C = torch.view_as_complex(self.C)
    	A = -torch.exp(self.log_A_real) + 1j * self.A_imag
    	# 2. Generate Kernel K (Diagonal SSM formula)
    	# K_t = C * exp(A * dt * t) * B
    	dtA = A * dt.unsqueeze(-1) 
    	K_exp = torch.exp(dtA.unsqueeze(-1) * torch.arange(L, device=u.device))
    	C_tilde = C * (torch.exp(dtA) - 1.) / A
    	k = 2 * torch.einsum('hn, hnl -> hl', C_tilde, K_exp).real
    	# 3. Direct Convolution (y = k * u)
    	# weight shape: (out_channels, in_channels/groups, kernel_size) -> (H, 1, L)
    	weight = k.flip(-1).unsqueeze(1)
    	
    	# Causal padding (L-1 zeros at the start) to maintain output length L
    	u_padded = torch.nn.functional.pad(u, (L - 1, 0))
    	y = torch.nn.functional.conv1d(u_padded, weight, groups=self.h)
    	# 4. Skip Connection
    	y = y + u * self.D.unsqueeze(-1)
    	if not self.transposed: y = y.transpose(-1, -2)
    	return y, None