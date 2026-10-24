import torch
import torch.nn as nn


class MultiScaleFourierEmbedding(nn.Module):
    """Deterministic multi-frequency Fourier feature mapping for orbital coordinate MLPs."""
    def __init__(self, num_frequencies: int = 16):
        super().__init__()
        freqs = torch.linspace(0.5, 4.0, num_frequencies).unsqueeze(0)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        proj = t @ self.freqs
        return torch.cat([t, torch.sin(proj), torch.cos(proj)], dim=-1)


class OrbitGuardPINN(nn.Module):
    """
    Kryvond OrbitGuard — Physics-Informed Neural Network (PINN) for LEO Trajectory
    Propagation & Conjunction Risk Scoring.

    Governing Differential Equation (Earth J2 Oblateness Perturbation in ECI Frame):
      d^2(r)/dt^2 = - (mu / |r|^3) * r + a_J2(r)
    Normalized in Canonical Earth Units (DU = R_earth = 6378.137 km, TU = 806.811 s, mu = 1.0).
    """
    MU: float = 1.0
    J2: float = 1.08262668e-3
    R_E: float = 1.0

    def __init__(self, hidden_dim: int = 128, num_frequencies: int = 16):
        super().__init__()
        self.fourier = MultiScaleFourierEmbedding(num_frequencies=num_frequencies)
        in_dim = (1 + 2 * num_frequencies) + 6
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, t: torch.Tensor, state0: torch.Tensor) -> torch.Tensor:
        """
        Hard-constraint initial condition ansatz so r(0) == r0 and v(0) == v0 identically.
        """
        r0 = state0[:, :3]
        v0 = state0[:, 3:6]
        r0_norm = torch.norm(r0, dim=-1, keepdim=True).clamp(min=1e-4)
        a0_kepler = -self.MU * r0 / (r0_norm ** 3)

        t_feat = self.fourier(t)
        delta_r = self.net(torch.cat([t_feat, state0], dim=-1))

        envelope = torch.exp(-0.25 * (t ** 2))
        return r0 + v0 * t + 0.5 * a0_kepler * (t ** 2) * envelope + (t ** 2) * delta_r

    def predict_state_and_acceleration(
        self, t: torch.Tensor, state0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes position r(t), velocity v(t), and acceleration a(t) via PyTorch Autograd."""
        t_req = t.clone().detach().requires_grad_(True)
        r = self.forward(t_req, state0)

        velocities = []
        accelerations = []
        for i in range(3):
            v_i = torch.autograd.grad(
                r[:, i], t_req, grad_outputs=torch.ones_like(r[:, i]), create_graph=True
            )[0]
            a_i = torch.autograd.grad(
                v_i, t_req, grad_outputs=torch.ones_like(v_i), create_graph=True
            )[0]
            velocities.append(v_i)
            accelerations.append(a_i)

        v_pred = torch.cat(velocities, dim=-1)
        a_pred = torch.cat(accelerations, dim=-1)
        return r, v_pred, a_pred

    def compute_j2_pde_residual(self, t: torch.Tensor, state0: torch.Tensor) -> torch.Tensor:
        """Computes exact 2nd-order orbital mechanics J2 PDE residual via PyTorch Autograd."""
        r, _, acc_pred = self.predict_state_and_acceleration(t, state0)

        x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]
        r_norm = torch.norm(r, dim=-1, keepdim=True).clamp(min=1e-4)

        acc_kepler = -self.MU * r / (r_norm ** 3)

        z2_r2 = (z / r_norm) ** 2
        factor = (1.5 * self.J2 * self.MU * (self.R_E ** 2)) / (r_norm ** 5)
        ax_j2 = factor * x * (5.0 * z2_r2 - 1.0)
        ay_j2 = factor * y * (5.0 * z2_r2 - 1.0)
        az_j2 = factor * z * (5.0 * z2_r2 - 3.0)
        acc_j2 = torch.cat([ax_j2, ay_j2, az_j2], dim=-1)

        residual = acc_pred - (acc_kepler + acc_j2)
        return torch.mean(residual ** 2)
