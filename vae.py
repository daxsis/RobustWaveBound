from typing import Tuple
import torch
from torch import Tensor
from torch import nn

# from lgssm import LinearGaussianStateSpaceModel
from planar_flow import PlanarFlow


WINDOW_LENGTH = 100
BATCH_SIZE = 50
LR_RATE = 10e-3
NUM_EPOCHS = 3
TEST_DATA = 0.3  # 30%

GRAD_REGULARIZATION_LIMIT = 10
L2_REGULARIZATION = 1e-5
RNN_H_DIM = 500
DENSE_DIM = 500
Z_DIM = 3
X_DIM = 38
STD_DEVIATION_EPSILON = 10e-4
NUM_PLANAR_TRANSOFORMS = 20
TIME_AXIS = 1
DENSE_LAYERS = 2


class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim=38,
        h_dim=200,
        z_dim=3,
        seq_len=100,
        planar_length=20,
        epsilon=1e-4,
        device="cuda",
    ):
        super(VariationalAutoEncoder, self).__init__()
        self.z_dim = z_dim
        # Define epsilon for numerical stability
        self.epsilon = torch.tensor(epsilon)
        self.input_dim = input_dim
        self.seq_length = seq_len
        self.planar_length = planar_length
        self.h_dim = h_dim
        self.device = device
        self.zt_minus_1: Tensor = None  # reusing the last calculated Z

        # encoder - q net
        # Define the GRU
        self.h_qnet = torch.randn(self.h_dim, device=self.device)
        self.h_for_q_z = nn.GRUCell(
            input_size=self.input_dim, hidden_size=self.h_dim, device=self.device
        )
        # Define the linear layers for the mean and standard deviation
        self.q_dense = nn.Sequential(
            nn.Linear(self.h_dim + self.input_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        ).to(self.device)
        self.q_mu = nn.Linear(self.h_dim, self.z_dim, device=self.device)
        self.q_sigma = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim), nn.Softplus(1, 1)
        ).to(self.device)
        # Define the planar normalizing flow layer
        self.q_planar_nf = PlanarFlow(dim=self.z_dim, num_tranforms=self.planar_length)

        # decoder - p net
        # Initialize transition and observation noise for LG-SSM
        self.T_theta = nn.Linear(self.z_dim, self.z_dim, bias=False, device=self.device)
        self.O_theta = nn.Linear(self.z_dim, self.z_dim, bias=False, device=self.device)
        # Initialize transition and observation noise for LG-SSM
        self.transition_noise = torch.distributions.Normal(0, 1)
        self.observation_noise = torch.distributions.Normal(0, 1)

        self.h_pnet = torch.randn(1, self.h_dim, device=self.device)
        self.h_for_p_x = nn.GRUCell(
            input_size=self.z_dim, hidden_size=self.h_dim, device=self.device
        )
        self.p_dense = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        ).to(self.device)
        self.p_mu = nn.Linear(self.h_dim, self.input_dim, device=self.device)
        self.p_sigma = nn.Sequential(
            nn.Linear(self.h_dim, self.input_dim), nn.Softplus(1, 1)
        ).to(self.device)

    def encode(self, x: Tensor) -> Tensor:
        h_qnet = self.h_for_q_z(x, self.h_qnet)
        self.h_qnet = h_qnet  # update hidden state
        if self.zt_minus_1 is None:
            self.zt_minus_1 = torch.randn(x.size(0), device=self.device)
        h_qnet_concat: Tensor = torch.cat(
            [h_qnet, self.zt_minus_1],
            dim=-1,
        )
        z_params = self.q_dense(h_qnet_concat)
        mu_z = self.q_mu(z_params)
        log_var_z = self.q_sigma(z_params)

        return mu_z, log_var_z

    # Sample from Gaussian
    def reparametrize(self, mu_z: Tensor, log_var_z: Tensor) -> Tensor:
        std_z = torch.exp(0.5 * log_var_z)
        epsilon = torch.randn_like(std_z, device=self.device)

        return mu_z + epsilon * std_z  # z

    def decode(self, z):
        # LG-SSM state and observation equations
        # state equation
        zt = self.T_theta(z) + self.transition_noise.sample()
        # observation equation
        zt = self.O_theta(zt) + self.observation_noise.sample()

        # Ensure zt has a batch size dimension
        zt = zt.view(1, -1)
        h_pnet = self.h_for_p_x(zt, self.h_pnet)
        self.h_pnet = h_pnet  # update hidden for p net gru cell
        x_params = self.p_dense(h_pnet)
        mu_x = self.p_mu(x_params)
        log_var_x = self.p_sigma(x_params)

        return mu_x, log_var_x

    def forward(self, x: Tensor):
        mu_z, log_var_z = self.encode(x)
        z0: Tensor = self.reparametrize(mu_z, log_var_z)  # sample from prior q(z|x)
        # Transform through planar normalizing flow
        z, _ = self.q_planar_nf(z0)

        mu_x, log_var_x = self.decode(z)
        x_t: Tensor = self.reparametrize(mu_x, log_var_x)

        return x_t, z, mu_z, log_var_z

    # def calculate_l2_loss(self):
    #     l2_loss = torch.tensor(0.0, requires_grad=True)
    #     for param in self.parameters():
    #         l2_loss += torch.norm(param) ** 2
    #     return l2_loss


def loss_function(
    x: Tensor, x_t: Tensor, mu_z: Tensor, logvar_z: Tensor, z: Tensor, model, lambda_l2
):
    recon_loss = nn.functional.mse_loss(x_t, x, reduction="sum")  # log(p(x|z))
    prior = torch.distributions.Normal(
        torch.zeros_like(mu_z), torch.ones_like(mu_z)
    )  # prior p(z)
    log_p_z = prior.log_prob(z).sum(-1)  # log(p(z))
    posterior = torch.distributions.Normal(
        mu_z, torch.exp(0.5 * logvar_z)
    )  # posterior q(z|x)
    log_q_z_given_x = posterior.log_prob(z).sum(-1)  # log(q(z|x))
    # l2_reg = model.calculate_l2_loss()
    elbo = (
        log_p_z + recon_loss - log_q_z_given_x
    ).mean()  # Monte Carlo estimate of ELBO
    return -elbo  # return negative ELBO as the loss to be minimized


if __name__ == "__main__":
    x = torch.randn(1, 4, 28 * 28)
    vae = VariationalAutoEncoder(
        input_dim=X_DIM, h_dim=RNN_H_DIM, z_dim=Z_DIM, device="mps"
    )
    print(vae)
    print(len(list(vae.parameters())))
    # x_reconstructed, mu, sigma = vae(x)
    # print(x_reconstructed.shape)
    # print(mu.shape)
    # print(sigma.shape)
