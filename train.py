# import torch
# import torch.nn
# import matplotlib.pyplot as plt
# from planar_flow import PlanarFlow
# from target_distribution import TargetDistribution
# from loss import VariationalLoss
# from utils.plot import plot_transformation

# if __name__ == "__main__":
#     # ------------ parameters ------------
#     target_distr = "ring"  # U_1, U_2, U_3, U_4, ring
#     flow_length = 32
#     dim = 2
#     num_batches = 20000
#     batch_size = 128
#     lr = 6e-4
#     xlim = ylim = 7  # 5 for U_1 to U_4, 7 for ring
#     # ------------------------------------

#     density = TargetDistribution(target_distr)
#     model = PlanarFlow(dim, K=flow_length)
#     bound = VariationalLoss(density)
#     optimiser = torch.optim.Adam(model.parameters(), lr=lr)

#     # Train model.
#     for batch_num in range(1, num_batches + 1):
#         # Get batch from N(0,I).
#         batch = torch.zeros(size=(batch_size, 2)).normal_(mean=0, std=1)
#         # Pass batch through flow.
#         zk, log_jacobians = model(batch)
#         # Compute loss under target distribution.
#         loss = bound(batch, zk, log_jacobians)

#         optimiser.zero_grad()
#         loss.backward()
#         optimiser.step()

#         if batch_num % 1000 == 0:
#             print(f"(batch_num {batch_num:05d}/{num_batches}) loss: {loss}")

#         if batch_num == 1 or batch_num % 100 == 0:
#             # Save plots during training. Plots are saved to the 'train_plots' folder.
#             plot_training(model, flow_length, batch_num, lr, axlim)

#     if torch.isnan(log_jacobians).sum() == 0:
#         torch.save(
#             {
#                 "epoch": num_batches,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimiser.state_dict(),
#                 "loss": loss,
#             },
#             "models/model_" + target_distr + "_K_" + str(flow_length) + ".pt",
#         )


import time
import torch
import torchvision.datasets as datasets  # Standard datasets
from tqdm import tqdm
from torch import Tensor, nn, optim
from vae import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# Configuration
# Configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    torch.device("cuda")
else:
    torch.device("cpu")

# DEVICE = torch.device("mps") if  else  if  else "cpu")
WINDOW_LENGTH = 100
BATCH_SIZE = 50
LR_RATE = 10e-3
NUM_EPOCHS = 3
TEST_DATA = 0.3  # 30%

RNN_H_DIM = 500
DENSE_DIM = 500
Z_DIM = 3
X_DIM = 38
STD_DEVIATION_EPSILON = 10e-4
NUM_PLANAR_TRANSOFORMS = 20
TIME_AXIS = 1
DENSE_LAYERS = 2

GRAD_REGULARIZATION_LIMIT = 10
L2_REGULARIZATION = 10e-4

TARGET_DECAY = 10e-4
WAVEBOUND_ERROR_DEVIATION = 10e-2

# Dataset Loading
dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
source_model = VariationalAutoEncoder(X_DIM, RNN_H_DIM, Z_DIM, device=DEVICE).to(DEVICE)
target_model = VariationalAutoEncoder(X_DIM, RNN_H_DIM, Z_DIM, device=DEVICE).to(DEVICE)
robust_optimizer = optim.Adam(source_model.parameters(), lr=LR_RATE)
wave_optimizer = optim.Adam(source_model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

epoch_times = []
avg_loss = 0
loss_stats = {"train": [], "val": []}  # for early stop


def wave_empirical_risk(outputs, labes):
    return nn.MSELoss()(outputs, labes)


def compute_risk_with_bound(source_network, target_network):
    abs_diff = torch.abs(source_network - (target_network - WAVEBOUND_ERROR_DEVIATION))
    return abs_diff + (target_network - WAVEBOUND_ERROR_DEVIATION)


source_model.train()
target_model.train()

for epoch in range(1, NUM_EPOCHS + 1):
    loop = tqdm(enumerate(train_loader))
    start_time = time.process_time()
    for counter, (x, _) in loop:
        # Forward pass both
        x = x.to(DEVICE).view(x.shape[0], X_DIM)
        x_reconstructed, mu, sigma = source_model(x)
        x_reconstructed_t, mu_t, sigma_t = target_model(x)

        # Compute loss source network
        reconstruction_loss: Tensor = loss_fn(x_reconstructed, x)
        kl_div: Tensor = -torch.sum(
            1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
        )
        loss: Tensor = reconstruction_loss + kl_div

        # Compute loss target  network
        loss_t: Tensor = wave_empirical_risk(x_reconstructed_t, x)

        # Compute source + target loss
        wave_empirical_risk_bound: Tensor = compute_risk_with_bound(loss, loss_t)

        # Backprop
        robust_optimizer.zero_grad()
        # loss.backward()
        wave_empirical_risk_bound.backward()
        robust_optimizer.step()
        with torch.no_grad():
            for (
                source_params,
                target_params,
            ) in zip(source_model.parameters(), target_model.parameters()):
                source_params.data.mul_(TARGET_DECAY)
                torch.add(
                    source_params.data,
                    target_params.data,
                    alpha=(1 - TARGET_DECAY),
                    out=source_params.data,
                )
        loop.set_postfix(loss=loss.item())
        avg_loss += loss.item()
        if counter % 100 == 0:
            print(
                "Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(
                    epoch, (counter + 1), len(train_loader), avg_loss / (counter + 1)
                )
            )

    current_time = time.process_time()
    epoch_times.append(current_time - start_time)
    print(
        "Epoch {}/{} Done, Total Loss: {}".format(
            epoch, NUM_EPOCHS, avg_loss / len(train_loader)
        )
    )

print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
