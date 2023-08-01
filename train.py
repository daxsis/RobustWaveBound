import time
import torch
from tqdm import tqdm
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from dataset import Dataset
from utils import get_data
from vae import VariationalAutoEncoder, loss_function

# Configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# DEVICE = torch.device("mps") if  else  if  else "cpu")
WINDOW_LENGTH = 30  # h
BATCH_SIZE = 100  # h
LR_RATE = 10e-3  # h
NUM_EPOCHS = 3  # range: any
TEST_DATA = 0.3  # 30% range: 10-100%

RNN_H_DIM = 500
DENSE_DIM = 500
Z_DIM = 3  # h range: 3-10
X_DIM = 38
STD_DEVIATION_EPSILON = 10e-4  # h
NUM_PLANAR_TRANSOFORMS = 20  # h
TIME_AXIS = 1
DENSE_LAYERS = 2  # h

GRAD_REGULARIZATIONLIMIT = 10  # h
L2_REGULARIZATION = 10e-4  # h

TARGET_DECAY = 0.5  # h range: 0-1
WAVEBOUND_ERROR_DEVIATION = 1e-4  # h range: idk

# Dataset Loading
(x_train, _), (x_test, y_test) = get_data(
    "machine-1-1", None, None, train_start=0, test_start=0
)
data = Dataset(data=x_train, window=WINDOW_LENGTH)
train_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE)
target_model = VariationalAutoEncoder(X_DIM, RNN_H_DIM, Z_DIM, device=DEVICE).to(DEVICE)
source_model = VariationalAutoEncoder(X_DIM, RNN_H_DIM, Z_DIM, device=DEVICE).to(DEVICE)
target_optimizer = optim.Adam(target_model.parameters(), lr=LR_RATE)
source_optimizer = optim.Adam(source_model.parameters(), lr=LR_RATE)

epoch_times = []
avg_loss = 0
loss_stats = {"train": [], "val": []}  # for early stop


def wave_empirical_risk(outputs, labes):
    return nn.MSELoss()(outputs, labes)


def compute_risk_with_bound(source_network, target_network):
    abs_diff = torch.abs(source_network - (target_network - WAVEBOUND_ERROR_DEVIATION))
    return abs_diff + (target_network - WAVEBOUND_ERROR_DEVIATION)


target_model.train()
source_model.train()

for epoch in range(1, NUM_EPOCHS + 1):
    loop = tqdm(enumerate(train_loader))
    start_time = time.process_time()
    for counter, data in loop:
        batch = torch.as_tensor(data, device=DEVICE)
        batch_counter = 0
        window_time = 0
        window_times = []
        for window in batch:  # BATCH_SIZE
            batch_counter += 1
            window_loss = 0
            window_time = time.process_time()
            print(
                "We are {} out of {} records in batch #{} in {}s. Total time in batch {}s".format(
                    batch_counter,
                    BATCH_SIZE,
                    counter + 1,
                    window_time,
                    str(sum(window_times)),
                )
            )

            record_times = []
            for record in window:  # WINDOW_LENGTH
                z, mu_z, log_var_z, x_t, mu_x, log_var_x = target_model(record)
                source_model(record)

                source_loss = loss_function(
                    record, x_t, mu_x, log_var_x, mu_z, log_var_z
                )
                target_loss: Tensor = wave_empirical_risk(x_t.squeeze(), record)

                # Compute source + target loss
                wave_empirical_risk_bound: Tensor = compute_risk_with_bound(
                    source_loss, target_loss
                )

                window_loss += wave_empirical_risk_bound
                # Backprop
                source_optimizer.zero_grad()
                # target_optimizer.zero_grad()
                wave_empirical_risk_bound.backward(retain_graph=True)
                source_optimizer.step()
                # target_optimizer.step()
                with torch.no_grad():
                    for (
                        source_params,
                        target_params,
                    ) in zip(source_model.parameters(), target_model.parameters()):
                        target_params.data = TARGET_DECAY * target_params.data + (
                            (1 - TARGET_DECAY) * source_params.data
                        )

                del z, mu_z, log_var_z, x_t, mu_x, log_var_x

                if (
                    batch_counter % (WINDOW_LENGTH / 10) == 0
                ):  # print every 10 recs in window
                    print(
                        "Epoch {}......Batch: {}/{}...Window: {} %.... Average Loss For Window: {}".format(
                            epoch,
                            (counter + 1),
                            len(train_loader),
                            (batch_counter / BATCH_SIZE) * 100,
                            window_loss / WINDOW_LENGTH,
                        )
                    )

            window_time = time.process_time() - window_time
            window_times.append(window_time)

        loop.set_postfix(loss=target_loss.item())
        avg_loss += target_loss.item()
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
