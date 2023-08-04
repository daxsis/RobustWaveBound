import sys
import time
import torch
from tqdm import tqdm
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from dataset import Dataset
from ema import EMA
from utils import get_data
from vae import VariationalAutoEncoder, loss_function

# Configuration
if len(sys.argv) > 1 and sys.argv[1] is not None:
    DEVICE = sys.argv[1]
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# DEVICE = torch.device("mps") if  else  if  else "cpu")
WINDOW_LENGTH = 30  # h
BATCH_SIZE = 100  # h
LR_RATE = 10e-4  # h
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

TARGET_DECAY = 0.9999  # h range: 0-1
WAVEBOUND_ERROR_DEVIATION = 1e-4  # h range: idk

# Dataset Loading
(x_train, _), (x_test, y_test) = get_data(
    "machine-1-1", None, None, train_start=0, test_start=0
)
data = Dataset(x_train)
train_loader = DataLoader(dataset=data, batch_size=BATCH_SIZE)
# if torch.cuda.is_available() and torch.cuda.device_count() > 2:
#     DEVICE_SOURCE = torch.device("cuda:0")
#     DEVICE_TARGET = torch.device("cuda:1")
#     target_model = VariationalAutoEncoder(X_DIM, RNN_H_DIM, Z_DIM, device=DEVICE).to(
#         DEVICE_TARGET
#     )
#     source_model = VariationalAutoEncoder(X_DIM, RNN_H_DIM, Z_DIM, device=DEVICE).to(
#         DEVICE_SOURCE
#     )
# else:
target_model = VariationalAutoEncoder(X_DIM, RNN_H_DIM, Z_DIM, device=DEVICE).to(DEVICE)
ema = EMA(target_model, TARGET_DECAY)
ema.register()
source_model = VariationalAutoEncoder(X_DIM, RNN_H_DIM, Z_DIM, device=DEVICE).to(DEVICE)
target_optimizer = optim.Adam(target_model.parameters(), lr=LR_RATE)
source_optimizer = optim.Adam(source_model.parameters(), lr=LR_RATE)

epoch_times = []
avg_loss = 0
loss_stats = {"train": [], "val": []}  # for early stop

target_model.train()
source_model.train()

for epoch in range(1, NUM_EPOCHS + 1):
    loop = tqdm(enumerate(train_loader))
    start_time = time.process_time()
    for counter, data in loop:
        batch = torch.as_tensor(data, device=DEVICE)
        record_time = 0
        record_times = []
        for record in batch:  # BATCH_SIZE
            record_time = time.process_time()

            x_t, z, mu_x, logvar_x = target_model(record)
            s_x_t, s_z, s_mu_x, s_logvar_x = source_model(record)

            source_optimizer.zero_grad()
            source_loss = loss_function(
                record,
                x_t.squeeze(),
                mu_x.squeeze(),
                logvar_x.squeeze(),
                z.squeeze(),
            )
            source_loss.backward(inputs=list(source_model.parameters()))
            source_optimizer.step()

            target_optimizer.zero_grad()
            target_loss = loss_function(
                record,
                s_x_t.squeeze(),
                s_mu_x.squeeze(),
                s_logvar_x.squeeze(),
                s_z.squeeze(),
            )
            target_loss.backward(inputs=list(target_model.parameters()))
            target_optimizer.step()
            ema.update()
            ema.apply_shadow()

            # Backprop
            with torch.no_grad():
                # Move the in-place operation out of the `with torch.no_grad()` block
                for source_params, target_params in zip(
                    source_model.parameters(), target_model.parameters()
                ):
                    target_params.data = (
                        TARGET_DECAY * target_params.data
                        + (1 - TARGET_DECAY) * source_params.data
                    )

            # Delete to reduce memory consumption
            del record, source_params, target_params
            del z, x_t, mu_x
            del s_z, s_x_t, s_mu_x
            record_times.append(time.process_time() - record_time)

        loop.set_postfix(source_loss=source_loss.item())
        avg_loss += target_loss.item()
        print(
            "\nEpoch {} | Step: {}/{} | AvrgEpoch: {} | Batch: {} | {:.2f}s".format(
                epoch,
                (counter + 1),
                len(train_loader),
                avg_loss / (counter + 1),
                target_loss.item(),
                sum(record_times),
            )
        )

        del source_loss, target_loss
        batch.detach()
        torch.cuda.empty_cache()  # try empty cache

    current_time = time.process_time()
    epoch_times.append(current_time - start_time)
    print(
        "Epoch {}/{} Done, Total Loss: {}, Done in {}s".format(
            epoch, NUM_EPOCHS, avg_loss / len(train_loader), current_time - start_time
        )
    )

print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
