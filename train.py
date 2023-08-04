import sys
import time
import torch
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from dataset import Dataset
from ema import EMA
from leak_finder import LeakFinder
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
leakfinder = LeakFinder()

for epoch in range(1, NUM_EPOCHS + 1):
    loop = tqdm(enumerate(train_loader))
    start_time = time.process_time()
    for counter, data in loop:
        leakfinder.set_batch(data)
        batch = torch.as_tensor(data, device=DEVICE)
        record_time = 0
        record_times = []
        leakfinder.get_cuda_perc()
        for record in batch:  # BATCH_SIZE
            record_time = time.process_time()
            leakfinder.get_cuda_perc()
            x_t, z, mu_z, logvar_z = target_model(record)
            leakfinder.get_cuda_perc()
            s_x_t, s_z, s_mu_z, s_logvar_z = source_model(record)
            leakfinder.get_cuda_perc()

            source_optimizer.zero_grad()
            source_loss = loss_function(
                record,
                x_t.squeeze(),
                mu_z.squeeze(),
                logvar_z.squeeze(),
                z.squeeze(),
                source_model,
                L2_REGULARIZATION,
            )
            leakfinder.get_cuda_perc()
            source_loss.backward(inputs=list(source_model.parameters()))
            leakfinder.get_cuda_perc()
            source_optimizer.step()
            leakfinder.get_cuda_perc()

            target_optimizer.zero_grad()
            target_loss = loss_function(
                record,
                s_x_t.squeeze(),
                s_mu_z.squeeze(),
                s_logvar_z.squeeze(),
                s_z.squeeze(),
                target_model,
                L2_REGULARIZATION,
            )
            leakfinder.get_cuda_perc()
            target_loss.backward(inputs=list(target_model.parameters()))
            leakfinder.get_cuda_perc()
            target_optimizer.step()
            leakfinder.get_cuda_perc()

            # Backprop
            with torch.no_grad():
                ema.update()
                leakfinder.get_cuda_perc()
                ema.apply_shadow()
                leakfinder.get_cuda_perc()
                # Move the in-place operation out of the `with torch.no_grad()` block
                for source_params, target_params in zip(
                    source_model.parameters(), target_model.parameters()
                ):
                    target_params.data = (
                        TARGET_DECAY * target_params.data
                        + (1 - TARGET_DECAY) * source_params.data
                    )
                leakfinder.get_cuda_perc()

            # Delete to reduce memory consumption
            del record, source_params, target_params
            del x_t, z, mu_z, logvar_z
            del s_x_t, s_z, s_mu_z, s_logvar_z
            record_times.append(time.process_time() - record_time)
            leakfinder.get_cuda_perc()

        loop.set_postfix(source_loss=source_loss.item())
        leakfinder.get_cuda_perc()
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

        del source_loss, target_loss, batch
        torch.cuda.empty_cache()  # try empty cache
        leakfinder.get_cuda_perc()
        leakfinder.find_leaks()

    current_time = time.process_time()
    epoch_times.append(current_time - start_time)
    print(
        "Epoch {}/{} Done, Total Loss: {}, Done in {}s".format(
            epoch, NUM_EPOCHS, avg_loss / len(train_loader), current_time - start_time
        )
    )

print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
