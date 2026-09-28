"""
Step 3 - Digital twin model definition (CNN-LSTM hybrid).

Imported by 04_train.py; not meant to be run standalone, but a quick
smoke test lives in __main__ (python scripts/03_model.py) that builds
the model, runs a dummy forward pass, and reports the parameter count.
"""
import torch
import torch.nn as nn

# Sliding window length: 10 timesteps. FEMTO acc files are ~1 sample every
# 10s near end-of-life, so a 10-step window covers roughly the last
# ~100s of degradation trend without needing hundreds of steps of history;
# C-MAPSS cycles are coarser (1 row/flight) so 10 cycles is a similarly
# small-but-informative trend window. Same window size is reused for both
# so the architecture doesn't need per-dataset branching.
WINDOW_SIZE = 10


class DigitalTwinCNNLSTM(nn.Module):
    def __init__(self, n_features, window_size=WINDOW_SIZE):
        super().__init__()
        # MinMax-scaled sensor channels (Step 1) sit in a narrow band far from
        # 0 (e.g. 0.95-0.99), so their real signal is tiny relative to their
        # offset. Without rescaling, the LSTM barely responds to input at all
        # (empirically: batch-wise output std ~1e-4, i.e. it collapses to
        # predicting the mean RUL and gradients never recover in time for
        # early stopping). A BatchNorm on the raw input channels fixes this
        # by re-centering/rescaling per feature using batch statistics.
        self.bn0 = nn.BatchNorm1d(n_features)

        # Conv1d expects (batch, channels, length) -> channels = n_features
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)

        self.fc_embedding = nn.Linear(64, 32)  # "embedding" layer, exposed for later phases
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, window_size, n_features)
        x = x.transpose(1, 2)          # -> (batch, n_features, window_size)
        x = self.bn0(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)          # -> (batch, window_size, 64) for LSTM

        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # final timestep summarizes the window's trend

        embedding = self.relu(self.fc_embedding(last_step))
        rul_pred = self.fc_out(embedding)
        return rul_pred.squeeze(-1), embedding


if __name__ == "__main__":
    model = DigitalTwinCNNLSTM(n_features=10)
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(4, WINDOW_SIZE, 10)
    pred, emb = model(dummy)
    print(f"param count: {n_params:,}")  # kept under ~500K: this model is sent every
    # federated round in later phases, so its size directly drives per-round
    # communication cost (upload+download) across all client sites.
    print(f"pred shape: {tuple(pred.shape)}, embedding shape: {tuple(emb.shape)}")
    assert n_params < 500_000, "model exceeds the 500K parameter budget"
