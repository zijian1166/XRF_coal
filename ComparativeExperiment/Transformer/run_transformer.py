#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if train_df.shape[1] < 2:
        raise ValueError("train.csv must have at least one feature column and one label column")

    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train_raw = train_df.iloc[:, -1].values
    X_test = test_df.values.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    n_classes = len(le.classes_)
    n_features = X_train.shape[1]

    class TabTransformer(nn.Module):
        def __init__(self, n_features, n_classes, d_model=64, nhead=4, num_layers=2, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.cls = nn.Linear(d_model, n_classes)

        def forward(self, x):
            # x: [B, F]
            x = x.unsqueeze(-1)  # [B, F, 1]
            x = self.input_proj(x)  # [B, F, d_model]
            x = self.encoder(x)  # [B, F, d_model]
            x = x.mean(dim=1)  # [B, d_model]
            return self.cls(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabTransformer(n_features=n_features, n_classes=n_classes).to(device)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).long()

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    epochs = 30
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(device)
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    y_pred = le.inverse_transform(preds)

    result = pd.DataFrame({"y_pred": y_pred})
    for i, cls in enumerate(le.classes_):
        result[f"prob_{cls}"] = probs[:, i]

    local_out = base_dir / "Transformer_result.csv"
    result.to_csv(local_out, index=False)

    result_dir = data_dir / "Result"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_out = result_dir / "result.csv"
    result.to_csv(result_out, index=False)

    print("Transformer finished.")
    print(f"Saved: {local_out}")
    print(f"Saved: {result_out}")


if __name__ == "__main__":
    raise SystemExit(main())
