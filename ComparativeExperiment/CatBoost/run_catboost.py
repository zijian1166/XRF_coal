#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if train_df.shape[1] < 2:
        raise ValueError("train.csv must have at least one feature column and one label column")

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test = test_df.values

    classes = np.unique(y_train)
    n_classes = len(classes)

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass" if n_classes > 2 else "Logloss",
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(-1)
    y_prob = model.predict_proba(X_test)

    result = pd.DataFrame({"y_pred": y_pred})
    for i, cls in enumerate(model.classes_):
        result[f"prob_{cls}"] = y_prob[:, i]

    local_out = base_dir / "CatBoost_result.csv"
    result.to_csv(local_out, index=False)

    result_dir = data_dir / "Result"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_out = result_dir / "result.csv"
    result.to_csv(result_out, index=False)

    print("CatBoost finished.")
    print(f"Saved: {local_out}")
    print(f"Saved: {result_out}")


if __name__ == "__main__":
    raise SystemExit(main())
