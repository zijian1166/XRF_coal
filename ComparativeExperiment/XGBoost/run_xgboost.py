#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


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
    y_raw = train_df.iloc[:, -1].values
    X_test = test_df.values

    # =========================
    # 1. 标签编码（关键修复点）
    # =========================
    le = LabelEncoder()
    y_train = le.fit_transform(y_raw)

    classes = le.classes_
    n_classes = len(classes)

    print("Original classes:", classes)
    print("Encoded as:", np.arange(n_classes))

    # =========================
    # 2. 构建模型
    # =========================
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(X_train, y_train)

    # =========================
    # 3. 预测
    # =========================
    y_pred_enc = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # 解码回原始标签
    y_pred = le.inverse_transform(y_pred_enc)

    # =========================
    # 4. 保存结果
    # =========================
    result = pd.DataFrame({"y_pred": y_pred})

    for i, cls in enumerate(classes):
        result[f"prob_{cls}"] = y_prob[:, i]

    local_out = base_dir / "XGBoost_result.csv"
    result.to_csv(local_out, index=False)

    result_dir = data_dir / "Result"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_out = result_dir / "result.csv"
    result.to_csv(result_out, index=False)

    print("XGBoost finished.")
    print(f"Saved: {local_out}")
    print(f"Saved: {result_out}")


if __name__ == "__main__":
    raise SystemExit(main())