#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tabpfn import TabPFNClassifier
from sklearn.preprocessing import StandardScaler

# =============================
# 1. 读取训练集与测试集
# =============================
base_dir = Path(__file__).resolve().parent
intermediate_dir = base_dir / "intermediate"
results_dir = base_dir / "results"
train_path = intermediate_dir / "train.csv"
test_path = intermediate_dir / "test.csv"
model_path = base_dir / "tabpfn-v2-classifier.ckpt"
results_dir.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 训练集最后一列是标签
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# 测试集不含标签，只取全部特征
X_test = test_df.values

print("训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)

# =============================
# 2. 特征标准化（推荐）
# =============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# 3. 加载 TabPFN 模型并训练
# =============================
clf = TabPFNClassifier(model_path=str(model_path))
clf.fit(X_train, y_train)

# =============================
# 4. 推理预测
# =============================
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

print("预测前20个：", y_pred[:20])
print("预测概率前5个：\n", y_prob[:5])

# =============================
# 5. 保存预测结果
# =============================
labels = clf.classes_
max_prob = y_prob.max(axis=1)
output = pd.DataFrame({
    "y_pred": y_pred,
    "max_prob": max_prob,
})
output_path = results_dir / "test_predictions.csv"
output.to_csv(output_path, index=False)
print(f"预测结果已保存到 {output_path}")

# =============================
# 6. 可视化每个点的置信度
# =============================
plt.figure(figsize=(10, 5))
plt.scatter(range(len(max_prob)), max_prob, c=max_prob, cmap="viridis", s=40)
plt.colorbar(label="Confidence")
plt.title("Prediction Confidence for Each Sample")
plt.xlabel("Sample Index")
plt.ylabel("Confidence")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
