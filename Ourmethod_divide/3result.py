#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# =============================
# 1. 读取预测结果与真实标签
# =============================
base_dir = Path(__file__).resolve().parent
intermediate_dir = base_dir / "intermediate"
results_dir = base_dir / "results"
pred_path = results_dir / "test_predictions.csv"
true_path = intermediate_dir / "true.csv"
results_dir.mkdir(parents=True, exist_ok=True)

pred_df = pd.read_csv(pred_path)
true_df = pd.read_csv(true_path)

# true.csv 最后一列是标签
if true_df.shape[1] < 1:
    raise ValueError("true.csv 为空或没有标签列")

y_true = true_df.iloc[:, -1].values

if "y_pred" not in pred_df.columns or "max_prob" not in pred_df.columns:
    raise ValueError("test_predictions.csv must contain y_pred and max_prob columns")

y_pred = pred_df["y_pred"].values
max_prob = pred_df["max_prob"].values

if len(y_true) != len(y_pred):
    raise ValueError("预测结果与真实标签数量不一致")

if len(y_true) != len(max_prob):
    raise ValueError("预测结果与置信度数量不一致")

analysis = pd.DataFrame({
    "y_true": y_true,
    "y_pred": y_pred,
    "correct": (y_true == y_pred),
    "max_prob": max_prob,
})
analysis_path = results_dir / "analysis.csv"
analysis.to_csv(analysis_path, index=False)
print(f"分析结果已保存到 {analysis_path}")

# =============================
# 2. 输出分类报告
# =============================
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# =============================
# 3. 混淆矩阵可视化
# =============================
labels = np.unique(np.concatenate([y_true, y_pred]))
cm = confusion_matrix(y_true, y_pred, labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# =============================
# 4. 正确/错误预测可视化
# =============================
correct = (y_true == y_pred)
plt.figure(figsize=(10, 4))
plt.scatter(range(len(correct)), correct.astype(int), c=correct, cmap="coolwarm", s=40)
plt.yticks([0, 1], ["Wrong", "Correct"])
plt.title("Prediction Correctness")
plt.xlabel("Sample Index")
plt.ylabel("Result")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
