from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# =============================
# 1. 加载 CSV 数据
# =============================
csv_path = Path(__file__).resolve().parent / "Data_10classification.csv"
model_path = "tabpfn-v2-classifier.ckpt"  # 本地模型路径

df = pd.read_csv(csv_path)
print("数据前5行：")
print(df.head())

# 假设最后一列是标签列
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# =============================
# 2. 打乱 + 划分 80/20
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=41
)

print("训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)

# =============================
# 3. 特征标准化（推荐）
# =============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# 4. 加载本地 TabPFN 模型
# =============================
clf = TabPFNClassifier(model_path=model_path)
clf.fit(X_train, y_train)

# =============================
# 5. 推理预测
# =============================
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

print("预测前20个：", y_pred[:20])
print("预测概率前5个：\n", y_prob[:5])

# =============================
# 6. 结果分析：分类报告 + 混淆矩阵
# =============================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

labels = np.unique(np.concatenate([y_test, y_pred]))
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# =============================
# 8. 保存结果到 CSV
# =============================
output = pd.DataFrame({
    "y_test": y_test,
    "y_pred": y_pred,
    "prob": y_prob[:, 1]
})
output.to_csv("TabPFN_predictions.csv", index=False)
print("预测结果已保存到 TabPFN_predictions.csv")
