# XRF Coal Project

本项目用于煤样 XRF 谱数据的解析与机器学习分类对比实验，包含：

- 基于 PyMca 的单谱拟合与元素峰面积提取
- 批量 `.mca` 文件处理与 CSV 输出
- 多模型分类对比（传统 ML + 集成模型 + 深度模型）

---

## 1. XRF 谱分析（PyMca）

### 1.1 单谱拟合
脚本：`xrf_analysis_pymca.py`

功能：
- 读取 `.mca` 文件并进行谱线拟合
- 输出元素峰的拟合面积与统计不确定度
- 可视化拟合结果与元素分色面积图

常用命令：

```bash
python3 xrf_analysis_pymca.py \
  --path Data/Original_data/煤炭压片_004_1.mca \
  --plot spectrum_pymca.png
```

会额外生成：
- `spectrum_pymca_elements.png`（元素分色图）

更多说明见：`XRF_Pymca/XRF_Pymca.md`

### 1.2 批量输出元素峰面积
脚本：`tocsv.py`

功能：
- 扫描 `Data/Original_data` 目录下所有 `.mca`
- 输出元素 `fitarea` 表

输出顺序：
```
Si, Al, Fe, Ca, Mg, P, K, Na, Ti, V, Mn, S
```

命令：

```bash
python3 tocsv.py --output fitarea.csv
```

### 1.3 CSV 随机打乱
脚本：`数据/打乱.py`

功能：将 CSV 行随机打乱输出。

```bash
python3 数据/打乱.py --input fitarea.csv --output 数据/fitarea_shuffle.csv
```

---

## 2. 机器学习分类对比实验

### 2.1 数据目录结构
`ComparativeExperiment/` 下的数据：
- `train.csv`：训练集（最后一列为标签）
- `test.csv`：测试集（仅特征）
- `true.csv`：测试集真实标签

输出目录：
- `Result/result.csv`：每个模型推理输出（通用）
- 各模型目录：保存专属结果和分析

### 2.2 每个模型的结构
每个模型目录中包含：
- `run_*.py`：训练 + 推理 + 输出预测结果
- `AnalysisResults.py`：与 `true.csv` 对比，输出混淆矩阵、分类报告，并将指标写入 `ComparativeExperiment.csv`

### 2.3 已实现模型
- LogisticRegression
- SVM
- NaiveBayes
- DecisionTree
- RandomForest
- ExtraTrees
- XGBoost
- LightGBM
- CatBoost
- MLP
- Transformer

### 2.4 结果汇总
`ComparativeExperiment/ComparativeExperiment.csv` 记录各模型的 weighted 平均指标：
- `precision`
- `recall`
- `f1_score`
- `support`

表头：
```
algorithm,precision,recall,f1_score,support
```

---

## 3. 依赖环境

主要依赖：
- Python 3.x
- numpy, pandas, matplotlib
- scikit-learn
- PyMca5
- xgboost / lightgbm / catboost（可选）
- torch（Transformer 需要）

安装示例：

```bash
python3 -m pip install numpy pandas matplotlib scikit-learn PyMca5
python3 -m pip install xgboost lightgbm catboost
python3 -m pip install torch
```

---

## 4. 常见问题

- **拟合曲线偏差很大**：检查 PyMca `.cfg` 是否匹配仪器与标定
- **结果 CSV 科学计数法**：脚本已改为普通小数输出
- **不同模型分析结果相同**：确保分析脚本读取自身模型输出文件

---

## 5. 建议流程

1. 用 PyMca 拟合 `.mca` 并批量输出 `fitarea.csv`
2. 按需要打乱或划分训练/测试集
3. 运行各模型 `run_*.py`
4. 运行 `AnalysisResults.py` 生成混淆矩阵和汇总指标
5. 在 `ComparativeExperiment.csv` 中对比模型表现
