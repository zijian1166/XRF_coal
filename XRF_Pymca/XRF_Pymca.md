# XRF PyMca

本文档主要说明 `xrf_analysis_pymca.py` 的作用、参数、输出与结果解读，帮助快速理解单谱拟合流程。

## 1. 适用数据与前置条件

- 数据格式：`.mca`（PMCA/Spectrum 格式）。
- 依赖：PyMca5、NumPy、Matplotlib。
- 关键前提：必须使用与仪器匹配的 PyMca 配置（探测器、几何、管压/靶材、能量标定）。

如果拟合曲线与实测谱偏差较大，通常是配置不匹配、标定不准或元素列表设置不合理导致的。

## 2. 核心脚本：`xrf_analysis_pymca.py`

功能：读取单个 `.mca`，进行拟合、输出元素组结果，并可生成两类图：
- 原始谱 + 拟合曲线（带峰位标注）
- 分元素填色面积图（元素贡献可视化）

基本用法：

```bash
python3 xrf_analysis_pymca.py \
  --path Data/Original_data/煤炭压片_004_1.mca \
  --plot spectrum_pymca.png
```

分元素面积图会自动保存为 `spectrum_pymca_elements.png`，也可指定：

```bash
python3 xrf_analysis_pymca.py \
  --path Data/Original_data/煤炭压片_004_1.mca \
  --plot spectrum_pymca.png \
  --plot-elements spectrum_elements.png
```

常用参数：
- `--config`：指定 PyMca `.cfg` 配置（建议）
- `--calib`：能量标定 `zero,gain`
- `--fix-calib`：固定标定参数
- `--elements`：元素列表（逗号分隔）
- `--xmin/--xmax`：拟合范围（通道）
- `--fast`：降低迭代次数，加速（精度降低）

## 3. 批量输出：`tocsv.py`

功能：遍历目录下所有 `.mca`，输出指定元素的 `fitarea` 表。

默认元素顺序：
```
Si, Al, Fe, Ca, Mg, P, K, Na, Ti, V, Mn, S
```

输出 CSV 最后一列为“煤样编号”（从文件名中提取数字，如 `煤炭压片_006-1_3.mca` → `006`）。

用法：

```bash
python3 tocsv.py --input-dir Data/Original_data --output fitarea.csv
```

## 4. CSV 随机打乱：`数据/打乱.py`

功能：对 `fitarea.csv` 或指定 CSV 进行随机打乱并输出到脚本目录。

```bash
python3 数据/打乱.py --input fitarea.csv --output 数据/fitarea_shuffle.csv --seed 42
```

## 5. 结果字段解释

输出表中常见字段：

- `group`：谱线组（元素 + 线系，如 `Fe K`、`Ca K`）
- `fitarea`：拟合得到的净峰面积（该元素贡献，已去背景）
- `sigmaarea`：`fitarea` 的统计不确定度（1σ）
- `mcaarea`：原始谱在该组峰区内的总面积（含背景/散射）

## 6. 如何解读拟合结果

1) **相对强弱**
   - `fitarea` 越大 → 元素含量（或响应）越强。
   - `Ca K` 最大通常说明钙含量或响应较高。

2) **拟合可靠性**
   - `sigmaarea` 接近 `fitarea`：不确定度高，可能是弱峰或重叠峰。
   - `sigmaarea` 远小于 `fitarea`：峰识别较可靠。

3) **`fitarea` vs `mcaarea`**
   - `mcaarea` 通常大于 `fitarea`，因为它包含背景与散射。

## 7. 常见问题与排查

- **拟合曲线偏差很大**
  - 检查 `.cfg` 是否匹配仪器
  - 校准参数 `zero/gain` 是否正确
  - `xmin/xmax` 是否覆盖主要峰位

- **元素标注缺失或过少**
  - 目前默认标注所有拟合组，但仍以 `result['groups']` 为准
  - 若某元素未出现在 `groups`，说明该元素未被配置或未拟合成功

- **运行时间过长**
  - 使用 `--fast` 降低迭代次数
  - 缩小 `xmin/xmax` 拟合范围

## 8. 输出文件说明

- `spectrum_pymca.png`：原始谱 + 拟合 + 峰位标注
- `spectrum_pymca_elements.png`：分元素填色面积图
- `fitarea.csv`：批量元素 `fitarea` 汇总

如需扩展输出（更多元素、更多字段、浓度定量等），可在配置文件或脚本中进一步设置。
