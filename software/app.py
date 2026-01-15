#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

try:
    from PyMca5.PyMcaIO import ConfigDict
    from PyMca5.PyMcaPhysics.xrf import ClassMcaTheory
    PYMCA_AVAILABLE = True
except Exception:
    PYMCA_AVAILABLE = False

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except Exception as exc:
    raise SystemExit(
        "Missing GUI deps. Please install PyQt5 and matplotlib.\n"
        "Example: python3 -m pip install PyQt5 matplotlib\n"
        f"Error: {exc}"
    )


APP_TITLE = "X射线荧光光谱检测煤质快速分析系统"
SAMPLE_MCA = Path(__file__).resolve().parent / "煤炭压片_006-3_2.mca"
REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    sys.path.insert(0, str(REPO_ROOT))
    from xrf_analysis_pymca import DEFAULT_CFG
except Exception:
    DEFAULT_CFG = ""


def read_mca(path):
    lines = Path(path).read_text(errors="ignore").splitlines()
    header = {}
    data = []
    in_data = False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s == "<<DATA>>":
            in_data = True
            continue
        if in_data:
            if s.startswith("<<"):
                break
            try:
                data.append(int(s))
            except ValueError:
                continue
            continue
        if " - " in s:
            key, val = s.split(" - ", 1)
            header[key.strip()] = val.strip()
    return header, np.array(data, dtype=float)


def configure_matplotlib_font():
    try:
        import matplotlib as mpl
        from matplotlib import font_manager
    except Exception:
        return
    candidates = [
        "PingFang SC",
        "Heiti SC",
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Source Han Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]
    available = set()
    for font_path in font_manager.findSystemFonts(fontext="ttf"):
        try:
            available.add(font_manager.FontProperties(fname=font_path).get_name())
        except Exception:
            continue
    for name in candidates:
        if name in available:
            mpl.rcParams["font.sans-serif"] = [name]
            mpl.rcParams["axes.unicode_minus"] = False
            return


def load_default_config():
    if not DEFAULT_CFG:
        return None
    cfg = ConfigDict.ConfigDict()
    cfg.readfp(DEFAULT_CFG.splitlines().__iter__())
    return cfg


def run_pymca_fit(counts, config):
    x = np.arange(counts.size, dtype=float)
    mca_fit = ClassMcaTheory.ClassMcaTheory()
    config = mca_fit.configure(config)
    mca_fit.setData(x, counts, xmin=config["fit"]["xmin"], xmax=config["fit"]["xmax"])
    mca_fit.estimate()
    _, result = mca_fit.startfit(digest=1)
    return x, result


def align_xy(x, y):
    y = np.asarray(y)
    if x is None or len(x) != len(y):
        x = np.arange(len(y), dtype=float)
    return x, y


def build_model(name):
    if name == "LogisticRegression":
        return Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, n_jobs=1))]
        )
    if name == "SVM":
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))])
    if name == "NaiveBayes":
        return Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())])
    if name == "DecisionTree":
        return Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=42))])
    if name == "RandomForest":
        return Pipeline(
            [("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=42))]
        )
    if name == "ExtraTrees":
        return Pipeline(
            [("scaler", StandardScaler()), ("clf", ExtraTreesClassifier(n_estimators=300, random_state=42))]
        )
    if name == "MLP":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        )
    if name == "Ourmethod":
        try:
            from tabpfn import TabPFNClassifier
        except Exception as exc:
            raise RuntimeError("未安装 tabpfn") from exc
        model_path = REPO_ROOT / "Ourmethod_divide" / "tabpfn-v2-classifier.ckpt"
        if not model_path.exists():
            raise RuntimeError(f"未找到模型文件: {model_path}")
        return TabPFNClassifier(model_path=str(model_path))
    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError("未安装 xgboost") from exc
        return XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
        )
    if name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise RuntimeError("未安装 lightgbm") from exc
        return LGBMClassifier(n_estimators=120, learning_rate=0.05, num_leaves=31, random_state=42)
    if name == "CatBoost":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise RuntimeError("未安装 catboost") from exc
        return CatBoostClassifier(iterations=150, depth=6, learning_rate=0.1, verbose=False, random_seed=42)
    raise ValueError(f"Unknown model: {name}")


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        fig.tight_layout()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1280, 720)
        self.dark_mode = False

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        central.setObjectName("central")
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        top_bar = QtWidgets.QFrame()
        top_bar.setObjectName("topBar")
        top_layout = QtWidgets.QGridLayout(top_bar)
        top_layout.setContentsMargins(14, 10, 14, 10)
        logo_path = Path(__file__).resolve().parent / "fIZDNpp4w.webp"
        self.logo_label = QtWidgets.QLabel()
        self.logo_label.setObjectName("appLogo")
        self.logo_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        if logo_path.exists():
            pixmap = QtGui.QPixmap(str(logo_path))
            if not pixmap.isNull():
                self.logo_label.setPixmap(
                    pixmap.scaledToHeight(96, QtCore.Qt.SmoothTransformation)
                )
        self.logo_label.setFixedHeight(104)
        title = QtWidgets.QLabel(APP_TITLE)
        title.setObjectName("appTitle")
        title.setAlignment(QtCore.Qt.AlignCenter)
        subtitle = QtWidgets.QLabel("XRF · PyMca")
        subtitle.setObjectName("appSubtitle")
        self.theme_toggle = QtWidgets.QPushButton("深色模式")
        self.theme_toggle.setObjectName("ghostButton")
        right_box = QtWidgets.QHBoxLayout()
        right_box.setSpacing(10)
        right_box.addWidget(subtitle)
        right_box.addWidget(self.theme_toggle)
        top_layout.setColumnStretch(0, 1)
        top_layout.setColumnStretch(1, 2)
        top_layout.setColumnStretch(2, 1)
        top_layout.addWidget(self.logo_label, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        top_layout.addWidget(title, 0, 1, 1, 1)
        top_layout.addLayout(right_box, 0, 2, 1, 1, QtCore.Qt.AlignRight)
        main_layout.addWidget(top_bar)

        body = QtWidgets.QHBoxLayout()
        main_layout.addLayout(body)

        self.control_panel = QtWidgets.QGroupBox("数据与操作")
        self.control_panel.setObjectName("controlPanel")
        self.control_panel.setFixedWidth(320)
        body.addWidget(self.control_panel)
        control_layout = QtWidgets.QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(14, 14, 14, 14)
        control_layout.setSpacing(12)

        self.file_path = QtWidgets.QLineEdit()
        self.file_path.setObjectName("pathInput")
        self.file_path.setPlaceholderText("请选择 .mca 文件")
        control_layout.addWidget(self.file_path)

        file_buttons = QtWidgets.QHBoxLayout()
        control_layout.addLayout(file_buttons)
        self.btn_open = QtWidgets.QPushButton("选择文件")
        self.btn_open.setObjectName("primaryButton")
        self.btn_load = QtWidgets.QPushButton("加载并显示")
        self.btn_load.setObjectName("ghostButton")
        file_buttons.addWidget(self.btn_open)
        file_buttons.addWidget(self.btn_load)

        self.info_box = QtWidgets.QTextEdit()
        self.info_box.setObjectName("infoBox")
        self.info_box.setReadOnly(True)
        self.info_box.setPlaceholderText("文件信息")
        self.info_box.setFixedHeight(220)
        control_layout.addWidget(self.info_box)

        self.tab_group = QtWidgets.QGroupBox("功能导航")
        self.tab_group.setObjectName("tabGroup")
        tab_group_layout = QtWidgets.QVBoxLayout(self.tab_group)
        tab_group_layout.setContentsMargins(10, 12, 10, 12)
        tab_group_layout.setSpacing(10)
        self.tab_buttons = []
        for name in ["原始谱可视化", "元素面积可视化", "批量转换", "快速分析"]:
            btn = QtWidgets.QPushButton(name)
            btn.setCheckable(True)
            btn.setObjectName("navButton")
            btn.setMinimumHeight(36)
            self.tab_buttons.append(btn)
            tab_group_layout.addWidget(btn)
        control_layout.addWidget(self.tab_group)

        control_layout.addStretch(1)

        right_panel = QtWidgets.QFrame()
        right_panel.setObjectName("rightPanel")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        body.addWidget(right_panel, 1)

        self.pages = QtWidgets.QStackedWidget()
        right_layout.addWidget(self.pages, 1)

        self.tab_raw = QtWidgets.QWidget()
        self.pages.addWidget(self.tab_raw)
        raw_layout = QtWidgets.QVBoxLayout(self.tab_raw)
        raw_layout.setContentsMargins(12, 12, 12, 12)
        raw_layout.setSpacing(8)
        self.raw_canvas = MplCanvas(self.tab_raw)
        raw_layout.addWidget(self.raw_canvas)

        self.tab_elements = QtWidgets.QWidget()
        self.pages.addWidget(self.tab_elements)
        elements_layout = QtWidgets.QVBoxLayout(self.tab_elements)
        self.elements_hint = QtWidgets.QLabel("元素面积可视化（拟合后显示）")
        elements_layout.addWidget(self.elements_hint)
        self.elements_canvas_plain = MplCanvas(self.tab_elements)
        self.elements_canvas_color = MplCanvas(self.tab_elements)
        elements_layout.addWidget(self.elements_canvas_plain, 1)
        elements_layout.addWidget(self.elements_canvas_color, 1)

        self.tab_batch = QtWidgets.QWidget()
        self.pages.addWidget(self.tab_batch)
        batch_layout = QtWidgets.QVBoxLayout(self.tab_batch)
        batch_layout.addWidget(QtWidgets.QLabel("未实现：批量转换 .mca -> CSV"))

        self.tab_ml = QtWidgets.QWidget()
        self.pages.addWidget(self.tab_ml)
        ml_layout = QtWidgets.QVBoxLayout(self.tab_ml)
        ml_layout.setContentsMargins(12, 12, 12, 12)
        ml_layout.setSpacing(10)

        ml_controls = QtWidgets.QFrame()
        ml_controls.setObjectName("mlControls")
        ml_controls_layout = QtWidgets.QGridLayout(ml_controls)
        ml_controls_layout.setContentsMargins(10, 10, 10, 10)
        ml_controls_layout.setHorizontalSpacing(10)
        ml_controls_layout.setVerticalSpacing(8)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(
            [
                "LogisticRegression",
                "SVM",
                "NaiveBayes",
                "DecisionTree",
                "RandomForest",
                "ExtraTrees",
                "MLP",
                "Ourmethod",
                "XGBoost",
                "LightGBM",
                "CatBoost",
            ]
        )
        ml_controls_layout.addWidget(QtWidgets.QLabel("模型"), 0, 0)
        ml_controls_layout.addWidget(self.model_combo, 0, 1)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["使用 train/test/true", "使用总CSV自动划分"])
        ml_controls_layout.addWidget(QtWidgets.QLabel("模式"), 0, 2)
        ml_controls_layout.addWidget(self.mode_combo, 0, 3)

        self.train_path = QtWidgets.QLineEdit()
        self.test_path = QtWidgets.QLineEdit()
        self.true_path = QtWidgets.QLineEdit()
        self.train_path.setPlaceholderText("选择 train.csv")
        self.test_path.setPlaceholderText("选择 test.csv")
        self.true_path.setPlaceholderText("选择 true.csv")

        self.btn_train_pick = QtWidgets.QPushButton("选择")
        self.btn_test_pick = QtWidgets.QPushButton("选择")
        self.btn_true_pick = QtWidgets.QPushButton("选择")

        ml_controls_layout.addWidget(QtWidgets.QLabel("Train"), 1, 0)
        ml_controls_layout.addWidget(self.train_path, 1, 1)
        ml_controls_layout.addWidget(self.btn_train_pick, 1, 2)
        ml_controls_layout.addWidget(QtWidgets.QLabel("Test"), 2, 0)
        ml_controls_layout.addWidget(self.test_path, 2, 1)
        ml_controls_layout.addWidget(self.btn_test_pick, 2, 2)
        ml_controls_layout.addWidget(QtWidgets.QLabel("True"), 3, 0)
        ml_controls_layout.addWidget(self.true_path, 3, 1)
        ml_controls_layout.addWidget(self.btn_true_pick, 3, 2)

        self.full_csv_path = QtWidgets.QLineEdit()
        self.full_csv_path.setPlaceholderText("选择总的 CSV")
        self.btn_full_pick = QtWidgets.QPushButton("选择")
        self.split_ratio = QtWidgets.QDoubleSpinBox()
        self.split_ratio.setRange(0.5, 0.9)
        self.split_ratio.setSingleStep(0.05)
        self.split_ratio.setValue(0.8)
        self.shuffle_check = QtWidgets.QCheckBox("打乱")
        self.shuffle_check.setChecked(True)

        ml_controls_layout.addWidget(QtWidgets.QLabel("总CSV"), 4, 0)
        ml_controls_layout.addWidget(self.full_csv_path, 4, 1)
        ml_controls_layout.addWidget(self.btn_full_pick, 4, 2)
        ml_controls_layout.addWidget(QtWidgets.QLabel("训练比例"), 4, 3)
        ml_controls_layout.addWidget(self.split_ratio, 4, 4)
        ml_controls_layout.addWidget(self.shuffle_check, 4, 5)

        self.btn_run_ml = QtWidgets.QPushButton("开始分析")
        self.btn_run_ml.setObjectName("primaryButton")
        ml_controls_layout.addWidget(self.btn_run_ml, 5, 0, 1, 2)

        ml_layout.addWidget(ml_controls)

        metrics_row = QtWidgets.QHBoxLayout()
        self.metrics_label = QtWidgets.QLabel("准确率：-")
        self.ml_status_label = QtWidgets.QLabel("状态：未开始")
        self.ml_status_label.setObjectName("mlStatusLabel")
        metrics_row.addWidget(self.metrics_label)
        metrics_row.addStretch(1)
        metrics_row.addWidget(self.ml_status_label)
        ml_layout.addLayout(metrics_row)

        self.report_table = QtWidgets.QTableWidget()
        self.report_table.setColumnCount(4)
        self.report_table.setHorizontalHeaderLabels(["precision", "recall", "f1-score", "support"])
        self.report_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.report_table.verticalHeader().setVisible(True)
        self.report_table.setMinimumHeight(180)
        self.report_table.setObjectName("reportTable")
        ml_layout.addWidget(self.report_table)

        charts = QtWidgets.QHBoxLayout()
        self.cm_canvas = MplCanvas(self.tab_ml)
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["y_true", "y_pred", "correct"])
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.results_table.setObjectName("resultsTable")
        charts.addWidget(self.cm_canvas, 1)
        charts.addWidget(self.results_table, 1)
        ml_layout.addLayout(charts, 1)

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        self.apply_style()

        self.btn_open.clicked.connect(self.pick_file)
        self.btn_load.clicked.connect(self.load_and_plot)
        self.theme_toggle.clicked.connect(self.toggle_theme)
        for idx, btn in enumerate(self.tab_buttons):
            btn.clicked.connect(lambda checked, i=idx: self.set_active_tab(i))
        self.mode_combo.currentIndexChanged.connect(self.update_ml_mode)
        self.btn_train_pick.clicked.connect(lambda: self.pick_csv(self.train_path))
        self.btn_test_pick.clicked.connect(lambda: self.pick_csv(self.test_path))
        self.btn_true_pick.clicked.connect(lambda: self.pick_csv(self.true_path))
        self.btn_full_pick.clicked.connect(lambda: self.pick_csv(self.full_csv_path))
        self.btn_run_ml.clicked.connect(self.run_quick_analysis)
        self.set_active_tab(0)
        self.update_ml_mode()

        if SAMPLE_MCA.exists():
            self.file_path.setText(str(SAMPLE_MCA))
            self.load_and_plot()

    def pick_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 MCA 文件", str(Path.cwd()), "MCA Files (*.mca);;All Files (*)"
        )
        if path:
            self.file_path.setText(path)

    def pick_csv(self, target_line_edit):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 CSV 文件", str(Path.cwd()), "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            target_line_edit.setText(path)

    def update_ml_mode(self):
        use_split = self.mode_combo.currentIndex() == 0
        for widget in [self.train_path, self.test_path, self.true_path, self.btn_train_pick, self.btn_test_pick, self.btn_true_pick]:
            widget.setEnabled(use_split)
        for widget in [self.full_csv_path, self.btn_full_pick, self.split_ratio, self.shuffle_check]:
            widget.setEnabled(not use_split)

    def run_quick_analysis(self):
        self.ml_status_label.setText("状态：分析中...")
        QtWidgets.QApplication.processEvents()
        try:
            model_name = self.model_combo.currentText()
            model = build_model(model_name)
        except Exception as exc:
            self.status.showMessage(f"模型加载失败: {exc}", 5000)
            self.ml_status_label.setText("状态：失败")
            return

        if self.mode_combo.currentIndex() == 0:
            train_path = Path(self.train_path.text().strip())
            test_path = Path(self.test_path.text().strip())
            true_path = Path(self.true_path.text().strip())
            if not train_path.exists() or not test_path.exists() or not true_path.exists():
                self.status.showMessage("请选择有效的 train/test/true CSV", 5000)
                self.ml_status_label.setText("状态：失败")
                return
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            true_df = pd.read_csv(true_path)
            if train_df.shape[1] < 2 or true_df.shape[1] < 1:
                self.status.showMessage("CSV 格式不正确", 5000)
                self.ml_status_label.setText("状态：失败")
                return
            X_train = train_df.iloc[:, :-1].values
            y_train = train_df.iloc[:, -1].values
            X_test = test_df.values
            y_true = true_df.iloc[:, -1].values
        else:
            full_path = Path(self.full_csv_path.text().strip())
            if not full_path.exists():
                self.status.showMessage("请选择有效的总 CSV", 5000)
                self.ml_status_label.setText("状态：失败")
                return
            full_df = pd.read_csv(full_path)
            if full_df.shape[1] < 2:
                self.status.showMessage("CSV 格式不正确", 5000)
                self.ml_status_label.setText("状态：失败")
                return
            X = full_df.iloc[:, :-1].values
            y = full_df.iloc[:, -1].values
            X_train, X_test, y_train, y_true = train_test_split(
                X,
                y,
                test_size=1 - float(self.split_ratio.value()),
                shuffle=self.shuffle_check.isChecked(),
                random_state=42,
            )

        try:
            if model_name == "Ourmethod":
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                try:
                    y_prob = model.predict_proba(X_test_s)
                    max_prob = np.max(y_prob, axis=1)
                except Exception:
                    max_prob = np.zeros(len(y_pred))
            elif model_name in {"XGBoost", "LightGBM", "CatBoost"}:
                le = LabelEncoder()
                y_train_enc = le.fit_transform(y_train)
                model.fit(X_train, y_train_enc)
                y_pred_enc = model.predict(X_test)
                y_pred = le.inverse_transform(y_pred_enc)
                try:
                    y_prob = model.predict_proba(X_test)
                    max_prob = np.max(y_prob, axis=1)
                except Exception:
                    max_prob = np.zeros(len(y_pred))
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                try:
                    y_prob = model.predict_proba(X_test)
                    max_prob = np.max(y_prob, axis=1)
                except Exception:
                    max_prob = np.zeros(len(y_pred))
        except Exception as exc:
            self.status.showMessage(f"训练或推理失败: {exc}", 5000)
            self.ml_status_label.setText("状态：失败")
            return

        acc = accuracy_score(y_true, y_pred)
        report_text = classification_report(y_true, y_pred, zero_division=0)
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        self.metrics_label.setText(f"准确率：{acc:.4f}")
        self.populate_report_table(report_dict)
        self.populate_results_table(y_true, y_pred)

        ax_cm = self.cm_canvas.ax
        ax_cm.clear()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax_cm, cmap="Blues", values_format="d", colorbar=False)
        ax_cm.set_title("混淆矩阵")
        self.cm_canvas.draw()

        self.status.showMessage("快速分析完成", 3000)
        self.ml_status_label.setText("状态：完成")

    def populate_report_table(self, report_dict):
        rows = []
        for key, values in report_dict.items():
            if not isinstance(values, dict):
                continue
            rows.append((key, values))
        self.report_table.setRowCount(len(rows))
        for row_idx, (label, values) in enumerate(rows):
            self.report_table.setVerticalHeaderItem(row_idx, QtWidgets.QTableWidgetItem(str(label)))
            self.report_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(f"{values.get('precision', 0.0):.4f}"))
            self.report_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(f"{values.get('recall', 0.0):.4f}"))
            self.report_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(f"{values.get('f1-score', 0.0):.4f}"))
            self.report_table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(f"{values.get('support', 0.0):.0f}"))
        for i in range(self.report_table.rowCount()):
            for j in range(self.report_table.columnCount()):
                item = self.report_table.item(i, j)
                if item is not None:
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
        for i in range(self.report_table.rowCount()):
            header = self.report_table.verticalHeaderItem(i)
            if header is not None:
                header.setTextAlignment(QtCore.Qt.AlignCenter)

    def populate_results_table(self, y_true, y_pred):
        n = len(y_true)
        self.results_table.setRowCount(n)
        for i in range(n):
            self.results_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(y_true[i])))
            self.results_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(y_pred[i])))
            self.results_table.setItem(i, 2, QtWidgets.QTableWidgetItem("正确" if y_true[i] == y_pred[i] else "错误"))
        for i in range(self.results_table.rowCount()):
            for j in range(self.results_table.columnCount()):
                item = self.results_table.item(i, j)
                if item is not None:
                    item.setTextAlignment(QtCore.Qt.AlignCenter)

    def apply_style(self):
        if self.dark_mode:
            self.setStyleSheet(
                """
                QWidget#central {
                  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1c1c1e, stop:0.5 #15151a, stop:1 #0f1115);
                  font-family: "SF Pro Display", "Helvetica Neue", "Arial";
                  color: #f2f2f7;
                }
                QFrame#topBar {
                  background: rgba(28, 28, 30, 0.92);
                  border: 1px solid rgba(255, 255, 255, 0.08);
                  border-radius: 12px;
                }
                QLabel#appTitle {
                  font-size: 22px;
                  font-weight: 600;
                }
                QLabel#appSubtitle {
                  font-size: 12px;
                  color: #a1a1a6;
                }
                QGroupBox#controlPanel {
                  background: rgba(28, 28, 30, 0.92);
                  border: 1px solid rgba(255, 255, 255, 0.08);
                  border-radius: 16px;
                  margin-top: 12px;
                  padding: 10px;
                }
                QGroupBox#controlPanel::title {
                  subcontrol-origin: margin;
                  subcontrol-position: top left;
                  padding: 0 8px;
                  color: #d1d1d6;
                }
                QLineEdit#pathInput {
                  background: #1c1c1e;
                  border: 1px solid rgba(235, 235, 245, 0.2);
                  border-radius: 10px;
                  padding: 8px 10px;
                  color: #f2f2f7;
                }
                QPushButton#primaryButton {
                  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0a84ff, stop:1 #6ea8ff);
                  color: white;
                  border: none;
                  padding: 8px 12px;
                  border-radius: 10px;
                  font-weight: 600;
                }
                QPushButton#primaryButton:hover {
                  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #007aff, stop:1 #5a9bff);
                }
                QPushButton#ghostButton {
                  background: rgba(120, 120, 128, 0.25);
                  color: #f2f2f7;
                  border: 1px solid rgba(235, 235, 245, 0.2);
                  padding: 8px 12px;
                  border-radius: 10px;
                }
                QTextEdit#infoBox {
                  background: #1c1c1e;
                  border: 1px solid rgba(235, 235, 245, 0.2);
                  border-radius: 12px;
                  padding: 8px;
                  color: #f2f2f7;
                }
                QFrame#rightPanel {
                  background: transparent;
                }
                QGroupBox#tabGroup {
                  background: rgba(28, 28, 30, 0.92);
                  border: 1px solid rgba(255, 255, 255, 0.08);
                  border-radius: 16px;
                  padding: 8px;
                }
            QGroupBox#tabGroup::title {
              subcontrol-origin: margin;
              subcontrol-position: top left;
              padding: 0 8px;
              color: #d1d1d6;
            }
            QLabel#mlStatusLabel {
              color: #6ea8ff;
              font-weight: 600;
            }
            QTableWidget#reportTable, QTableWidget#resultsTable {
              background: #1c1c1e;
              border: 1px solid rgba(235, 235, 245, 0.2);
              border-radius: 12px;
              gridline-color: rgba(235, 235, 245, 0.15);
              color: #f2f2f7;
            }
            QHeaderView::section {
              background: rgba(120, 120, 128, 0.25);
              padding: 6px 8px;
              border: none;
              color: #f2f2f7;
            }
            QFrame#mlControls {
              background: rgba(28, 28, 30, 0.92);
              border: 1px solid rgba(255, 255, 255, 0.08);
              border-radius: 14px;
            }
                QPushButton#navButton {
                  background: rgba(120, 120, 128, 0.25);
                  color: #f2f2f7;
                  border: 1px solid rgba(235, 235, 245, 0.2);
                  padding: 8px 14px;
                  border-radius: 12px;
                  text-align: left;
                }
                QPushButton#navButton:checked {
                  background: #1c1c1e;
                  color: #0a84ff;
                  border: 1px solid rgba(235, 235, 245, 0.25);
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QWidget#central {
                  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f6f7fb, stop:0.5 #eef1f8, stop:1 #e6ecf6);
                  font-family: "SF Pro Display", "Helvetica Neue", "Arial";
                  color: #1c1c1e;
                }
            QFrame#topBar {
              background: rgba(255, 255, 255, 0.75);
              border: 1px solid rgba(0, 0, 0, 0.06);
              border-radius: 12px;
            }
            QLabel#appTitle {
              font-size: 22px;
              font-weight: 600;
            }
            QLabel#appSubtitle {
              font-size: 12px;
              color: #6e6e73;
            }
            QGroupBox#controlPanel {
              background: rgba(255, 255, 255, 0.9);
              border: 1px solid rgba(0, 0, 0, 0.06);
              border-radius: 16px;
              margin-top: 12px;
              padding: 10px;
            }
            QGroupBox#controlPanel::title {
              subcontrol-origin: margin;
              subcontrol-position: top left;
              padding: 0 8px;
              color: #3a3a3c;
            }
            QLineEdit#pathInput {
              background: #ffffff;
              border: 1px solid rgba(60, 60, 67, 0.2);
              border-radius: 10px;
              padding: 8px 10px;
            }
            QPushButton#primaryButton {
              background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #0a84ff, stop:1 #6ea8ff);
              color: white;
              border: none;
              padding: 8px 12px;
              border-radius: 10px;
              font-weight: 600;
            }
            QPushButton#primaryButton:hover {
              background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #007aff, stop:1 #5a9bff);
            }
            QPushButton#ghostButton {
              background: rgba(120, 120, 128, 0.12);
              color: #1c1c1e;
              border: 1px solid rgba(60, 60, 67, 0.2);
              padding: 8px 12px;
              border-radius: 10px;
            }
            QTextEdit#infoBox {
              background: #ffffff;
              border: 1px solid rgba(60, 60, 67, 0.2);
              border-radius: 12px;
              padding: 8px;
            }
            QTabWidget#mainTabs::pane {
              border: 1px solid rgba(0, 0, 0, 0.06);
              border-radius: 14px;
              background: rgba(255, 255, 255, 0.9);
            }
            QTabBar::tab {
              background: rgba(120, 120, 128, 0.08);
              border: 1px solid rgba(0, 0, 0, 0.06);
              padding: 8px 14px;
              border-top-left-radius: 10px;
              border-top-right-radius: 10px;
              margin-right: 4px;
            }
            QTabBar::tab:selected {
              background: #ffffff;
              color: #0a84ff;
            }
            QFrame#rightPanel {
              background: transparent;
            }
            QGroupBox#tabGroup {
              background: rgba(255, 255, 255, 0.9);
              border: 1px solid rgba(0, 0, 0, 0.06);
              border-radius: 16px;
              padding: 8px;
            }
            QGroupBox#tabGroup::title {
              subcontrol-origin: margin;
              subcontrol-position: top left;
              padding: 0 8px;
              color: #3a3a3c;
            }
            QLabel#mlStatusLabel {
              color: #0a84ff;
              font-weight: 600;
            }
            QTableWidget#reportTable, QTableWidget#resultsTable {
              background: #ffffff;
              border: 1px solid rgba(60, 60, 67, 0.2);
              border-radius: 12px;
              gridline-color: rgba(60, 60, 67, 0.15);
            }
            QHeaderView::section {
              background: rgba(120, 120, 128, 0.12);
              padding: 6px 8px;
              border: none;
            }
            QFrame#mlControls {
              background: rgba(255, 255, 255, 0.9);
              border: 1px solid rgba(0, 0, 0, 0.06);
              border-radius: 14px;
            }
            QPushButton#navButton {
              background: rgba(120, 120, 128, 0.12);
              color: #1c1c1e;
              border: 1px solid rgba(60, 60, 67, 0.2);
              padding: 8px 14px;
              border-radius: 12px;
              text-align: left;
            }
            QPushButton#navButton:checked {
              background: #ffffff;
              color: #0a84ff;
              border: 1px solid rgba(60, 60, 67, 0.25);
            }
            """
        )

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.theme_toggle.setText("浅色模式" if self.dark_mode else "深色模式")
        self.apply_style()

    def set_active_tab(self, index):
        self.pages.setCurrentIndex(index)
        for i, btn in enumerate(self.tab_buttons):
            btn.setChecked(i == index)

    def load_and_plot(self):
        path = self.file_path.text().strip()
        if not path:
            self.status.showMessage("请先选择文件", 3000)
            return
        try:
            header, counts = read_mca(path)
        except Exception as exc:
            self.status.showMessage(f"读取失败: {exc}", 5000)
            return
        if counts.size == 0:
            self.status.showMessage("未解析到谱数据", 5000)
            return

        self.info_box.clear()
        self.info_box.append(f"文件: {Path(path).name}")
        self.info_box.append(f"通道数: {counts.size}")
        if "LIVE_TIME" in header:
            self.info_box.append(f"LIVE_TIME: {header['LIVE_TIME']}")
        if "REAL_TIME" in header:
            self.info_box.append(f"REAL_TIME: {header['REAL_TIME']}")
        if "START_TIME" in header:
            self.info_box.append(f"START_TIME: {header['START_TIME']}")
        self.info_box.append("\nHeader keys:")
        self.info_box.append(", ".join(sorted(header.keys())))

        x = np.arange(counts.size)
        ax = self.raw_canvas.ax
        ax.clear()
        ax.plot(x, counts, lw=1.0, color="#1f2d3d")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Counts")
        ax.set_title(Path(path).name)
        ax.grid(alpha=0.2, linestyle="--")
        configure_matplotlib_font()
        self.raw_canvas.draw()
        self.status.showMessage("已加载并显示原始谱", 3000)

        self.update_elements_view(counts)

    def update_elements_view(self, counts):
        if not PYMCA_AVAILABLE:
            self.elements_hint.setText("缺少 PyMca5，无法进行元素拟合")
            return
        cfg = load_default_config()
        if cfg is None:
            self.elements_hint.setText("未找到 PyMca 配置，无法进行元素拟合")
            return
        try:
            x, result = run_pymca_fit(counts, cfg)
        except Exception as exc:
            self.elements_hint.setText(f"拟合失败: {exc}")
            return

        self.elements_hint.setText("元素拟合结果")
        xdata = result.get("xdata", None)
        ydata = result.get("ydata", counts)
        yfit = result.get("yfit", ydata)
        background = result.get("continuum", np.zeros_like(ydata))
        xdata, ydata = align_xy(xdata, ydata)
        _, yfit = align_xy(xdata, yfit)
        _, background = align_xy(xdata, background)
        configure_matplotlib_font()

        ax_plain = self.elements_canvas_plain.ax
        ax_plain.clear()
        ax_plain.plot(xdata, ydata, lw=1.0, color="#2b2b2b", label="data")
        ax_plain.plot(xdata, yfit, lw=1.0, color="#e67e22", label="fit")
        ax_plain.plot(xdata, background, lw=1.0, color="#7f8c8d", ls="--", label="background")
        ax_plain.set_title("拟合曲线（无分色）")
        ax_plain.set_xlabel("Channel")
        ax_plain.set_ylabel("Counts")
        ax_plain.grid(alpha=0.2, linestyle="--")
        ax_plain.legend(loc="upper right", fontsize=8, frameon=False)
        self.elements_canvas_plain.draw()

        ax_color = self.elements_canvas_color.ax
        ax_color.clear()
        ax_color.plot(xdata, ydata, lw=1.0, color="#2b2b2b", label="data")
        ax_color.plot(xdata, yfit, lw=1.0, color="#111111", alpha=0.8, label="fit")
        ax_color.plot(xdata, background, lw=1.0, color="#7f8c8d", ls="--", label="background")
        colors = [
            "#e74c3c",
            "#e67e22",
            "#f1c40f",
            "#2ecc71",
            "#1abc9c",
            "#3498db",
            "#9b59b6",
            "#e84393",
            "#16a085",
            "#d35400",
            "#2980b9",
            "#8e44ad",
        ]
        groups = result.get("groups", [])
        for i, group in enumerate(groups):
            if group.strip().upper() == "K K":
                continue
            y_group = result.get(f"y{group}")
            if y_group is None:
                continue
            label = group.split()[0] if group else group
            ax_color.fill_between(
                xdata,
                background,
                background + y_group,
                color=colors[i % len(colors)],
                alpha=0.45,
                linewidth=0.0,
                label=label,
            )
        ax_color.set_title("元素面积分色")
        ax_color.set_xlabel("Channel")
        ax_color.set_ylabel("Counts")
        ax_color.grid(alpha=0.2, linestyle="--")
        ax_color.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
        self.elements_canvas_color.draw()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
