#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent
    result_path = base_dir / "MLP_result.csv"
    true_path = data_dir / "true.csv"

    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")
    if not true_path.exists():
        raise FileNotFoundError(f"True file not found: {true_path}")

    pred_df = pd.read_csv(result_path)
    true_df = pd.read_csv(true_path)

    if "y_pred" not in pred_df.columns:
        raise ValueError("MLP_result.csv must contain y_pred column")
    if true_df.shape[1] < 1:
        raise ValueError("true.csv must contain label column")

    y_pred = pred_df["y_pred"].values
    y_true = true_df.iloc[:, -1].values

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")

    analysis = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })
    analysis_path = base_dir / "Analysis.csv"
    analysis.to_csv(analysis_path, index=False)

    acc = accuracy_score(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    weighted = report.get("weighted avg", {})
    summary_row = pd.DataFrame(
        [
            {
                "algorithm": "MLP",
                "precision": weighted.get("precision", 0.0),
                "recall": weighted.get("recall", 0.0),
                "f1_score": weighted.get("f1-score", 0.0),
                "support": weighted.get("support", 0.0),
            }
        ]
    )
    summary_path = data_dir / "ComparativeExperiment.csv"
    write_header = not summary_path.exists()
    summary_row.to_csv(summary_path, mode="a", index=False, header=write_header)
    print(f"Saved summary to {summary_path}")

    print(f"Accuracy: {acc:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels, columns=labels))
    print(f"Saved analysis to {analysis_path}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
