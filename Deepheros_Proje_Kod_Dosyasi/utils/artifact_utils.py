import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

def save_predictions(y_true, y_pred, y_probs=None, ids=None, output_path="preds_test.csv"):
    data = {
        'id': ids if ids is not None else range(len(y_true)),
        'y_true': y_true,
        'y_pred': y_pred
    }

    if y_probs is not None:
        for i in range(y_probs.shape[1]):
            data[f'prob_class_{i}'] = y_probs[:, i]

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[OK] Saved predictions to: {output_path}")

def compute_metrics(y_true, y_pred):
    return {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'F1_micro': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'F1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'Precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'Recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    }

def save_metrics(metrics, output_path="metrics.json"):
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Saved metrics to: {output_path}")

def save_classification_report(y_true, y_pred, output_path="class_report.json"):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    report_dict = {
        'per_class': {},
        'macro_avg': {
            'precision': float(report.get('macro avg', {}).get('precision', 0)),
            'recall': float(report.get('macro avg', {}).get('recall', 0)),
            'f1_score': float(report.get('macro avg', {}).get('f1-score', 0)),
            'support': int(report.get('macro avg', {}).get('support', 0))
        },
        'weighted_avg': {
            'precision': float(report.get('weighted avg', {}).get('precision', 0)),
            'recall': float(report.get('weighted avg', {}).get('recall', 0)),
            'f1_score': float(report.get('weighted avg', {}).get('f1-score', 0)),
            'support': int(report.get('weighted avg', {}).get('support', 0))
        },
        'accuracy': float(report.get('accuracy', 0))
    }

    for key, value in report.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(value, dict):
            try:
                class_id = int(key)
                report_dict['per_class'][f'class_{class_id}'] = {
                    'precision': float(value.get('precision', 0)),
                    'recall': float(value.get('recall', 0)),
                    'f1_score': float(value.get('f1-score', 0)),
                    'support': int(value.get('support', 0))
                }
            except ValueError:
                pass

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"[OK] Saved classification report to: {output_path}")

def save_confusion_matrix(y_true, y_pred, class_names=None, output_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else range(len(np.unique(y_true))),
                yticklabels=class_names if class_names else range(len(np.unique(y_true))))
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved confusion matrix to: {output_path}")

def save_training_curve(history, output_path="train_curve.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.get('train_loss', [])) + 1)

    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', marker='o')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

    metric_key = 'val_accuracy' if 'val_accuracy' in history else 'val_f1_micro'
    if metric_key in history:
        train_key = metric_key.replace('val_', 'train_')
        if train_key in history:
            axes[1].plot(epochs, history[train_key], 'b-', label=f'Train {metric_key.replace("val_", "").title()}', marker='o')
        axes[1].plot(epochs, history[metric_key], 'r-', label=f'Val {metric_key.replace("val_", "").title()}', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_key.replace('val_', '').title())
        axes[1].set_title(f'Training and Validation {metric_key.replace("val_", "").title()}')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved training curve to: {output_path}")

def save_history_csv(history, output_path="history.csv"):
    df = pd.DataFrame(history)
    df.to_csv(output_path, index=False)
    print(f"[OK] Saved training history to: {output_path}")

def save_run_config(config, output_path="run_config.json"):
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Saved run config to: {output_path}")

def save_all_artifacts(y_true, y_pred, y_probs=None, ids=None, history=None,
                      config=None, class_names=None, artifact_dir="./artifacts/model"):
    os.makedirs(artifact_dir, exist_ok=True)

    save_predictions(y_true, y_pred, y_probs, ids,
                    os.path.join(artifact_dir, "preds_test.csv"))

    metrics = compute_metrics(y_true, y_pred)
    save_metrics(metrics, os.path.join(artifact_dir, "metrics.json"))

    save_classification_report(y_true, y_pred,
                              os.path.join(artifact_dir, "class_report.json"))

    save_confusion_matrix(y_true, y_pred, class_names,
                         os.path.join(artifact_dir, "confusion_matrix.png"))

    if history:
        save_training_curve(history, os.path.join(artifact_dir, "train_curve.png"))
        save_history_csv(history, os.path.join(artifact_dir, "history.csv"))

    if config:
        save_run_config(config, os.path.join(artifact_dir, "run_config.json"))

    print(f"\n[OK] All artifacts saved to: {artifact_dir}")
