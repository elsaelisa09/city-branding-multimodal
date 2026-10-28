from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

try:
    from .train import run_epoch
except ImportError:
    if __package__:
        raise
    from train import run_epoch


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def plot_training_curves(history_df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train")
    axes[0].plot(
        history_df["epoch"],
        history_df["val_loss"],
        marker="o",
        label="Validation",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Focal Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        history_df["epoch"],
        history_df["train_macro_f1"],
        marker="o",
        label="Train Macro F1",
    )
    axes[1].plot(
        history_df["epoch"],
        history_df["val_macro_f1"],
        marker="o",
        label="Validation Macro F1",
    )
    axes[1].plot(
        history_df["epoch"],
        history_df["train_accuracy"],
        linestyle="--",
        alpha=0.7,
        label="Train Accuracy",
    )
    axes[1].plot(
        history_df["epoch"],
        history_df["val_accuracy"],
        linestyle="--",
        alpha=0.7,
        label="Validation Accuracy",
    )
    axes[1].set_title("Metric")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    for image_features, text_features, labels in loader:
        image_features = image_features.to(device)
        text_features = text_features.to(device)
        logits = model(image_features, text_features)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def save_confusion_matrix(y_true, y_pred, label_names, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = len(label_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title("Confusion Matrix - Validation")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return cm


def evaluate_checkpoint(
    model,
    val_loader,
    criterion,
    val_df,
    label_names,
    device,
    checkpoint_path,
    confusion_matrix_path,
    prediction_csv_path,
):
    model.load_state_dict(load_state_dict(checkpoint_path, device))
    best_val_metrics = run_epoch(model, val_loader, criterion, device)
    print("Best checkpoint validation metrics:", best_val_metrics)

    y_true, y_pred, y_prob = predict_loader(model, val_loader, device)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    )
    print(report)

    cm = save_confusion_matrix(
        y_true,
        y_pred,
        label_names,
        confusion_matrix_path,
    )

    prediction_df = val_df[["file", "caption", "label"]].copy()
    prediction_df["prediction"] = [label_names[idx] for idx in y_pred]
    prediction_df["correct"] = prediction_df["label"].eq(prediction_df["prediction"])
    prediction_csv_path = Path(prediction_csv_path)
    prediction_csv_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(prediction_csv_path, index=False, encoding="utf-8")

    print(f"Confusion matrix tersimpan di: {confusion_matrix_path}")
    print(f"Prediksi validation tersimpan di: {prediction_csv_path}")

    return {
        "metrics": best_val_metrics,
        "classification_report": report,
        "confusion_matrix": cm,
        "prediction_df": prediction_df,
        "probabilities": y_prob,
    }
