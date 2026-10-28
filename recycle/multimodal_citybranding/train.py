import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_class_alpha(label_ids, num_classes, device):
    class_counts = np.bincount(label_ids, minlength=num_classes)
    alpha = class_counts.sum() / np.maximum(class_counts, 1)
    alpha = alpha / alpha.mean()
    alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
    return alpha, class_counts


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_targets = []
    all_preds = []

    for image_features, text_features, labels in loader:
        image_features = image_features.to(device)
        text_features = text_features.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(image_features, text_features)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_targets.extend(labels.detach().cpu().tolist())
        all_preds.extend(torch.argmax(logits.detach(), dim=1).cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return {"loss": avg_loss, "accuracy": accuracy, "macro_f1": macro_f1}


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    early_stopping_patience,
    best_model_path,
    history_csv_path,
    metadata_path=None,
    metadata=None,
    min_delta=1e-4,
):
    best_model_path = Path(best_model_path)
    history_csv_path = Path(history_csv_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    history_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": current_lr,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['accuracy']:.4f} "
            f"f1 {train_metrics['macro_f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"acc {val_metrics['accuracy']:.4f} "
            f"f1 {val_metrics['macro_f1']:.4f} | "
            f"lr {current_lr:.2e}"
        )

        improved = val_metrics["loss"] < best_val_loss - min_delta
        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)

            if metadata_path is not None:
                best_metadata = dict(metadata or {})
                best_metadata.update(
                    {
                        "best_epoch": epoch,
                        "best_val_loss": best_val_loss,
                    }
                )
                metadata_path.write_text(
                    json.dumps(best_metadata, indent=2),
                    encoding="utf-8",
                )

            print(f"  -> checkpoint terbaik disimpan: {best_model_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                "Early stopping aktif setelah "
                f"{early_stopping_patience} epoch tanpa improvement."
            )
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)
    print(f"History training tersimpan di: {history_csv_path}")

    return {
        "history": history_df,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

