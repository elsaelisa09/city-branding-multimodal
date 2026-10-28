from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "images_skema2"
DEFAULT_LABEL_FILE = PROJECT_ROOT / "dataset" / "labels_skema2.xlsx"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"


@dataclass
class PipelineConfig:
    seed: int = 42
    batch_size: int = 16
    epochs: int = 24
    early_stopping_patience: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    project_dim: int = 256
    dropout: float = 0.30
    focal_gamma: float = 2.0
    text_max_length: int = 160
    test_size: float = 0.2
    clip_model_name: str = "openai/clip-vit-base-patch32"
    text_model_name: str = "cardiffnlp/twitter-xlm-roberta-base"
    image_dir: Path = DEFAULT_IMAGE_DIR
    label_file: Path = DEFAULT_LABEL_FILE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    device: str = "auto"
    augment_train_images: bool = True


def artifact_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "best_model": output_dir / "best_intermediate_fusion.pt",
        "metadata": output_dir / "best_intermediate_fusion_metadata.json",
        "history_csv": output_dir / "training_history.csv",
        "label_distribution": output_dir / "label_distribution.png",
        "training_curves": output_dir / "training_curves.png",
        "confusion_matrix": output_dir / "confusion_matrix.png",
        "prediction_csv": output_dir / "validation_predictions.csv",
    }


def resolve_device(device_name):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA diminta, tetapi tidak tersedia. Pipeline memakai CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def run_pipeline(config):
    import torch
    from torch.utils.data import DataLoader

    try:
        from .data_loader import (
            FusionFeatureDataset,
            labels_to_tensor,
            load_labeled_dataframe,
            save_label_distribution,
            split_labeled_dataframe,
        )
        from .evaluation import evaluate_checkpoint, plot_training_curves
        from .models import (
            FocalLoss,
            IntermediateFusionClassifier,
            MultimodalFeatureExtractor,
        )
        from .train import build_class_alpha, seed_everything, train_model
    except ImportError:
        if __package__:
            raise
        from data_loader import (
            FusionFeatureDataset,
            labels_to_tensor,
            load_labeled_dataframe,
            save_label_distribution,
            split_labeled_dataframe,
        )
        from evaluation import evaluate_checkpoint, plot_training_curves
        from models import (
            FocalLoss,
            IntermediateFusionClassifier,
            MultimodalFeatureExtractor,
        )
        from train import build_class_alpha, seed_everything, train_model

    seed_everything(config.seed)
    device = resolve_device(config.device)
    paths = artifact_paths(config.output_dir)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output directory: {Path(config.output_dir).resolve()}")

    loaded = load_labeled_dataframe(
        label_file=config.label_file,
        image_dir=config.image_dir,
    )
    df = loaded["df"]
    label_names = loaded["label_names"]
    label_to_id = loaded["label_to_id"]
    id_to_label = loaded["id_to_label"]

    if loaded["missing_image_count"]:
        print(
            "Warning: "
            f"{loaded['missing_image_count']} gambar tidak ditemukan dan dibuang."
        )
        print(loaded["missing_image_preview"].to_string(index=False))

    print(f"Dataset siap: {len(df)} data berlabel")
    print(f"Label mapping: {label_to_id}")

    label_counts = save_label_distribution(df, paths["label_distribution"])
    print(label_counts)
    print(f"Visualisasi distribusi label tersimpan di: {paths['label_distribution']}")

    train_df, val_df, split_counts = split_labeled_dataframe(
        df,
        label_names,
        test_size=config.test_size,
        seed=config.seed,
    )
    print(f"Train: {len(train_df)} data ({len(train_df) / len(df):.1%})")
    print(f"Validation: {len(val_df)} data ({len(val_df) / len(df):.1%})")
    print(split_counts)

    feature_extractor = MultimodalFeatureExtractor(
        clip_model_name=config.clip_model_name,
        text_model_name=config.text_model_name,
        device=device,
    )
    print(f"CLIP embedding dim: {feature_extractor.image_dim}")
    print(f"Text embedding dim: {feature_extractor.text_dim}")

    train_image_features = feature_extractor.extract_image_features(
        train_df["image_path"].tolist(),
        augment=config.augment_train_images,
        batch_size=config.batch_size,
        desc="train image features",
    )
    val_image_features = feature_extractor.extract_image_features(
        val_df["image_path"].tolist(),
        augment=False,
        batch_size=config.batch_size,
        desc="validation image features",
    )
    train_text_features = feature_extractor.extract_text_features(
        train_df["caption"].tolist(),
        batch_size=config.batch_size,
        max_length=config.text_max_length,
        desc="train text features",
    )
    val_text_features = feature_extractor.extract_text_features(
        val_df["caption"].tolist(),
        batch_size=config.batch_size,
        max_length=config.text_max_length,
        desc="validation text features",
    )

    y_train = labels_to_tensor(train_df)
    y_val = labels_to_tensor(val_df)

    print("train_image_features:", tuple(train_image_features.shape))
    print("train_text_features:", tuple(train_text_features.shape))
    print("val_image_features:", tuple(val_image_features.shape))
    print("val_text_features:", tuple(val_text_features.shape))

    train_dataset = FusionFeatureDataset(
        train_image_features,
        train_text_features,
        y_train,
    )
    val_dataset = FusionFeatureDataset(val_image_features, val_text_features, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    num_classes = len(label_names)
    alpha, class_counts = build_class_alpha(
        train_df["label_id"].values,
        num_classes=num_classes,
        device=device,
    )
    print("Class counts train:", dict(zip(label_names, class_counts.tolist())))
    print(
        "Focal alpha:",
        dict(zip(label_names, alpha.detach().cpu().numpy().round(3).tolist())),
    )

    model = IntermediateFusionClassifier(
        image_dim=train_image_features.shape[1],
        text_dim=train_text_features.shape[1],
        project_dim=config.project_dim,
        num_classes=num_classes,
        dropout=config.dropout,
    ).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=config.focal_gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    metadata = {
        "clip_model_name": config.clip_model_name,
        "text_model_name": config.text_model_name,
        "image_dim": int(train_image_features.shape[1]),
        "text_dim": int(train_text_features.shape[1]),
        "project_dim": config.project_dim,
        "num_classes": num_classes,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "early_stopping_patience": config.early_stopping_patience,
        "class_counts_train": {
            label: int(count) for label, count in zip(label_names, class_counts.tolist())
        },
    }

    training_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=config.epochs,
        early_stopping_patience=config.early_stopping_patience,
        best_model_path=paths["best_model"],
        history_csv_path=paths["history_csv"],
        metadata_path=paths["metadata"],
        metadata=metadata,
    )

    plot_training_curves(training_result["history"], paths["training_curves"])
    print(f"Kurva training tersimpan di: {paths['training_curves']}")

    evaluation_result = evaluate_checkpoint(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        val_df=val_df,
        label_names=label_names,
        device=device,
        checkpoint_path=paths["best_model"],
        confusion_matrix_path=paths["confusion_matrix"],
        prediction_csv_path=paths["prediction_csv"],
    )
    return {
        "training": training_result,
        "evaluation": evaluation_result,
        "paths": paths,
    }


def parse_args():
    parser = ArgumentParser(
        description="Training intermediate fusion city branding multimodal."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--project-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--text-max-length", type=int, default=160)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--text-model-name",
        default="cardiffnlp/twitter-xlm-roberta-base",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Matikan augmentasi ringan untuk fitur gambar training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        project_dim=args.project_dim,
        dropout=args.dropout,
        focal_gamma=args.focal_gamma,
        text_max_length=args.text_max_length,
        test_size=args.test_size,
        clip_model_name=args.clip_model_name,
        text_model_name=args.text_model_name,
        image_dir=args.image_dir,
        label_file=args.label_file,
        output_dir=args.output_dir,
        device=args.device,
        augment_train_images=not args.no_augment,
    )
    try:
        run_pipeline(config)
    except ModuleNotFoundError as exc:
        missing_package = exc.name
        print(f"Dependency belum tersedia: {missing_package}")
        print(
            "Install dulu dengan: "
            "pip install matplotlib scikit-learn transformers pillow tqdm pandas torch"
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
