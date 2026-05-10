from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "images_skema2"
DEFAULT_LABEL_FILE = PROJECT_ROOT / "dataset" / "labels_skema2.xlsx"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"


class FusionFeatureDataset(Dataset):
    def __init__(self, image_features, text_features, labels):
        self.image_features = image_features.float()
        self.text_features = text_features.float()
        self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.image_features[idx], self.text_features[idx], self.labels[idx]


def _column_index(cell_ref):
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch.upper()) - ord("A") + 1
    return index - 1


def read_xlsx_basic(path):
    path = Path(path)
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared_strings.append(
                    "".join(text.text or "" for text in item.findall(".//a:t", ns))
                )

        sheet_files = sorted(
            name
            for name in archive.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not sheet_files:
            raise ValueError(f"Tidak ada worksheet di {path}")

        root = ET.fromstring(archive.read(sheet_files[0]))
        rows = []
        for row in root.findall(".//a:sheetData/a:row", ns):
            values = {}
            max_idx = -1
            for cell in row.findall("a:c", ns):
                idx = _column_index(cell.get("r", "A1"))
                max_idx = max(max_idx, idx)
                cell_type = cell.get("t")

                if cell_type == "inlineStr":
                    text_node = cell.find("a:is/a:t", ns)
                    value = "" if text_node is None else text_node.text or ""
                else:
                    value_node = cell.find("a:v", ns)
                    value = "" if value_node is None else value_node.text or ""
                    if cell_type == "s" and value != "":
                        value = shared_strings[int(value)]

                values[idx] = value

            rows.append([values.get(i, "") for i in range(max_idx + 1)])

    if not rows:
        return pd.DataFrame()

    header = rows[0]
    records = []
    for row in rows[1:]:
        padded = row + [""] * max(0, len(header) - len(row))
        records.append(
            {
                header[i]: padded[i] if i < len(padded) else ""
                for i in range(len(header))
            }
        )

    return pd.DataFrame(records)


def load_labeled_dataframe(label_file=DEFAULT_LABEL_FILE, image_dir=DEFAULT_IMAGE_DIR):
    df_raw = read_xlsx_basic(label_file)
    required_columns = {"File", "Caption", "Anotasi"}
    missing_columns = required_columns.difference(df_raw.columns)
    if missing_columns:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing_columns)}")

    df = df_raw.copy()
    df["label"] = df["Anotasi"].fillna("").astype(str).str.strip()
    df = df[df["label"].ne("")].copy()
    df["file"] = df["File"].fillna("").astype(str).str.strip()
    df["caption"] = (
        df["Caption"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["image_path"] = df["file"].apply(lambda file_name: Path(image_dir) / file_name)

    missing_images = ~df["image_path"].apply(lambda path: path.exists())
    missing_preview = df.loc[missing_images, ["file", "label"]].head().copy()
    df = df.loc[~missing_images].reset_index(drop=True)

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])
    label_names = list(label_encoder.classes_)
    label_to_id = {label: int(idx) for idx, label in enumerate(label_names)}
    id_to_label = {int(idx): label for idx, label in enumerate(label_names)}

    return {
        "df": df,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "missing_image_count": int(missing_images.sum()),
        "missing_image_preview": missing_preview,
    }


def save_label_distribution(df, output_path, title="Distribusi Label Skema 2"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_counts = df["label"].value_counts().sort_values()
    total_data = int(label_counts.sum())

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(label_counts.index, label_counts.values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Jumlah Data")
    ax.set_ylabel("Label")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    max_count = max(label_counts.values) if len(label_counts) else 0
    for bar, value in zip(bars, label_counts.values):
        percent = 100 * value / total_data if total_data else 0
        ax.text(
            value + max_count * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value} ({percent:.1f}%)",
            va="center",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return label_counts.sort_values(ascending=False).to_frame("count")


def split_labeled_dataframe(df, label_names, test_size=0.2, seed=42):
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label_id"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    split_counts = pd.concat(
        [
            train_df["label"].value_counts().rename("train"),
            val_df["label"].value_counts().rename("validation"),
        ],
        axis=1,
    ).fillna(0).astype(int).loc[label_names]

    return train_df, val_df, split_counts


def labels_to_tensor(df):
    return torch.tensor(df["label_id"].values, dtype=torch.long)

