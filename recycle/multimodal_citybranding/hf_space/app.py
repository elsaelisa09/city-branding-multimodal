import json
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
CHECKPOINT_PATH = ARTIFACT_DIR / "best_intermediate_fusion.pt"
METADATA_PATH = ARTIFACT_DIR / "best_intermediate_fusion_metadata.json"
TEXT_MAX_LENGTH = 160


class IntermediateFusionClassifier(nn.Module):
    def __init__(
        self,
        image_dim=512,
        text_dim=768,
        project_dim=256,
        num_classes=4,
        dropout=0.30,
    ):
        super().__init__()
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mlp = nn.Sequential(
            nn.Linear(project_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, image_features, text_features):
        image_projected = self.image_projection(image_features)
        text_projected = self.text_projection(text_features)
        fused = torch.cat([image_projected, text_projected], dim=1)
        return self.mlp(fused)


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_clip_image(clip_model, pixel_values):
    if hasattr(clip_model, "vision_model") and hasattr(clip_model, "visual_projection"):
        vision_outputs = clip_model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        return clip_model.visual_projection(vision_outputs.pooler_output)
    return clip_model.get_image_features(pixel_values=pixel_values)


def build_runtime():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {METADATA_PATH}")

    metadata = load_json(METADATA_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id_to_label = {int(key): value for key, value in metadata["id_to_label"].items()}
    label_names = [id_to_label[index] for index in sorted(id_to_label)]

    clip_processor = CLIPProcessor.from_pretrained(metadata["clip_model_name"])
    clip_model = CLIPModel.from_pretrained(metadata["clip_model_name"]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(metadata["text_model_name"])
    text_model = AutoModel.from_pretrained(metadata["text_model_name"]).to(device)

    classifier = IntermediateFusionClassifier(
        image_dim=int(metadata["image_dim"]),
        text_dim=int(metadata["text_dim"]),
        project_dim=int(metadata["project_dim"]),
        num_classes=int(metadata["num_classes"]),
        dropout=float(metadata.get("dropout", 0.30)),
    ).to(device)
    classifier.load_state_dict(load_state_dict(CHECKPOINT_PATH, device))

    clip_model.eval()
    text_model.eval()
    classifier.eval()

    return {
        "device": device,
        "label_names": label_names,
        "clip_processor": clip_processor,
        "clip_model": clip_model,
        "tokenizer": tokenizer,
        "text_model": text_model,
        "classifier": classifier,
    }


RUNTIME = build_runtime()


@torch.inference_mode()
def predict(image, text):
    if image is None:
        raise gr.Error("Upload gambar terlebih dahulu.")
    if text is None or not str(text).strip():
        raise gr.Error("Isi teks atau caption terlebih dahulu.")

    device = RUNTIME["device"]
    image = ImageOps.exif_transpose(image).convert("RGB")

    image_inputs = RUNTIME["clip_processor"](
        images=image,
        return_tensors="pt",
    )
    pixel_values = image_inputs["pixel_values"].to(device)
    image_features = encode_clip_image(RUNTIME["clip_model"], pixel_values)
    image_features = F.normalize(image_features, dim=-1)

    text_inputs = RUNTIME["tokenizer"](
        [str(text)],
        padding=True,
        truncation=True,
        max_length=TEXT_MAX_LENGTH,
        return_tensors="pt",
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    text_outputs = RUNTIME["text_model"](**text_inputs)
    text_features = mean_pooling(
        text_outputs.last_hidden_state,
        text_inputs["attention_mask"],
    )
    text_features = F.normalize(text_features, dim=-1)

    logits = RUNTIME["classifier"](image_features, text_features)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

    ranked = sorted(
        zip(RUNTIME["label_names"], probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    top_label, top_probability = ranked[0]
    rows = [
        {
            "rank": rank,
            "label": label,
            "probability": probability,
            "percentage": f"{probability * 100:.2f}%",
        }
        for rank, (label, probability) in enumerate(ranked, start=1)
    ]

    return (
        f"{top_label} ({top_probability * 100:.2f}%)",
        pd.DataFrame(rows),
    )


with gr.Blocks(title="City Branding Multimodal Inference") as demo:
    gr.Markdown("# City Branding Multimodal Inference")

    with gr.Row():
        image_input = gr.Image(label="Gambar", type="pil")
        text_input = gr.Textbox(label="Teks / Caption", lines=6)

    predict_button = gr.Button("Prediksi", variant="primary")
    top_prediction = gr.Textbox(label="Prediksi Label")
    probability_table = gr.Dataframe(
        headers=["rank", "label", "probability", "percentage"],
        label="Probabilitas Semua Label",
        interactive=False,
    )

    predict_button.click(
        fn=predict,
        inputs=[image_input, text_input],
        outputs=[top_prediction, probability_table],
    )


if __name__ == "__main__":
    demo.queue().launch()
