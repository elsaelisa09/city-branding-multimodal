import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor


RESAMPLE = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def light_image_augment(image):
    if random.random() < 0.35:
        image = ImageOps.mirror(image)

    if random.random() < 0.45:
        width, height = image.size
        scale = random.uniform(0.92, 1.0)
        crop_width = max(1, int(width * scale))
        crop_height = max(1, int(height * scale))
        left = random.randint(0, max(0, width - crop_width))
        top = random.randint(0, max(0, height - crop_height))
        image = image.crop((left, top, left + crop_width, top + crop_height)).resize(
            (width, height), RESAMPLE
        )

    if random.random() < 0.50:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.92, 1.08))
    if random.random() < 0.50:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.92, 1.08))
    if random.random() < 0.35:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.92, 1.08))

    return image


def load_image(path, augment=False):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if augment:
        image = light_image_augment(image)
    return image


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class MultimodalFeatureExtractor:
    def __init__(self, clip_model_name, text_model_name, device):
        self.clip_model_name = clip_model_name
        self.text_model_name = text_model_name
        self.device = device

        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name).to(device)

        self.clip_model.eval()
        self.text_model.eval()

        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

    @property
    def image_dim(self):
        return int(self.clip_model.config.projection_dim)

    @property
    def text_dim(self):
        return int(self.text_model.config.hidden_size)

    def _clip_image_output_to_tensor(self, output):
        if torch.is_tensor(output):
            return output

        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            pooled_output = output.pooler_output
            if (
                hasattr(self.clip_model, "visual_projection")
                and pooled_output.shape[-1]
                == self.clip_model.visual_projection.in_features
            ):
                return self.clip_model.visual_projection(pooled_output)
            return pooled_output

        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item

        raise TypeError(f"Output CLIP tidak bisa dikonversi ke tensor: {type(output)}")

    def _encode_clip_images(self, pixel_values):
        if hasattr(self.clip_model, "vision_model") and hasattr(
            self.clip_model, "visual_projection"
        ):
            vision_outputs = self.clip_model.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
            return self.clip_model.visual_projection(vision_outputs.pooler_output)

        return self._clip_image_output_to_tensor(
            self.clip_model.get_image_features(pixel_values=pixel_values)
        )

    @torch.no_grad()
    def extract_image_features(
        self,
        image_paths,
        augment=False,
        batch_size=16,
        desc="image features",
    ):
        features = []
        self.clip_model.eval()

        for start in tqdm(range(0, len(image_paths), batch_size), desc=desc):
            batch_paths = image_paths[start : start + batch_size]
            images = [load_image(path, augment=augment) for path in batch_paths]
            inputs = self.clip_processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            batch_features = self._encode_clip_images(pixel_values)
            batch_features = F.normalize(batch_features, dim=-1)
            features.append(batch_features.cpu())

        return torch.cat(features, dim=0)

    @torch.no_grad()
    def extract_text_features(
        self,
        captions,
        batch_size=16,
        max_length=160,
        desc="text features",
    ):
        features = []
        self.text_model.eval()
        captions = [caption if str(caption).strip() else " " for caption in captions]

        for start in tqdm(range(0, len(captions), batch_size), desc=desc):
            batch_text = captions[start : start + batch_size]
            inputs = self.tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.text_model(**inputs)
            batch_features = mean_pooling(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )
            batch_features = F.normalize(batch_features, dim=-1)
            features.append(batch_features.cpu())

        return torch.cat(features, dim=0)

