# Laporan Model Multimodal Intermediate Fusion untuk City Branding Classification

## 1. Ringkasan Eksekutif

Model ini menggunakan **Intermediate Fusion** untuk menggabungkan fitur visual dari gambar dan fitur semantik dari teks dalam mengklasifikasikan image berdasarkan kategori city branding. Model dilatih menggunakan **CLIP ViT-B/32** untuk ekstraksi fitur gambar dan **Twitter XLM-RoBERTa** untuk ekstraksi fitur teks, kemudian digabung dan diklasifikasikan dengan MLP 3-layer.

---

## 2. Arsitektur Model

### 2.1 Overview Pipeline

```
Input Gambar → CLIP ViT-B/32 → Image Features (512 dims)
                                     ↓
                            Image Projection (→256 dims)
                                     ↓
                                    Concat (512 dims)
                                     ↓
Input Teks → Twitter XLM-RoBERTa → Text Features (768 dims)
                                     ↓
                            Text Projection (→256 dims)
                                     ↓
                                     MLP Classifier
                                     ↓
                              Output (4 classes)
```

### 2.2 Komponen Utama

#### A. **Image Feature Extractor: CLIP ViT-B/32**

- **Model**: OpenAI CLIP Vision Transformer Base (Patch 32)
- **Output Dimensionality**: 512
- **Preprocessing**:
  - Konversi ke RGB
  - Auto-orient berdasarkan EXIF data
  - Resize ke 224×224 (standar CLIP)
- **Frozen**: Ya (tidak di-fine-tune)

#### B. **Text Feature Extractor: Twitter XLM-RoBERTa**

- **Model**: CardiffNLP Twitter XLM-RoBERTa Base
- **Output Dimensionality**: 768 (hidden state)
- **Preprocessing**:
  - Tokenisasi dengan max length 160
  - Padding & truncation
  - Mean pooling dengan attention mask
- **Frozen**: Ya (tidak di-fine-tune)

#### C. **Image Projection Layer**

```
Linear (512 → 256)
     ↓
LayerNorm
     ↓
GELU Activation
     ↓
Dropout (30%)
```

#### D. **Text Projection Layer**

```
Linear (768 → 256)
     ↓
LayerNorm
     ↓
GELU Activation
     ↓
Dropout (30%)
```

#### E. **Classification Head (MLP)**

```
Input (512 dims, concat dari kedua projection)
     ↓
Linear (512 → 256) + LayerNorm + GELU + Dropout(30%)
     ↓
Linear (256 → 128) + LayerNorm + GELU + Dropout(30%)
     ↓
Linear (128 → num_classes=4)
```

### 2.3 Strategi Fusioning

**Intermediate Fusion**: Fitur gambar dan teks diproyeksikan ke dimensi yang sama (256), kemudian di-concatenate sebelum melewati classification head. Ini berbeda dengan:

- **Early Fusion**: Menggabung raw inputs (tidak optimal)
- **Late Fusion**: Menggabung predictions dari 2 model terpisah (kehilangan interaksi)

---

## 3. Pemrosesan Data

### 3.1 Data Loading & Preprocessing

**Sumber Data**:

- File metadata: `dataset/labels_skema2.xlsx`
- Direktori gambar: `dataset/images_skema2/`
- Format: Excel dengan kolom [File, Caption, Anotasi]

**Cleaning Steps**:

```python
1. Remove rows dengan label kosong
2. Verify image files exist
3. Normalize captions (remove extra whitespace)
4. Label encoding: Anotasi → numeric IDs (0-3)
```

**Dataset Split**:

- Training: 80% (stratified by label)
- Validation: 20% (stratified by label)
- Random seed: 42 (reproducibility)

### 3.2 Image Preprocessing & Augmentation

#### Training Set (dengan augmentasi):

```
1. EXIF orientation correction
2. Convert to RGB
3. Random horizontal flip (35% probability)
4. Random crop & resize (45% probability)
5. Random brightness adjustment ±8% (50% probability)
6. Random contrast adjustment ±8% (50% probability)
7. Random color adjustment ±8% (35% probability)
```

#### Validation Set (deterministik):

```
1. EXIF orientation correction
2. Convert to RGB
3. Tanpa augmentasi (reproducible)
```

**Augmentation rationale**: Meningkatkan generalisasi dengan memperkaya variasi training samples tanpa mengubah semantic meaning.

### 3.3 Text Preprocessing

```python
Captions:
  ├─ Handle missing values → replace with ' '
  ├─ Tokenize dengan AutoTokenizer
  ├─ Max length: 160 tokens
  ├─ Padding & truncation: enabled
  ├─ Return: input_ids, attention_mask

Feature Pooling:
  └─ Mean pooling dengan attention mask
     (hanya token yang relevant dihitung)
```

---

## 4. Feature Extraction

### 4.1 Image Features (CLIP ViT-B/32)

**Process**:

```
1. Load gambar dengan PIL → PIL.Image
2. Apply augmentasi (jika training)
3. Preprocess dengan CLIPProcessor
   └─ Normalize ke [0, 1]
   └─ Resize ke 224×224
   └─ Permute ke (C, H, W)
4. Forward ke CLIP vision encoder
5. Get pooled output → 512 dims
6. L2 normalize (F.normalize, dim=-1)
```

**Output**: Tensor (batch_size, 512) - normalized embeddings

### 4.2 Text Features (Twitter XLM-RoBERTa)

**Process**:

```
1. Tokenize captions
2. Pad/truncate to max_length=160
3. Forward ke text encoder
   └─ Get last_hidden_state (batch_size, seq_len, 768)
4. Mean pooling dengan attention mask
   └─ Mask invalid tokens
   └─ Average valid token embeddings
5. L2 normalize (F.normalize, dim=-1)
```

**Output**: Tensor (batch_size, 768) - normalized embeddings

### 4.3 Normalisasi L2

Kedua fitur di-normalize L2 untuk:

- Consistency dalam representasi
- Better gradient flow during training
- Lebih robust terhadap scale variations

---

## 5. Training Configuration

### 5.1 Hyperparameters

| Parameter               | Value             | Alasan                                                   |
| ----------------------- | ----------------- | -------------------------------------------------------- |
| Batch Size              | 16                | Balance antara memory & gradient stability               |
| Learning Rate           | 2e-4              | Low LR untuk fine-tuning classifier saja                 |
| Weight Decay            | 1e-4              | L2 regularization untuk prevent overfitting              |
| Epochs                  | 24                | Max epochs sebelum early stopping                        |
| Early Stopping Patience | 5                 | Stop jika validation loss tidak improve                  |
| Optimizer               | AdamW             | Adaptive learning rate dengan weight decay               |
| Scheduler               | ReduceLROnPlateau | Reduce LR jika val loss plateau (factor=0.5, patience=2) |
| Dropout Rate            | 0.30              | Moderate dropout untuk regularization                    |

### 5.2 Loss Function: Focal Loss

**Formula**:
$$L_{focal} = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$

Dimana:

- $p_t$ = model's estimated probability untuk true class
- $\gamma$ = focusing parameter (set to 2.0)
- $\alpha_t$ = class weight untuk class $t$

**Mengapa Focal Loss?**

- Dataset imbalanced (beberapa label less frequent)
- Focal loss down-weight easy examples, focus pada hard examples
- Mencegah model bias terhadap majority class

**Class Weights** (computed dari train data):

```
alpha = class_distribution_inverse / mean(class_distribution_inverse)
```

---

## 6. Pipeline Training

### 6.1 Training Loop

```python
For each epoch:
  1. Forward pass: (img_features, txt_features) → logits
  2. Compute focal loss
  3. Backward pass dengan gradient clipping (max_norm=1.0)
  4. Update weights dengan optimizer
  5. Track: loss, accuracy, macro-F1

Validation Loop:
  1. Forward pass (no grad)
  2. Compute metrics
  3. If val_loss < best_loss:
       └─ Save checkpoint
       └─ Save metadata
  4. Adjust learning rate via scheduler
```

### 6.2 Early Stopping

Model berhenti training jika:

- Validation loss tidak improve untuk 5 consecutive epochs
- Atau mencapai 24 epochs
- Best checkpoint di-load untuk inference

---

## 7. Dataset & Label Distribution

### 7.1 Data Statistics

```
Total samples: [jumlah tergantung file]
Split:
  - Training: 80% (stratified)
  - Validation: 20% (stratified)

Labels: 4 classes
  - Class 0: [Label name]
  - Class 1: [Label name]
  - Class 2: [Label name]
  - Class 3: [Label name]
```

**Stratified split** memastikan distribution label konsisten di train & val sets.

---

## 8. Evaluasi & Hasil

### 8.1 Metrics

Selama training, kami track:

| Metric       | Definition                  | Ketika digunakan                    |
| ------------ | --------------------------- | ----------------------------------- |
| **Loss**     | Focal Loss                  | Untuk optimization & early stopping |
| **Accuracy** | Correct predictions / Total | Overall performance                 |
| **Macro-F1** | Mean F1 per class           | Handle class imbalance              |

### 8.2 Validation Performance

Hasil dievaluasi pada validation set dengan:

1. **Classification Report**:

   ```
   - Precision: Dari predicted positives, berapa yg benar?
   - Recall: Dari actual positives, berapa yg terdeteksi?
   - F1-Score: Harmonic mean precision & recall
   - Support: Jumlah samples per class
   ```

2. **Confusion Matrix**:

   ```
   - Menunjukkan true positives vs false positives
   - Membantu identify classes yang sering salah klasifikasi
   - Visualisasi heatmap untuk easy interpretation
   ```

3. **Prediction Output**:
   ```
   Saved ke: ouputmodel/validation_predictions.csv
   Columns: [file, caption, label, prediction, correct]
   ```

### 8.3 Training History

**Saved ke**: `ouputmodel/training_history.csv`

Track per epoch:

- Train Loss / Accuracy / Macro-F1
- Validation Loss / Accuracy / Macro-F1
- Learning rate (updated via scheduler)

**Learning curves** divisualisasikan:

- Loss convergence
- Accuracy improvement
- F1-score trends

---

## 9. Output & Artifacts

Semua hasil training disimpan di `ouputmodel/`:

| File                                     | Description                    |
| ---------------------------------------- | ------------------------------ |
| `best_intermediate_fusion.pt`            | Model weights (state dict)     |
| `best_intermediate_fusion_metadata.json` | Model config & hyperparameters |
| `training_history.csv`                   | Training metrics per epoch     |
| `label_distribution.png`                 | Bar chart label frequencies    |
| `training_curves.png`                    | Loss & accuracy plots          |
| `confusion_matrix.png`                   | Heatmap confusion matrix       |
| `validation_predictions.csv`             | Per-sample predictions         |

---

## 10. Troubleshooting & Insights

### 10.1 Error dalam Code

Jika terdapat error `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`:

**Root Cause**: Fungsi `F.normalize()` hanya bekerja dengan tensors, bukan model outputs.

**Solution**: Extract tensor dari output terlebih dahulu:

```python
# ❌ SALAH:
batch_features = clip_model.get_image_features(pixel_values=pixel_values)
batch_features = F.normalize(batch_features, dim=-1)

# ✅ BENAR:
batch_features = clip_model.get_image_features(pixel_values=pixel_values)
if hasattr(batch_features, 'last_hidden_state'):  # or image_embeds
    batch_features = batch_features.last_hidden_state
batch_features = F.normalize(batch_features, dim=-1)
```

Sudah di-fix dalam helper function `encode_clip_images()` dan `clip_image_output_to_tensor()`.

### 10.2 Best Practices

✅ **Dilakukan dengan baik**:

- Stratified split untuk handle imbalanced data
- Augmentasi hanya di training set
- Frozen pre-trained encoders (CLIP & XLM-RoBERTa)
- L2 normalization untuk fitur consistency
- Focal loss untuk imbalanced learning
- Gradient clipping untuk training stability
- Early stopping untuk prevent overfitting

💡 **Improvement potential**:

- Hyperparameter tuning via grid/random search
- Ensemble dengan LateFusion model
- Fine-tune dengan lower LR pada pretrained encoders
- Advanced augmentations (mixup, cutmix)
- Cross-validation untuk robust metrics

---

## 11. Kesimpulan

Model **Multimodal Intermediate Fusion** menggabungkan kekuatan:

- **CLIP**: Understanding visual content dengan semantic alignment
- **XLM-RoBERTa**: Multilingual text understanding
- **Intermediate Fusion**: Capture interaction antara visual & textual modalities

Architecture ini optimal untuk tasks yang require reasoning tentang kedua modalities secara bersamaan, seperti city branding classification yang melibatkan visual aesthetics dan textual description.

---

## Appendix: Key Configuration

```python
# Model Architecture
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
TEXT_MODEL_NAME = 'cardiffnlp/twitter-xlm-roberta-base'
PROJECT_DIM = 256
TEXT_MAX_LENGTH = 160

# Training
BATCH_SIZE = 16
EPOCHS = 24
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

# Hardware
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
IMAGE_DIR = Path('dataset/images_skema2')
LABEL_FILE = Path('dataset/labels_skema2.xlsx')
OUTPUT_DIR = Path('ouputmodel')
```
