# 🎭 Multi-Label Emotion Classification — SemEval-2018 Task 1 E-c

**CS-4112: Deep Learning | Assignment 2**  
Maryam Abbas · Rida Zubair · Ayman Irshad  
FAST-NUCES · Spring 2026

---

## Overview

This repository improves on the official bag-of-words (BoW) baseline for **SemEval-2025 Task 11** (Bridging the Gap in Text-Based Emotion Detection). The original baseline used a simple feedforward neural network with CountVectorizer features. We replace it with **DistilBERT** — a lightweight transformer that understands context, word order, and semantic meaning — fine-tuned on the **SemEval-2018 Task 1 E-c** English dataset for multi-label emotion classification.

### What We Improved

| Component | Baseline (Original) | Our Approach |
|---|---|---|
| Feature extraction | Bag-of-Words (CountVectorizer) | DistilBERT contextual embeddings |
| Model | 1-hidden-layer FFN (100 units) | `distilbert-base-uncased` fine-tuned |
| Labels | 5 emotions | 11 emotions |
| Optimizer | SGD (lr=0.1) | AdamW (lr=2e-5) with warmup |
| Training | 1000 epochs, full-batch | 5 epochs, mini-batch (bs=16) |
| Imbalance handling | Frequency-based pos weights | Negative/positive ratio pos weights |
| Threshold | Fixed 0.45 | Per-label tuned thresholds |
| Early stopping | ❌ | ✅ (patience=2) |

### 11 Emotion Labels

`anger` · `anticipation` · `disgust` · `fear` · `joy` · `love` · `optimism` · `pessimism` · `sadness` · `surprise` · `trust`

---

## Results

### Baseline vs. Our Model (Validation Set)

| Metric | BoW Baseline (5 labels) | DistilBERT — Default θ=0.5 | DistilBERT — Tuned θ |
|---|---|---|---|
| Micro F1 | 0.5415 | **0.5986** | 0.5284 |
| Macro F1 | 0.4382 | 0.3141 | **0.5098** |
| Micro Precision | 0.4366 | 0.7691 | 0.7353 |
| Micro Recall | 0.7126 | 0.4901 | 0.4124 |
| Macro Precision | 0.3652 | 0.3487 | 0.7144 |
| Macro Recall | 0.6011 | 0.2933 | 0.4576 |

> **Note:** Macro-F1 is the primary metric for this task (penalises poor minority-class performance equally). Tuned thresholds per label give the best macro-F1.

### Per-Label Results (Tuned Thresholds)

| Label | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|
| joy | — | — | 0.827 | 0.25 |
| anger | — | — | 0.734 | 0.41 |
| disgust | — | — | 0.725 | 0.43 |
| optimism | — | — | 0.721 | 0.39 |
| sadness | — | — | 0.618 | 0.41 |
| love | — | — | 0.525 | 0.29 |
| fear | — | — | 0.512 | 0.29 |
| pessimism | — | — | 0.395 | 0.21 |
| anticipation | — | — | 0.304 | 0.17 |
| trust | — | — | 0.172 | 0.09 |
| surprise | — | — | 0.076 | 0.05 |

---

## Repository Structure

```
emotion-classification/
├── notebooks/
│   └── improved_emotion_baseline_colab_v2.ipynb   # Main training notebook
├── models/
│   └── README.md                                   # Instructions to download saved model
├── reports/
│   └── A2_Report.pdf                               # Assignment 2 report
├── requirements.txt
└── README.md
```

> **Note on model weights:** The fine-tuned DistilBERT model (~260 MB) is too large to store directly in Git. See the [Loading the Saved Model](#loading-the-saved-model) section below.

---

## Quickstart (Google Colab — Recommended)

The notebook is designed to run end-to-end on **Google Colab with a free T4 GPU**.

### Step 1 — Open in Colab

Click: **Runtime → Change runtime type → T4 GPU**

Then upload or open `notebooks/improved_emotion_baseline_colab_v2.ipynb`.

### Step 2 — Set up Kaggle credentials

The notebook downloads the dataset from Kaggle automatically. You need a Kaggle account and API key.

1. Go to [kaggle.com](https://www.kaggle.com) → Account → API → **Create New API Token**
2. This downloads `kaggle.json`. When the notebook prompts, upload this file.

### Step 3 — Run all cells

The notebook will:
1. Install dependencies
2. Download the SemEval-2018 E-c dataset automatically via `kagglehub`
3. Tokenize with `distilbert-base-uncased`
4. Compute class weights for label imbalance
5. Fine-tune DistilBERT for 5 epochs with early stopping
6. Tune per-label prediction thresholds on the dev set
7. Report macro-F1, micro-F1, precision, recall, and per-label breakdown
8. **Save the best model** to `/content/emotion_model_output/`

---

## Saving and Downloading the Model (Colab)

After training finishes, **add this cell at the end of your notebook** to save the model and zip it for download:

```python
# === CELL: Save model and download ===
import shutil
from google.colab import files

# The Trainer already saves the best model to OUTPUT_DIR during training.
# This zips it so you can download it.
SAVE_PATH = "/content/emotion_model_output"   # where Trainer saved checkpoints
ZIP_NAME  = "/content/distilbert_emotion_model"

shutil.make_archive(ZIP_NAME, "zip", SAVE_PATH)
files.download(ZIP_NAME + ".zip")
print("✅ Model saved and download started.")
```

> The zip will contain `model.safetensors`, `config.json`, `tokenizer_config.json`, and related files.

### Alternatively — Save to Google Drive

```python
from google.colab import drive
import shutil

drive.mount("/content/drive")

src = "/content/emotion_model_output"
dst = "/content/drive/MyDrive/emotion_model_output"
shutil.copytree(src, dst, dirs_exist_ok=True)
print("✅ Model copied to Google Drive.")
```

---

## Loading the Saved Model

Once you have the saved model folder (downloaded zip or Drive), you can load it anywhere:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL_PATH = "./emotion_model_output"   # path to your unzipped folder

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

LABEL_COLUMNS = [
    "anger", "anticipation", "disgust", "fear", "joy", "love",
    "optimism", "pessimism", "sadness", "surprise", "trust"
]

# Tuned thresholds from training
THRESHOLDS = {
    "anger": 0.41, "anticipation": 0.17, "disgust": 0.43,
    "fear": 0.29,  "joy": 0.25,         "love": 0.29,
    "optimism": 0.39, "pessimism": 0.21, "sadness": 0.41,
    "surprise": 0.05, "trust": 0.09
}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze().numpy()
    probs = 1 / (1 + np.exp(-logits))   # sigmoid
    return {
        label: bool(probs[i] >= THRESHOLDS[label])
        for i, label in enumerate(LABEL_COLUMNS)
    }

print(predict("I'm so excited about this, but also a little nervous!"))
```

---

## Running Locally (without Colab)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/emotion-classification.git
cd emotion-classification

# 2. Create environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up Kaggle (needed for dataset download)
#    Place your kaggle.json at ~/.kaggle/kaggle.json

# 5. Launch the notebook
jupyter notebook notebooks/improved_emotion_baseline_colab_v2.ipynb
```

> ⚠️ A GPU is strongly recommended. On CPU, training will be very slow (~2+ hours vs ~10 mins on T4).

---

## Where to Edit the Notebook

All key hyperparameters are in **Cell 3** (the CONFIG block):

```python
# === Cell 3 — CONFIG (edit here) ===
MODEL_NAME         = "distilbert-base-uncased"   # swap for "bert-base-uncased", etc.
LEARNING_RATE      = 2e-5
EPOCHS             = 5
MAX_LEN            = 128
TRAIN_BATCH_SIZE   = 16
VALID_BATCH_SIZE   = 32
WEIGHT_DECAY       = 0.01
WARMUP_RATIO       = 0.1
EARLY_STOPPING_PATIENCE = 2
SEED               = 42
OUTPUT_DIR         = "/content/emotion_model_output"
```

To add model-saving + download, insert a new cell **after the final evaluation cell**.

---

## Dependencies

```
torch>=2.0
transformers>=4.40
datasets
accelerate
scikit-learn
pandas
numpy
kagglehub
sentencepiece
iterative-stratification
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## References

- Mohammad, S. M. et al. (2018). SemEval-2018 Task 1: Affect in Tweets. *Proceedings of SemEval-2018.*
- de Kock, C. (2025). SemEval-2025 Task 11 Starter Notebook. [GitHub](https://github.com/sXpXr/semeval-task-11)
- Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108.*
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL-HLT 2019.*
- Wolf, T. et al. (2020). HuggingFace Transformers. *EMNLP 2020.*

---

## Team

| Name | Student ID |
|---|---|
| Maryam Abbas | 23i-6004 |
| Rida Zubair | 23i-2590 |
| Ayman Irshad | 23f-0724 |

**Course:** CS-4112 Deep Learning · FAST-NUCES · Spring 2026
