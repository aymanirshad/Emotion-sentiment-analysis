# models/

This directory is a placeholder for your saved model weights.

The fine-tuned DistilBERT model (~260 MB) is too large to commit directly to GitHub.

## Option A — Download from Google Drive

If the team has shared the model on Google Drive, download the zip and unzip it here:

```
models/
└── emotion_model_output/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── vocab.txt
    └── special_tokens_map.json
```

## Option B — Re-train yourself

Run the notebook `notebooks/improved_emotion_baseline_colab_v2.ipynb` end-to-end on Colab (T4 GPU, ~10 minutes). The Trainer saves the best checkpoint automatically to `/content/emotion_model_output/`. Download it using the save cell described in the README.

## Option C — Use Hugging Face Hub (recommended for sharing)

After training, push to HuggingFace Hub so anyone can load it in one line:

```python
# In Colab, after training:
trainer.push_to_hub("your-hf-username/distilbert-semeval2018-emotion")
```

Then load anywhere:
```python
from transformers import pipeline
clf = pipeline("text-classification", model="your-hf-username/distilbert-semeval2018-emotion", top_k=None)
```
