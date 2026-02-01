# TrOCR Offline Model Loading Guide

This guide explains how to download and use the TrOCR model locally without relying on HuggingFace API access. This is useful when:

- Running in offline or restricted network environments
- Dealing with conflicting dependency versions (e.g., TrOCR needs newer transformers, but doctr needs older)
- Avoiding repeated downloads and network latency

## Quick Start

### Step 1: Download the Model (requires internet)

Run this on a machine with internet access:

```bash
python download_trocr_model.py
```

This downloads and saves the model to `./trocr_model/` by default. Customize the output location:

```bash
python download_trocr_model.py --output_dir /path/to/save/trocr
```

The script downloads:
- Model weights (`trocr_model/model/`)
- Tokenizer (`trocr_model/tokenizer/`)
- Feature extractor (`trocr_model/feature_extractor/`)

### Step 2: Load the Model (offline)

Use the offline loader in your code:

```python
from load_trocr_offline import load_trocr_model

# Simple function-based approach
model, tokenizer, feature_extractor = load_trocr_model("./trocr_model")

# Or use the class-based loader
from load_trocr_offline import TrOCRLoader

loader = TrOCRLoader("./trocr_model")
model = loader.model
tokenizer = loader.tokenizer
feature_extractor = loader.feature_extractor
```

### Step 3: Use the Model

```python
from PIL import Image

# Option 1: Direct inference
image = Image.open("plate.png").convert("RGB")
pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Option 2: Use the helper method
from load_trocr_offline import TrOCRLoader

loader = TrOCRLoader("./trocr_model")
text = loader.recognize_text("plate.png")
```

## Two Scripts Included

### `download_trocr_model.py`

Downloads the TrOCR model from HuggingFace and saves all components locally.

**Requirements:**
- Internet connection
- transformers library (any recent version)

**Usage:**
```bash
python download_trocr_model.py [--output_dir ./trocr_model]
```

**Output:**
```
trocr_model/
├── model/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── tokenizer/
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── ...
└── feature_extractor/
    ├── preprocessor_config.json
    └── ...
```

### `load_trocr_offline.py`

Loads the locally saved TrOCR model without any internet access.

**Classes:**

#### `TrOCRLoader(model_dir)`

Context manager-like class for loading model components.

```python
loader = TrOCRLoader("./trocr_model")

# Properties auto-load and cache components
model = loader.model
tokenizer = loader.tokenizer
feature_extractor = loader.feature_extractor

# Helper method for text recognition
text = loader.recognize_text(image)
```

#### `load_trocr_model(model_dir)`

Simple function that returns a tuple of components:

```python
model, tokenizer, feature_extractor = load_trocr_model("./trocr_model")
```

## Deployment Workflow

### Machine with Internet (Download):
```bash
python download_trocr_model.py --output_dir ./trocr_model
tar -czf trocr_model.tar.gz trocr_model/
# Transfer trocr_model.tar.gz to target machine
```

### Target Machine (Use Offline):
```bash
tar -xzf trocr_model.tar.gz
# Now use load_trocr_offline.py to load the model
```

## Model Specifications

- **Model:** microsoft/trocr-small-printed
- **Type:** Vision-to-Sequence (Image → Text)
- **Input:** RGB images (variable size, processed by feature extractor)
- **Output:** Text recognition of printed text on images
- **Size:** ~300MB (model weights)

## Requirements

**For downloading:**
- transformers (any recent version)
- torch
- PIL/Pillow

**For loading offline:**
- transformers
- torch
- PIL/Pillow
- (No internet connection needed!)

## Environment Variables

Set these before importing transformers if running in truly offline mode:

```bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

## Troubleshooting

### Error: "Model directory not found"
Make sure you've run `download_trocr_model.py` and the output directory exists.

### Error: "Missing model components"
The directory exists but is incomplete. Run `download_trocr_model.py` again to re-download.

### Model takes a long time to load first time
Components are cached on first load and reused. Subsequent calls are instant.

### GPU not being used
Make sure you move the model to GPU after loading:

```python
loader = TrOCRLoader("./trocr_model")
model = loader.model.to("cuda")  # Move to GPU
```

## Comparison with HuggingFace API

| Aspect | HuggingFace API | Offline Local |
|--------|-----------------|---------------|
| **Network Required** | Yes | No |
| **First Load Speed** | Slower (download) | Depends on storage |
| **Subsequent Loads** | Fast (cached) | Fast (cached) |
| **Dependency Conflicts** | Can occur | None (fixed version) |
| **Storage** | Varies (~300MB) | ~300MB |
| **Setup** | Single line | Download + load |

## Notes

- The offline loader uses `local_files_only=True` to prevent any network calls
- Model weights are binary files (`.bin`), not text
- The feature extractor handles image preprocessing automatically
- Tokenizer handles text decoding with special token removal
