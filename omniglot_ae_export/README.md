# Omniglot Autoencoder Decoder

Standalone decoder for generating 56×56 Omniglot character images from 32-dimensional latent codes.

## Files

- **`decoder_traced.pt`** (0.25 MB) - TorchScript traced model, ready to use
- **`decoder_weights.pt`** - State dict (for reference)

## Quick Start

```python
import torch

# Load traced model
decoder = torch.jit.load('decoder_traced.pt')
decoder.eval()

# Generate images from random latent codes
z = torch.randn(10, 32)  # 10 images
images = decoder(z)  # Output: (10, 1, 56, 56)

# Save
from PIL import Image
img = (images[0, 0].cpu().numpy() * 255).astype('uint8')
Image.fromarray(img).save('sample.png')
```

## Model Specs

- **Input**: 32-dimensional latent code
- **Output**: 56×56 grayscale image (single channel)
- **Parameters**: 62,137
- **Architecture**: 3 transposed convolutions + 1 linear layer
- **Activation**: SiLU (smooth activation)

## Usage

The decoder takes a 32-dim latent vector and outputs a 56×56 image:

```python
z = torch.randn(batch_size, 32)  # Input
img = decoder(z)  # (batch_size, 1, 56, 56) output in [0, 1]
```

## No Dependencies

Only requires PyTorch. Uses TorchScript, so no architecture definition needed.

## From

Extracted from the Omniglot autoencoder (118K total params). This decoder alone is perfect for:
- Image generation from latent codes
- VAE decoder replacement
- Lightweight generative models
- Mobile/edge deployment
