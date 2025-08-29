# Codec Latent Denoiser

This project explores speech enhancement in the latent space of neural audio codecs (DAC) by training a denoiser neural network on continuous quantized representations, inspired by the work ["Efficient Speech Enhancement via Embeddings from Pre-trained Generative Audioencoders"](https://arxiv.org/pdf/2506.11514). Instead of processing raw audio signals, this method works with the compressed representations, offering potential efficiency gains while maintaining high-quality denoising performance.

## Project Structure

```
├── apps/
│   └── demo.py                # Streamlit demo application for interactive model testing
├── notebooks/
│   └── demo.ipynb             # Interactive demo notebook
├── src/
│   ├── codec_latent_denoiser/ # Core package
│   │   ├── __init__.py        # Package initialization and exports
│   │   ├── config.py          # Configuration classes for model components
│   │   ├── model.py           # Core model implementation (CodecLatentDenoiser, MLPDenoiser)
│   │   └── processor.py       # Audio processing utilities
│   ├── lightning_utils.py     # PyTorch Lightning utilities (modules, datasets, callbacks)
│   ├── train.py               # Training script
│   └── train_configs/         # Configuration files for training
```

## Installation

```bash
git clone git@github.com:gokulkarthik/codec-latent-denoiser.git
cd codec-latent-denoiser
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

Copy `.env.example` as `.env` and fill in the required values.

## Usage

### Training

To train the model:

```bash
poetry run python src/train.py --config-name=e2
```

Configure the hyperparameters in `src/train_configs/` directory.

### Speech Denoising

The primary use case is removing noise from speech audio. Here's how to use the model for denoising:

```python
import torch
from codec_latent_denoiser import CodecLatentDenoiser, CodecLatentDenoiserProcessor

# Set model checkpoint path
model_path = "gokulkarthik/codec-latent-denoiser-e2"

# Load model and processor
processor = CodecLatentDenoiserProcessor.from_pretrained(model_path)
model = CodecLatentDenoiser.from_pretrained(model_path).eval()

# Prepare input audio
# noisy_audio should be a torch.Tensor of shape [num_audio_samples] 
# at the sampling rate of processor.sampling_rate
noisy_audio = your_noisy_audio_tensor  # Replace with your audio data

# Denoise the audio
with torch.inference_mode():
    inputs = processor(noisy_audio)["input_values"]
    clean_audio = model(inputs, denoise=True, decode=True).audio_generated[0][0]
```

For more detailed examples and interactive exploration, check the Jupyter notebook at `notebooks/demo.ipynb`.

### Interactive Demo

Launch the Streamlit demo application:

```bash
poetry run streamlit run apps/demo.py
```

This provides a web interface for testing the model with multiple audio files interactively.