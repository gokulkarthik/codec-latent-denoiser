# Codec Latent Denoiser
This project explores speech enhancement in the latent space of neural audio codecs (DAC) by training a lightweight denoiser on continuous quantized representations, inspired from the work ["Efficient Speech Enhancement via Embeddings from Pre-trained Generative Audioencoders"](https://arxiv.org/pdf/2506.11514). 

## Structure

- `apps/`: Application files
  - `demo.py`: Streamlit demo application for interactive model testing
- `notebooks/`: Jupyter notebooks for experiments and analysis
  - `demo.ipynb`: Interactive demo notebook
- `outputs/`: Directory for model outputs, checkpoints, and logs
- `src/`: Main source code
  - `codec_latent_denoiser/`: Core package
    - `__init__.py`: Package initialization and exports
    - `config.py`: Configuration classes for model components
    - `model.py`: Core model implementation (CodecLatentDenoiser, MLPDenoiser)
    - `processor.py`: Audio processing utilities
  - `lightning_utils.py`: PyTorch Lightning utilities (modules, datasets, callbacks)
  - `train.py`: Training script
  - `train_configs/`: Configuration files for training
    - `default.yaml`: Default training configuration

## Setup

```bash
git clone git@github.com:gokulkarthik/codec-latent-denoiser.git
cd codec-latent-denoiser
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```
Copy `.env.example` as `.env` and fill the values.`


## Usage 

### Train the model:
```bash
poetry run python src/train.py --config-name=default
```
Configure the hyperparameters in `src/train_configs`

### Start the demo:
```bash
poetry run streamlit run apps/demo.py
```