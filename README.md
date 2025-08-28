# Codec Latent Denoiser

## Setup

```bash
git clone git@github.com:gokulkarthik/codec-latent-denoiser.git
cd codec-latent-denoiser
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

Copy `.env.example` as `.env` and fill the values.`

## Train the model:
```bash
poetry run python src/train.py --config-name=default
```
Configure the hyperparameters in `src/train_configs`