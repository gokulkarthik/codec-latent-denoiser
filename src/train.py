import hydra
import lightning as L
import logging
import sys
import torch

from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from codec_latent_denoiser import CodecLatentDenoiserProcessor
from lightning_utils import (
    CodecLatentDenoiserLightningDataModule,
    CodecLatentDenoiserLightningModule,
    HuggingFaceHubPushCallback
)


@hydra.main(version_base=None, config_path="train_configs", config_name="default")
def train(training_config: DictConfig) -> None:
    """Main training function."""
    try:
        training_config_name = sys.argv[1].split("=")[1]
    except IndexError:
        training_config_name = "default"

    logging.info("Loading model...")
    processor = CodecLatentDenoiserProcessor.from_pretrained(
        training_config.get("pretrained_codec_path", "descript/dac_16khz")
    )
    model = CodecLatentDenoiserLightningModule(
        pretrained_codec_path=training_config.get("pretrained_codec_path", "descript/dac_16khz"),
        learning_rate=training_config.get("learning_rate", 1e-3),
        weight_decay=training_config.get("weight_decay", 1e-5),
        train_only_denoiser=training_config.get("train_only_denoiser", True),
    )
    logging.info("Model loaded successfully.")

    logging.info("Loading data module...")
    data_module = CodecLatentDenoiserLightningDataModule(
        data_path_hf_hub=training_config.get(
            "data_path_hf_hub", "JacobLinCool/VoiceBank-DEMAND-16k"
        ),
        clean_speech_key=training_config.get("clean_speech_key", "clean"),
        noisy_speech_key=training_config.get("noisy_speech_key", "noisy"),
        processor=processor,
        batch_size=training_config.get("batch_size", 16),
    )
    logging.info("Data module loaded successfully.")

    logging.info("Setting trainer...")
    wandb_logger = WandbLogger(
        project="Codec-Latent-Denoiser",
        config=OmegaConf.to_container(training_config, resolve=True),
        entity=training_config.get("wandb_entity", "gokulkarthik"),
        name=training_config_name
        + "-"
        + datetime.now(timezone(timedelta(hours=4))).strftime("%y%m%d-%H%M"),
        log_model=True,
        tags=training_config.get("wandb_tags", []),
        notes=training_config.get("wandb_notes", ""),
    )

    trainer = L.Trainer(
        enable_checkpointing=False,
        callbacks=[],
        logger=wandb_logger,
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
        max_epochs=training_config.get("max_epochs", 10),
        precision=training_config.get("precision", "bf16-mixed"),
        num_nodes=training_config.get("num_nodes", 1),
        devices=training_config.get("devices", [0]),
        limit_train_batches=training_config.get("limit_train_batches", None),
        limit_val_batches=training_config.get("limit_val_batches", None),
        limit_test_batches=training_config.get("limit_test_batches", None),
        log_every_n_steps=training_config.get("log_every_n_steps", 100),
        val_check_interval=training_config.get("val_check_interval", 1.0),
        check_val_every_n_epoch=training_config.get("check_val_every_n_epoch", 1),
    )
    if trainer.is_global_zero:
        hf_hub_entity = training_config.get("hf_hub_entity", "gokulkarthik"),
        push_callback = HuggingFaceHubPushCallback(
            repo_id=training_config.get("ckpt_path", f"{hf_hub_entity}/codec-latent-denoiser-{training_config_name}"),
            push_every_n_epochs=training_config.get("push_every_n_epochs", 10),
            processor=processor,
        )
        trainer.callbacks.append(push_callback)
    logging.info("Trainer set successfully.")

    logging.info("Training...")
    trainer.fit(model, data_module)
    logging.info("Training completed.")

    logging.info("Testing...")
    trainer.test(trainer.model, datamodule=data_module)
    logging.info("Testing completed.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    load_dotenv()
    seed_everything(0)
    train()
