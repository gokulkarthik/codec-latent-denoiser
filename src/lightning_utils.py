import lightning as L
import torch
import torch.nn as nn

from datasets import load_dataset, Audio
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Dataset
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    DeepNoiseSuppressionMeanOpinionScore,
    NonIntrusiveSpeechQualityAssessment,
)
from transformers import get_scheduler, DacConfig, DacModel
from typing import Literal

from codec_latent_denoiser import (
    CodecLatentDenoiser,
    CodecLatentDenoiserConfig,
    CodecLatentDenoiserOutput,
    CodecLatentDenoiserProcessor,
)


class CodecLatentDenoiserLightningModule(L.LightningModule):
    """Lightning module for training the Codec Latent Denoiser."""

    def __init__(
        self,
        pretrained_codec_path: str,
        denoiser_type: Literal["mlp", "llama"] = "mlp",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        train_only_denoiser: bool = True,
    ) -> None:
        super().__init__()
        self.pretrained_codec_path = pretrained_codec_path
        self.denoiser_type = denoiser_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_only_denoiser = train_only_denoiser

        codec_config = DacConfig.from_pretrained(pretrained_codec_path)
        sampling_rate = codec_config.sampling_rate
        self.model_config = CodecLatentDenoiserConfig(
            codec_config=codec_config, denoiser_type=denoiser_type
        )
        self.model = CodecLatentDenoiser(self.model_config)
        print(f"Loading codec from {pretrained_codec_path}...")
        self.model.codec = DacModel.from_pretrained(pretrained_codec_path)
        print(f"Codec loaded successfully from {pretrained_codec_path}.")

        self.loss_fn = nn.MSELoss()
        pesq_mode = "wb" if sampling_rate == 16000 else "nb"
        self.pesq = PerceptualEvaluationSpeechQuality(sampling_rate, pesq_mode)
        self.dnsmos = DeepNoiseSuppressionMeanOpinionScore(sampling_rate, False)
        self.nisqa = NonIntrusiveSpeechQualityAssessment(sampling_rate)

        if self.train_only_denoiser:
            for name, param in self.model.named_parameters():
                if "denoiser" not in name:
                    param.requires_grad = False
            self.model.denoiser.train()
            self.model.codec.eval()
        else:
            self.model.train()

    def forward(
        self, x: torch.Tensor, denoise: bool = True, decode: bool = False
    ) -> CodecLatentDenoiserOutput:
        """Forward pass through the model."""
        return self.model(x, denoise, decode)

    def common_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Common training/validation step."""
        clean_speech = batch["input_values_clean"]
        noisy_speech = batch["input_values_noisy"]
        clean_speech_quantized = self.model(
            clean_speech, denoise=False, decode=False
        ).audio_embeddings
        noisy_speech_quantized = self.model(
            noisy_speech, denoise=True, decode=False
        ).audio_embeddings
        loss = self.loss_fn(clean_speech_quantized, noisy_speech_quantized)
        return loss

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        noisy_speech = batch["input_values_noisy"]
        noisy_speech_denoised_values = self.model(
            noisy_speech, denoise=True, decode=True
        ).audio_generated
        noisy_speech_denoised = torch.zeros_like(noisy_speech)
        T_min = min(noisy_speech_denoised.shape[-1], noisy_speech.shape[-1])
        noisy_speech_denoised[:, :, :T_min] = noisy_speech_denoised_values[:, :, :T_min]
        return noisy_speech_denoised

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        loss = self.common_step(batch, batch_idx)
        self.log("train/learning_rate", current_lr, prog_bar=True, sync_dist=True)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        loss = self.common_step(batch, batch_idx)
        preds = self.predict_step(batch, batch_idx)
        pesq_score = self.pesq(
            preds=preds,
            target=batch["input_values_clean"],
        )
        dnsmos_score = self.dnsmos(preds=preds)[-1]
        nisqa_score = self.nisqa(preds=preds)[0]
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/pesq", pesq_score, prog_bar=True, sync_dist=True)
        self.log("val/dnsmos", dnsmos_score, prog_bar=True, sync_dist=True)
        self.log("val/nisqa", nisqa_score, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss = self.common_step(batch, batch_idx)
        preds = self.predict_step(batch, batch_idx)
        pesq_score = self.pesq(
            preds=preds,
            target=batch["input_values_clean"],
        )
        dnsmos_score = self.dnsmos(preds=preds)[-1]
        nisqa_score = self.nisqa(preds=preds)[0]
        self.log("test/loss", loss, prog_bar=True, sync_dist=True)
        self.log("test/pesq", pesq_score, prog_bar=True, sync_dist=True)
        self.log("test/dnsmos", dnsmos_score, prog_bar=True, sync_dist=True)
        self.log("test/nisqa", nisqa_score, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Reset metrics at end of validation epoch."""
        self.pesq.reset()
        self.dnsmos.reset()
        self.nisqa.reset()

    def on_test_epoch_end(self) -> None:
        """Reset metrics at end of test epoch."""
        self.pesq.reset()
        self.dnsmos.reset()
        self.nisqa.reset()

    def configure_optimizers(self) -> dict:
        """Configure optimizers and schedulers."""
        if self.train_only_denoiser:
            parameters = self.model.denoiser.parameters()
        else:
            parameters = self.model.parameters()

        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(0.05 * num_training_steps)

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class CodecLatentDenoiserDataset(Dataset):
    """Dataset wrapper for codec latent denoiser training."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset[idx]
        return {
            "input_values_clean": row["clean"]["array"],
            "input_values_noisy": row["noisy"]["array"],
        }


class CodecLatentDenoiserDatasetCollator:
    """Collator for batching dataset samples."""

    def __init__(
        self, processor: CodecLatentDenoiserProcessor, padding: str = "longest"
    ) -> None:
        self.processor = processor
        self.padding = padding

    def __call__(self, batch: list[dict]) -> dict:
        output = {}
        for key in batch[0].keys():
            if key in ["input_values_clean", "input_values_noisy"]:
                output[key] = self.processor(
                    [item[key] for item in batch],
                    padding=self.padding,
                    return_tensors="pt",
                    sampling_rate=self.processor.sampling_rate,
                )["input_values"]
            else:
                output[key] = torch.stack([item[key] for item in batch])
        return output


class CodecLatentDenoiserLightningDataModule(L.LightningDataModule):
    """Lightning data module for codec latent denoiser."""

    def __init__(
        self,
        data_path_hf_hub: str,
        clean_speech_key: str = "clean",
        noisy_speech_key: str = "noisy",
        processor: CodecLatentDenoiserProcessor = None,
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_path_hf_hub = data_path_hf_hub
        self.clean_speech_key = clean_speech_key
        self.noisy_speech_key = noisy_speech_key
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collator = CodecLatentDenoiserDatasetCollator(self.processor)

    def setup(self, stage: str) -> None:
        """Setup datasets for training/validation/testing."""
        dataset = load_dataset(self.data_path_hf_hub)
        if self.clean_speech_key != "clean":
            dataset = dataset.rename_column(self.clean_speech_key, "clean")
        if self.noisy_speech_key != "noisy":
            dataset = dataset.rename_column(self.noisy_speech_key, "noisy")
        dataset = dataset.select_columns(["clean", "noisy"])
        dataset = dataset.cast_column(
            "clean", Audio(sampling_rate=self.processor.sampling_rate)
        )
        dataset = dataset.cast_column(
            "noisy", Audio(sampling_rate=self.processor.sampling_rate)
        )

        self.dataset_train = dataset["train"]
        self.dataset_test = dataset["test"]
        if "val" not in dataset:
            dataset_train_val = dataset["train"].train_test_split(test_size=100)
            self.dataset_train = dataset_train_val["train"]
            self.dataset_val = dataset_train_val["test"]
            del dataset_train_val
        else:
            self.dataset_val = dataset["val"]
        del dataset

        self.dataset_train = CodecLatentDenoiserDataset(self.dataset_train)
        self.dataset_val = CodecLatentDenoiserDataset(self.dataset_val)
        self.dataset_test = CodecLatentDenoiserDataset(self.dataset_test)

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            persistent_workers=True,
        )


class HuggingFaceHubPushCallback(Callback):
    """Callback to push model to HuggingFace Hub."""

    def __init__(
        self,
        repo_id: str,
        push_every_n_epochs: int = 1,
        processor: CodecLatentDenoiserProcessor = None,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.push_every_n_epochs = push_every_n_epochs
        self.processor = processor

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Push model to hub at end of training epoch."""
        is_final_epoch = trainer.current_epoch == trainer.max_epochs - 1
        is_interval_epoch = (trainer.current_epoch + 1) % self.push_every_n_epochs == 0

        if is_final_epoch or is_interval_epoch:
            if self.processor:
                self.processor.push_to_hub(
                    repo_id=self.repo_id,
                    private=True,
                    commit_message="Add processor",
                )
            pl_module.model.push_to_hub(
                repo_id=self.repo_id,
                private=True,
                commit_message=f"Epoch {trainer.current_epoch + 1}",
            )
