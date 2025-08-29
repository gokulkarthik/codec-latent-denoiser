import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import PreTrainedModel, DacModel, DacConfig, LlamaConfig, LlamaModel
from transformers.utils import ModelOutput

from typing import Optional

from codec_latent_denoiser.config import CodecLatentDenoiserConfig


class MLPDenoiser(nn.Module):
    """MLP-based denoiser for codec latent space."""

    def __init__(self, config: CodecLatentDenoiserConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.codec_config.hidden_size
        self.layer1 = nn.Linear(self.hidden_size, self.hidden_size * 4, bias=True)
        self.layer2 = nn.Linear(self.hidden_size * 4, self.hidden_size * 4, bias=True)
        self.layer3 = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=True)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP denoiser."""
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x


class LLaMADenoiser(nn.Module):
    """LLaMA architecture based denoiser for codec latent space."""

    def __init__(self, config: CodecLatentDenoiserConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.codec_config.hidden_size
        denoiser_config = LlamaConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=self.hidden_size * 4,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
            vocab_size=32,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            use_cache=False,
        )
        self.backbone = LlamaModel(denoiser_config)
        self.out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LLaMA denoiser."""
        x_original = x
        x = self.backbone(inputs_embeds=x).last_hidden_state
        x = self.out(x)
        x = x + x_original
        return x


@dataclass
class CodecLatentDenoiserOutput(ModelOutput):
    """Output class for CodecLatentDenoiser."""

    audio_embeddings: torch.Tensor = None
    audio_generated: Optional[torch.Tensor] = None


class CodecLatentDenoiser(PreTrainedModel):
    """Main model for codec latent denoising."""

    config_class = CodecLatentDenoiserConfig

    def __init__(self, config: CodecLatentDenoiserConfig) -> None:
        if not isinstance(config.codec_config, DacConfig):
            config.codec_config = DacConfig(**config.codec_config)
        super().__init__(config)
        self.config = config
        self.codec = DacModel(self.config.codec_config)
        if self.config.denoiser_type == "mlp":
            self.denoiser = MLPDenoiser(self.config)
        elif self.config.denoiser_type == "llama":
            self.denoiser = LLaMADenoiser(self.config)
        else:
            raise ValueError(f"Invalid denoiser type: {self.config.denoiser_type}")

    def forward(
        self,
        x: torch.Tensor,
        denoise: bool = True,
        decode: bool = False,
    ) -> CodecLatentDenoiserOutput:
        """Forward pass through the model."""
        audio_embeddings = self.codec.encode(
            x
        ).quantized_representation  # [B, C, T] -> [B, D, T]
        if denoise:
            audio_embeddings = audio_embeddings.transpose(1, 2)  # [B, T, D]
            audio_embeddings = self.denoiser(audio_embeddings)  # [B, T, D]
            audio_embeddings = audio_embeddings.transpose(1, 2)  # [B, D, T]

        output = CodecLatentDenoiserOutput(audio_embeddings=audio_embeddings)

        if decode:
            output.audio_generated = torch.zeros_like(x)  # [B, C, T]
            audio_generated = self.codec.decode(
                quantized_representation=audio_embeddings
            ).audio_values.unsqueeze(1)  # [B, T_out] -> [B, 1, T_out]
            T_min = min(x.shape[2], audio_generated.shape[2])
            output.audio_generated[:, :, :T_min] = audio_generated[:, :, :T_min]

        return output
