import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import PreTrainedModel, DacModel, DacConfig
from transformers.utils import ModelOutput
from typing import Optional

from codec_latent_denoiser.config import CodecLatentDenoiserConfig


class MLPDenoiser(nn.Module):
    """MLP-based denoiser for codec latent space."""
    
    def __init__(self, config: CodecLatentDenoiserConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.codec_config.hidden_size
        self.layer1 = nn.Linear(self.hidden_size, self.hidden_size * 2, bias=False)
        self.layer2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(self.hidden_size, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP denoiser."""
        x_before = x
        x = self.layer_norm(x)
        x = self.dropout(self.activation(self.layer1(x)))
        x = self.layer2(x) + x_before
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
