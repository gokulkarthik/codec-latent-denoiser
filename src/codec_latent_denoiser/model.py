import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import PreTrainedModel, DacModel
from transformers.utils import ModelOutput
from typing import Optional

from codec_latent_denoiser.config import CodecLatentDenoiserConfig


class MLPDenoiser(nn.Module):
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
        x_before = x
        x = self.layer_norm(x)
        x = self.dropout(self.activation(self.layer1(x)))
        x = self.layer2(x) + x_before
        return x


@dataclass
class CodecLatentDenoiserOutput(ModelOutput):
    quantized_representation: torch.Tensor = None
    audio_generated: Optional[torch.Tensor] = None


class CodecLatentDenoiser(PreTrainedModel):
    config_class = CodecLatentDenoiserConfig

    def __init__(self, config: CodecLatentDenoiserConfig) -> None:
        super().__init__(config)
        self.config = config
        self.codec = DacModel.from_pretrained(config.pretrained_codec_path)

        if config.denoiser_type == "mlp":
            self.denoiser = MLPDenoiser(config)
        else:
            raise ValueError(f"Invalid denoiser type: {config.denoiser_type}")

    def forward(
        self,
        x: torch.Tensor,
        denoise: bool = True,
        decode: bool = False,
    ) -> torch.Tensor:
        audio_embeddings = self.codec.encode(x).quantized_representation  # [B, D, T]
        if denoise:
            audio_embeddings = audio_embeddings.transpose(1, 2)  # [B, T, D]
            audio_embeddings = self.denoiser(audio_embeddings)  # [B, T, D]
            quantized_representation = audio_embeddings.transpose(1, 2)  # [B, D, T]
        else:
            quantized_representation = audio_embeddings  # [B, D, T]

        output = CodecLatentDenoiserOutput(
            quantized_representation=quantized_representation
        )

        if decode:
            audio_generated = self.codec.decode(
                quantized_representation=quantized_representation
            ).audio_values
            output.audio_generated = audio_generated

        return output
