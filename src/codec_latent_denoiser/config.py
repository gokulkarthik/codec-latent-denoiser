from transformers import PretrainedConfig, DacConfig
from typing import Literal


class CodecLatentDenoiserConfig(PretrainedConfig):
    """Configuration class for CodecLatentDenoiser."""
    model_type = "codec_latent_denoiser"

    def __init__(
        self,
        codec_config: DacConfig = DacConfig(),
        denoiser_type: Literal["mlp", "llama"] = "mlp",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.codec_config = codec_config
        self.denoiser_type = denoiser_type
