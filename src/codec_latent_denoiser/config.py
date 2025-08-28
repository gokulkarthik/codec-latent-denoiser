from transformers import PretrainedConfig, DacConfig
from typing import Literal


class CodecLatentDenoiserConfig(PretrainedConfig):
    model_type = "codec_latent_denoiser"

    def __init__(
        self,
        pretrained_codec_path: str = "descript/dac_16khz",
        denoiser_type: Literal["mlp"] = "mlp",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pretrained_codec_path = pretrained_codec_path
        self.denoiser_type = denoiser_type
        self.codec_config = DacConfig.from_pretrained(pretrained_codec_path)
