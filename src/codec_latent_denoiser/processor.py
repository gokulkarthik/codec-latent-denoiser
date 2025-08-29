from transformers import DacFeatureExtractor
from typing import List, Dict, Any


class CodecLatentDenoiserProcessor(DacFeatureExtractor):
    """Processor for CodecLatentDenoiser model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        batch: List[Dict[str, Any]],
        padding: str = "longest",
        return_tensors: str = "pt",
        sampling_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Process audio batch."""
        return super().__call__(
            batch,
            padding=padding,
            return_tensors=return_tensors,
            sampling_rate=sampling_rate,
        )
