from transformers import DacFeatureExtractor


class CodecLatentDenoiserProcessor(DacFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, batch: list[dict], padding: str = "longest", return_tensors: str = "pt", sampling_rate: int = 16000) -> dict:
        return super().__call__(batch, padding=padding, return_tensors=return_tensors, sampling_rate=sampling_rate)
