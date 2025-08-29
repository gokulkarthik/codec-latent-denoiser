import pandas as pd
import random
import streamlit as st
import torch

from datasets import load_dataset, Dataset
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality as pesq
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility as stoi
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score as dnsmos
from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment as nisqa
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr

from codec_latent_denoiser import CodecLatentDenoiser, CodecLatentDenoiserProcessor

# Page config
st.set_page_config(page_title="Codec Latent Denoiser Demo", layout="wide")
st.title("💬 Codec Latent Denoiser Demo")


@st.cache_resource
def load_model_and_dataset(
    model_path: str, data_path: str
) -> tuple[CodecLatentDenoiserProcessor, CodecLatentDenoiser, Dataset]:
    """Load the model and dataset once and cache them."""
    with st.spinner("Loading model and dataset..."):
        processor = CodecLatentDenoiserProcessor.from_pretrained(model_path)
        model = CodecLatentDenoiser.from_pretrained(model_path).eval()
        ds = load_dataset(data_path, num_proc=32)["train"]

    return processor, model, ds


def compute_score(
    preds: torch.Tensor, target: torch.Tensor, sampling_rate: int = 16000
) -> dict:
    """Compute audio quality metrics."""
    result = {}

    score = si_snr(preds=preds, target=target).item()
    result["si_snr"] = round(score, 2)

    score = si_sdr(preds=preds, target=target).item()
    result["si_sdr"] = round(score, 2)

    score = stoi(preds=preds, target=target, fs=sampling_rate).item()
    result["stoi"] = round(score, 2)

    if sampling_rate in [16000, 8000]:
        mode = "wb" if sampling_rate == 16000 else "nb"
        score = pesq(preds=preds, target=target, fs=sampling_rate, mode=mode).item()
        result["pesq"] = round(score, 1)

    score = dnsmos(preds=preds, fs=sampling_rate, personalized=False)[-1].item()
    result["dnsmos"] = round(score, 1)

    score = nisqa(preds=preds, fs=sampling_rate)[0].item()
    result["nisqa"] = round(score, 1)

    return result


def main():
    """Main demo application."""
    # Model and dataset paths
    model_path = "gokulkarthik/codec-latent-denoiser-default"
    data_path = "JacobLinCool/VoiceBank-DEMAND-16k"

    # Load model and dataset
    processor, model, ds = load_model_and_dataset(model_path, data_path)
    sampling_rate = 16000

    st.success(f"✅ Model ({model_path}) and dataset ({data_path}) loaded successfully!")

    # Sample selection
    st.header("📊 Sample Selection")
    max_samples = len(ds)
    sample_idx = st.slider(
        "Select sample index", 0, max_samples - 1, value=random.randint(0, max_samples - 1)
    )

    if st.button("🎲 Random Sample"):
        sample_idx = random.randint(0, max_samples - 1)
        st.rerun()

    # Load sample
    sample = ds[sample_idx]
    clean = torch.from_numpy(sample['clean']['array'])
    noisy = torch.from_numpy(sample['noisy']['array'])

    # Display original audio
    st.header("🔊 Original Audio From Dataset")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Clean Audio")
        st.audio(clean.numpy(), sample_rate=sampling_rate)

        # Compute metrics for clean vs clean (baseline)
        clean_metrics = compute_score(preds=clean, target=clean)
        st.write("**Metrics (Clean vs Clean)**")
        for metric, value in clean_metrics.items():
            st.write(f"- {metric.upper()}: {value}")

    with col2:
        st.subheader("Noisy Audio")
        st.audio(noisy.numpy(), sample_rate=sampling_rate)

        # Compute metrics for noisy vs clean
        noisy_metrics = compute_score(preds=noisy, target=clean)
        st.write("**Metrics (Noisy vs Clean)**")
        for metric, value in noisy_metrics.items():
            st.write(f"- {metric.upper()}: {value}")

    # Run model
    if st.button("🚀 Run Model"):
        with st.spinner("Processing audio with Codec Latent Denoiser..."):
            with torch.inference_mode():
                inputs = processor(noisy)["input_values"]
                outputs = model(inputs, denoise=False, decode=True)
                outputs_denoised = model(inputs, denoise=True, decode=True)

                # Prepare generated audio
                noisy_generated = torch.zeros_like(noisy)
                noisy_denoised_generated = torch.zeros_like(noisy)
                T_min = min(outputs_denoised.audio_generated.shape[-1], noisy.shape[-1])
                noisy_generated[:T_min] = outputs.audio_generated[0][0][:T_min]
                noisy_denoised_generated[:T_min] = outputs_denoised.audio_generated[0][0][:T_min]

        # Display generated audio
        st.header("🔊 Generated Audio From Model")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Generated Without Denoiser")
            st.audio(noisy_generated.numpy(), sample_rate=sampling_rate)

            # Compute metrics for noisy generated vs clean
            noisy_gen_metrics = compute_score(preds=noisy_generated, target=clean)
            st.write("**Metrics (Generated Without Denoiser vs Clean):**")
            for metric, value in noisy_gen_metrics.items():
                st.write(f"- {metric.upper()}: {value}")

        with col2:
            st.subheader("Generated With Denoiser")
            st.audio(noisy_denoised_generated.numpy(), sample_rate=sampling_rate)

            # Compute metrics for denoised generated vs clean
            denoised_metrics = compute_score(preds=noisy_denoised_generated, target=clean)
            st.write("**Metrics (Generated With Denoiser vs Clean):**")
            for metric, value in denoised_metrics.items():
                st.write(f"- {metric.upper()}: {value}")

        # Comparison table
        st.header("📊 Metrics Comparison")
        comparison_data = {
            "Metric": ["SI-SNR", "SI-SDR", "STOI", "PESQ", "DNSMOS", "NISQA"],
            "Noisy": [
                noisy_metrics["si_snr"], noisy_metrics["si_sdr"], noisy_metrics["stoi"],
                noisy_metrics["pesq"], noisy_metrics["dnsmos"], noisy_metrics["nisqa"]
            ],
            "Generated Without Denoiser": [
                noisy_gen_metrics["si_snr"], noisy_gen_metrics["si_sdr"], noisy_gen_metrics["stoi"],
                noisy_gen_metrics["pesq"], noisy_gen_metrics["dnsmos"], noisy_gen_metrics["nisqa"]
            ],
            "Generated With Denoiser": [
                denoised_metrics["si_snr"], denoised_metrics["si_sdr"], denoised_metrics["stoi"],
                denoised_metrics["pesq"], denoised_metrics["dnsmos"], denoised_metrics["nisqa"]
            ]
        }
        comparison_df = pd.DataFrame(comparison_data).set_index("Metric")
        comparison_df['Denoiser Improvement %'] = (
            comparison_df['Generated With Denoiser'] - comparison_df['Generated Without Denoiser']
        ) * 100 / comparison_df['Generated Without Denoiser']
        st.dataframe(comparison_df)


if __name__ == "__main__":
    main()
