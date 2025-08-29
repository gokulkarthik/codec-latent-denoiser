# Notes on Design and Tradeoffs

## Core Idea

The paper’s central idea is **denoising in the higher-level embedding space** rather than directly transforming raw audio or spectrograms. This avoids re-learning low-level acoustics and leverages frozen generative encoders + vocoders.

While the paper experiments with different discriminative and generative audio encoders (WavLM, Whisper, Dasheng), my choice was to test the approach with **neural audio codec (NAC) embeddings**, which naturally combine feature extraction and high-fidelity reconstruction.

---

## Model Choices

* **Encoder/Codec**:

  * Surveyed WavLM and Whisper but found them less intuitive for generative reconstruction (focus on semantics rather than acoustic detail).
  * Considered newer NACs (WavTokenizer, MIMI from Moshi) but they lack stable Hugging Face integration or direct access and control of intermediate representations.
  * Selected **Descript DAC (16 kHz)** from Hugging Face Transformers because it provides straightforward access to both continuous and quantized representations, and integrates smoothly with HF tools.

* **Denoiser Architecture**:

  * Started with a **3-layer MLP**, which reduced noise but introduced artifacts (likely due to lack of temporal modeling).
  * Switched to a **2-layer Transformer (LLaMA-style)**. This maintained efficiency (\~30M params) but leveraged sequence modeling, yielding cleaner outputs without added artifacts.
  * Conclusion: shallow Transformers are a better fit than plain MLPs for sequential latent features.

---

## Data

* Used **`JacobLinCool/VoiceBank-DEMAND-16k`** from Hugging Face Hub: \~12k training / ~800 test pairs of noisy/clean speech.
* Chose this dataset for its **ease of loading and pairing**.
* If not available, fallback would have been **creating noisy data via low-bitrate codec re-encoding** (as suggested in the take-home brief).

---

## Evaluation Metrics

* **Intrusive**:

  * PESQ (0.5–4.5): perceptual speech quality.
  * STOI (0–1): intelligibility.
* **Non-Intrusive**:

  * DNSMOS (1–5): overall quality.
  * NISQAv2 (1–5): overall quality.
* **Classical**:

  * SI-SNR and SI-SDR (dB): scale-invariant checks of distortion removal.
* This mix allowed me to validate both **perceptual quality** and **signal-level improvements**, consistent with practices in the paper.

---

## Implementation Notes

* Built the model by subclassing Hugging Face’s pretrained model API, making it easy to swap codecs or denoisers and utlize easy intergration with HF ecosystem.
* Training implemented in **PyTorch Lightning** for flexibility in experiment management.
* Training setup using Lightning.ai free credits:

  * Initial runs on **1 × L4**.
  * Final run on **4 × H100**, 10 epochs, \~10 minutes.
* Metrics improved steadily during validation, especially with the Transformer denoiser.
* Made a **Streamlit demo** for interactive testing, chosen for speed of prototyping (familiarity + easy deployment).

---