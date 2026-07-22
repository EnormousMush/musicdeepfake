"""
SSL / codec encoder frontend (Stage 1, server/GPU).

Wraps music/speech self-supervised models + a neural codec behind ONE interface and
returns *hierarchical* pooled features: every layer, temporally pooled to {mean, std}.
Shape per clip: [n_layers, 2 * hidden_dim].

Layer-wise probing (in run_stage1.py) then picks the best layer per encoder, per the
testing plan — mid layers often beat the last for deepfake detection.

Three model "kinds", each with its own code path but the same output contract:
  - hf_ssl : HuggingFace AutoModel + Wav2Vec2FeatureExtractor, output_hidden_states
             (MERT / wav2vec2 / XLS-R). n_layers = transformer layers (+embedding).
  - muq    : the `muq` package (Tencent MuQ music SSL). Raw-wav tensor in,
             output_hidden_states out. n_layers = transformer layers.
  - encodec: HuggingFace EncodecModel — a neural CODEC, used as a codec-artifact probe.
             Its conv encoder has no transformer layers, so it yields a SINGLE "layer"
             (the continuous pre-quantization latent). n_layers = 1.

Deps are imported lazily inside __init__ so a MERT-only run needs neither `muq` nor a
codec install. Interface mirrors encoders/mel.py: encode_all_layers(wav, sr) -> np.ndarray
"""
import numpy as np
import torch
import librosa

# name -> dict(kind, hf_id, sr, trc[=trust_remote_code])
MODELS = {
    "mert":     dict(kind="hf_ssl",  hf_id="m-a-p/MERT-v1-95M",            sr=24000, trc=True),   # music SSL
    "wav2vec2": dict(kind="hf_ssl",  hf_id="facebook/wav2vec2-base",       sr=16000, trc=False),  # speech/general SSL
    "xlsr":     dict(kind="hf_ssl",  hf_id="facebook/wav2vec2-xls-r-300m", sr=16000, trc=False),  # multilingual speech SSL
    "muq":      dict(kind="muq",     hf_id="OpenMuQ/MuQ-large-msd-iter",   sr=24000, trc=False),  # music SSL (Tencent MuQ, ~300M)
    "encodec":  dict(kind="encodec", hf_id="facebook/encodec_24khz",       sr=24000, trc=False),  # neural codec probe
}


class SSLEncoder:
    def __init__(self, name: str, device: str = None):
        if name not in MODELS:
            raise ValueError(f"unknown encoder '{name}'; choose from {list(MODELS)}")
        m = MODELS[name]
        self.name = name
        self.kind = m["kind"]
        self.sr = m["sr"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.kind == "hf_ssl":
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(m["hf_id"], trust_remote_code=m["trc"])
            self.model = AutoModel.from_pretrained(m["hf_id"], trust_remote_code=m["trc"]).to(self.device).eval()
        elif self.kind == "muq":
            from muq import MuQ                                  # pip install muq
            self.model = MuQ.from_pretrained(m["hf_id"]).to(self.device).eval()
        elif self.kind == "encodec":
            from transformers import EncodecModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(m["hf_id"])
            self.model = EncodecModel.from_pretrained(m["hf_id"]).to(self.device).eval()
        else:
            raise ValueError(f"unhandled kind '{self.kind}'")

    @torch.no_grad()
    def _hidden(self, wav: np.ndarray, in_sr: int) -> "torch.Tensor":
        """Per-layer hidden states [n_layers, T, D] on device (no pooling)."""
        if in_sr != self.sr:
            wav = librosa.resample(wav, orig_sr=in_sr, target_sr=self.sr)
        wav = np.asarray(wav, dtype=np.float32)

        if self.kind == "hf_ssl":
            inputs = self.processor(wav, sampling_rate=self.sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs, output_hidden_states=True)
            return torch.stack(out.hidden_states, dim=0).squeeze(1)        # [L, T, D]
        elif self.kind == "muq":
            x = torch.from_numpy(wav).unsqueeze(0).to(self.device)         # [1, T]
            out = self.model(x, output_hidden_states=True)
            return torch.stack(out.hidden_states, dim=0).squeeze(1)        # [L, T, D]
        elif self.kind == "encodec":
            inputs = self.processor(raw_audio=wav, sampling_rate=self.sr, return_tensors="pt")
            iv = inputs["input_values"].to(self.device)                    # [1, 1, T]
            lat = self.model.encoder(iv)                                   # [1, D, T'] continuous latent
            return lat.permute(0, 2, 1)                                    # [1, T', D]  (n_layers = 1)
        raise ValueError(f"unhandled kind '{self.kind}'")

    @torch.no_grad()
    def encode_all_layers(self, wav: np.ndarray, in_sr: int) -> np.ndarray:
        """wav -> [n_layers, 2 * hidden_dim] (mean||std pooled per layer). For the linear probe."""
        hs = self._hidden(wav, in_sr)
        mean = hs.mean(dim=1)                                              # [L, D]
        std = hs.std(dim=1)                                                # [L, D]
        return torch.cat([mean, std], dim=-1).cpu().numpy()               # [L, 2D]

    @torch.no_grad()
    def encode_frames(self, wav: np.ndarray, in_sr: int, layer: int = -1) -> np.ndarray:
        """wav -> ONE layer's frame-level features [T, D] (time axis kept) for temporal
        classifiers (AASIST / SpecTTTra). Caching every layer's frames is huge, so pick one
        layer (e.g. the per-layer probe's best). layer=-1 = last."""
        return self._hidden(wav, in_sr)[layer].cpu().numpy()               # [T, D]
