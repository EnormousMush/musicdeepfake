"""
Server self-check — run BEFORE transferring any data.

  python check_server.py                 # torch + CUDA visibility
  python check_server.py --encoder mert  # also download+load the model, run 1s of noise

Confirms the GPU + torch + the SSL model all work with zero audio transferred.
If this passes, the environment is good; only then move the dataset over.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=None,
                    choices=[None, "mert", "wav2vec2", "xlsr", "muq", "encodec"])
    args = ap.parse_args()

    import torch
    print(f"torch {torch.__version__}")
    ok = torch.cuda.is_available()
    print(f"cuda available: {ok}")
    if ok:
        print(f"device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no CUDA — torch install does not match the server GPU/driver.")

    if args.encoder:
        import numpy as np
        sys.path.insert(0, ".")
        from encoders.ssl import SSLEncoder
        print(f"\nLoading encoder '{args.encoder}' (first run downloads weights)...")
        enc = SSLEncoder(args.encoder)
        wav = np.random.randn(enc.sr).astype("float32")   # 1 second of noise
        feat = enc.encode_all_layers(wav, enc.sr)
        print(f"OK: {args.encoder} on {enc.device}, feature shape {feat.shape} "
              f"= (n_layers, 2*hidden_dim)")
        print("\nEnvironment is good. Safe to transfer the dataset.")


if __name__ == "__main__":
    main()
