import os
import glob
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch

from stable_audio_tools.data.dataset import create_dataloader_from_config

# try to reuse load_model from pre_encode if available (keeps parity with existing loader)
try:
    from pre_encode import load_model
except Exception:
    # fallback: minimal loader using stable_audio_tools helpers
    from stable_audio_tools.models.pretrained import get_pretrained_model
    from stable_audio_tools.models.factory import create_model_from_config

    def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, model_half=False):
        if pretrained_name is not None:
            print(f"Loading pretrained model {pretrained_name}")
            model, model_config = get_pretrained_model(pretrained_name)
        elif model_config is not None and model_ckpt_path is not None:
            print(f"Creating model from config")
            model = create_model_from_config(model_config)
            print(f"Loading model checkpoint from {model_ckpt_path}")
            try:
                ck = torch.load(model_ckpt_path, map_location='cpu')
                if 'state_dict' in ck:
                    sd = ck['state_dict']
                else:
                    sd = ck
                model.load_state_dict(sd, strict=False)
            except Exception as e:
                print(f"Warning: failed to load checkpoint cleanly: {e}")

        model.eval().requires_grad_(False)
        if model_half:
            model.to(torch.float16)
        print("Done loading model")
        return model, model_config


def list_latent_files(latents_dir):
    exts = ("*.npy", "*.pt")
    files = []
    for e in exts:
        files.extend(sorted(glob.glob(os.path.join(latents_dir, "**", e), recursive=True)))
    return files


def load_latent(path):
    if path.endswith('.npy'):
        arr = np.load(path, allow_pickle=True)
        return arr
    else:
        obj = torch.load(path, map_location='cpu')
        if isinstance(obj, dict) and 'latents' in obj:
            return obj['latents']
        return obj


def ensure_latent_tensor(x):
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)

    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim == 3:
        return t
    return t.unsqueeze(0)


def rms_energy(audio, sr=44100):
    """Compute A-weighted RMS energy with STFT parameters matched to 2048x VAE downsampling.

    Uses n_fft=2048, hop_length=2048, center=False so that the number of
    output frames equals exactly T_audio / 2048, matching the VAE latent rate.

    Args:
        audio: (B, C, T) or (C, T) tensor
        sr: sample rate (default 44100)

    Returns:
        loudness_db: (B, frames) tensor of A-weighted RMS in dB
    """
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    B, C, T = audio.shape

    # Mixdown to mono
    audio_mono = audio.mean(dim=1)  # (B, T)

    # STFT parameters matched to 2048x VAE downsampling
    # center=False with n_fft=hop_length=2048 gives exactly T/2048 frames
    n_fft = 2048
    hop_length = 2048
    window = torch.hann_window(n_fft, device=audio.device)

    spec = torch.stft(
        audio_mono, n_fft=n_fft, hop_length=hop_length, window=window,
        return_complex=True, center=False
    )  # (B, F, frames)
    mag = spec.abs()  # (B, F, frames)

    # A-weighting curve
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr).to(audio.device)  # (F,)

    def a_weighting(f):
        f_sq = f ** 2
        ra = (f_sq + 20.6 ** 2) * (f_sq + 12200 ** 2)
        num = (12200 ** 2) * (f_sq) ** 2
        den = ra * torch.sqrt((f_sq + 107.7 ** 2) * (f_sq + 737.9 ** 2))
        a = num / (den + 1e-20)
        a_db = 2.0 + 20 * torch.log10(a + 1e-20)
        return a_db

    a_weight = a_weighting(freqs)  # (F,)
    a_weight_lin = 10 ** (a_weight / 20)  # dB to linear

    # Apply A-weighting
    mag_weighted = mag * a_weight_lin[None, :, None]  # (B, F, frames)

    # RMS per frame
    loudness = torch.sqrt((mag_weighted ** 2).sum(dim=1) / mag_weighted.shape[1])  # (B, frames)
    loudness_db = 20 * torch.log10(loudness + 1e-20)  # (B, frames)
    return loudness_db


def main():
    parser = argparse.ArgumentParser(description='Decode VAE latents and extract RMS energy')
    parser.add_argument('--latents-dir', type=str, help='Directory containing latent files (.npy or .pt)', required=False)
    parser.add_argument('--output-dir', type=str, help='Output directory for RMS .npy files', required=True)
    parser.add_argument('--model-config', type=str, help='Path to model config (for VAE)', default=None)
    parser.add_argument('--ckpt-path', type=str, help='Path to VAE checkpoint', default=None)
    parser.add_argument('--pretrained-name', type=str, help='Pretrained model name (optional)', default=None)
    parser.add_argument('--model-half', action='store_true', help='Use half precision for VAE')
    parser.add_argument('--dataset-config', type=str, help='Optional dataset config (pre_encoded) to use create_dataloader_from_config', default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use-data-parallel-vae', action='store_true', help='Wrap VAE in DataParallel when multiple GPUs are available.')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--sample-size', type=int, default=1320960, help='sample size passed to dataloader when using dataset-config')
    parser.add_argument('--num-workers', type=int, default=4, help='dataloader num workers when using dataset-config')
    parser.add_argument('--shuffle', action='store_true', help='shuffle dataloader when using dataset-config')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load VAE
    with open(args.model_config, 'r') as f:
        model_config = json.load(f)

    vae_model, model_config = load_model(
        model_config=model_config, model_ckpt_path=args.ckpt_path,
        pretrained_name=args.pretrained_name, model_half=args.model_half
    )
    vae_model = vae_model.pretransform

    # Device and multi-GPU
    use_cuda = (args.device.startswith('cuda') and torch.cuda.is_available())
    ngpu = torch.cuda.device_count() if use_cuda else 0
    device = torch.device(args.device if use_cuda else 'cpu')

    vae_is_dataparallel = False
    if ngpu > 1 and args.use_data_parallel_vae:
        print(f"Wrapping VAE with DataParallel across {ngpu} GPUs")
        try:
            class VAEDecodeWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, latents, **kwargs):
                    return self.model.decode(latents, **kwargs)
                def decode(self, latents, **kwargs):
                    return self.forward(latents, **kwargs)

            vae_model = torch.nn.DataParallel(VAEDecodeWrapper(vae_model))
            vae_is_dataparallel = True
        except Exception as e:
            print(f"Failed to wrap VAE in DataParallel: {e}")

    if use_cuda:
        vae_model.to(device)

    vae_sample_rate = model_config.get('sample_rate', 44100) if model_config is not None else 44100

    # Set up data source
    data_loader = None
    files = []
    if args.dataset_config is not None:
        with open(args.dataset_config, 'r') as f:
            dataset_config = json.load(f)

        if dataset_config.get('dataset_type', None) != 'pre_encoded':
            print('Warning: dataset_config.dataset_type is not "pre_encoded"; attempting to use it anyway')

        data_loader = create_dataloader_from_config(
            dataset_config,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            sample_rate=(model_config.get('sample_rate', 44100) if model_config is not None else 44100),
            audio_channels=model_config.get('audio_channels', 2) if model_config is not None else 2,
            num_workers=args.num_workers,
            shuffle=args.shuffle
        )
        print('Created dataloader from dataset-config')
    else:
        if args.latents_dir is None:
            raise ValueError('Please provide --latents-dir with latent files or --dataset-config')
        files = list_latent_files(args.latents_dir)
        print(f"Found {len(files)} latent files")

    # Build iterator
    if data_loader is not None:
        iterator = data_loader
    else:
        def file_batch_iterator():
            for i in range(0, len(files), args.batch_size):
                yield files[i:i + args.batch_size]
        iterator = file_batch_iterator()

    count = 0
    for batch in tqdm(iterator):
        if data_loader is not None:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                latents_batch = batch[0]
                metadata = batch[1]
            else:
                latents_batch, metadata = batch

            latents_batch = latents_batch.to(device=device)
            if args.model_half:
                latents_batch = latents_batch.to(torch.float16)

            t_latent = latents_batch.shape[-1]

            with torch.no_grad():
                if vae_is_dataparallel:
                    decoded = vae_model(latents_batch)
                else:
                    try:
                        decoded = vae_model.decode(latents_batch)
                    except Exception:
                        if hasattr(vae_model, 'pretransform') and hasattr(vae_model.pretransform, 'decode'):
                            decoded = vae_model.pretransform.decode(latents_batch)
                        else:
                            raise

            # Compute RMS on decoded audio (still on GPU)
            decoded_f32 = decoded.float()
            rms = rms_energy(decoded_f32, sr=vae_sample_rate)  # (B, frames)

            # assert same length
            assert rms.shape[-1] == t_latent, f"RMS length {rms.shape[-1]} does not match latent length {t_latent}"

            # Trim/pad to match latent length exactly
            if rms.shape[-1] > t_latent:
                rms = rms[..., :t_latent]
            elif rms.shape[-1] < t_latent:
                pad = t_latent - rms.shape[-1]
                rms = torch.nn.functional.pad(rms, (0, pad), value=rms[..., -1:].item() if rms.shape[-1] > 0 else 0.0)

            rms_np = rms.cpu().numpy()

            for j in range(rms_np.shape[0]):
                info = metadata[j] if j < len(metadata) else {}
                latent_path = info.get('latent_filename', '')
                out_name = os.path.splitext(os.path.basename(latent_path))[0] + '_rms.npy'
                print(f"Saving RMS for {latent_path} to {os.path.join(args.output_dir, out_name)}")
                np.save(os.path.join(args.output_dir, out_name), rms_np[j])
                count += 1

        else:
            assert False
            # batch is list of file paths
            batch_files = batch
            for p in batch_files:
                latent_obj = load_latent(p)
                latent_t = ensure_latent_tensor(latent_obj)
                t_latent = latent_t.shape[-1]
                latent_t = latent_t.to(device=device)
                if args.model_half:
                    latent_t = latent_t.to(torch.float16)

                with torch.no_grad():
                    if vae_is_dataparallel:
                        decoded = vae_model(latent_t)
                    else:
                        try:
                            decoded = vae_model.decode(latent_t)
                        except Exception:
                            if hasattr(vae_model, 'pretransform') and hasattr(vae_model.pretransform, 'decode'):
                                decoded = vae_model.pretransform.decode(latent_t)
                            else:
                                raise

                decoded_f32 = decoded.float()
                rms = rms_energy(decoded_f32, sr=vae_sample_rate)  # (B, frames)

                # Trim/pad to match latent length exactly
                if rms.shape[-1] > t_latent:
                    rms = rms[..., :t_latent]
                elif rms.shape[-1] < t_latent:
                    pad = t_latent - rms.shape[-1]
                    rms = torch.nn.functional.pad(rms, (0, pad), value=rms[..., -1:].item() if rms.shape[-1] > 0 else 0.0)

                rms_np = rms.cpu().numpy()

                for j in range(rms_np.shape[0]):
                    out_name = os.path.splitext(os.path.basename(p))[0] + '_rms.npy'
                    np.save(os.path.join(args.output_dir, out_name), rms_np[j])
                    count += 1

    print(f"Saved {count} RMS files to {args.output_dir}")


if __name__ == '__main__':
    main()
