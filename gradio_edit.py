"""Standalone Gradio interface for FlowEdit (`generate_diffusion_latent_flowedit`).

Mirrors the spirit of `run_gradio.py` but builds an edit-only UI exposing:
  - source prompt, target prompt, init audio
  - the four CFG scales (src/tar x inv/lfe)
  - inv_steps, lfe_steps, n_avg, noise_amt, seed

Each Generate click runs three sequential variants with identical params/seed:
  1. Inverse only       (lfe_steps=0, stochastic inverse)
  2. FlowEdit only      (deterministic_inverse=True, then LFE)
  3. Full pipeline      (stochastic inverse + LFE)

Intermediate samples are not wired in yet (Stage B).
"""

import argparse
import gc
import json

import gradio as gr
import numpy as np
import torch
import torchaudio
from einops import rearrange
from torchaudio import transforms as T

from stable_audio_tools.inference.generation import generate_diffusion_latent_flowedit
from stable_audio_tools.interface.aeiou import audio_spectrogram_image
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict


model = None
sample_rate = 44100
sample_size = 2097152
model_half = True

# Number of columns pre-allocated per LFE row in the UI.
# Layout: 1 main + (NUM_COLS - 1) intermediates per row.
# Must match the loop bound in create_edit_ui so the function's return arity
# always matches generate_button.click's outputs= list.
NUM_COLS = 5


def _pad(lst, n, fill=None):
    """Return a copy of lst extended to length n with `fill` (None by default)."""
    lst = list(lst)
    if len(lst) >= n:
        return lst[:n]
    return lst + [fill] * (n - len(lst))
num_cols = 5


def load_model(
    model_config=None,
    model_ckpt_path=None,
    pretrained_name=None,
    pretransform_ckpt_path=None,
    device="cuda",
    in_model_half=False,
):
    global model, sample_rate, sample_size, model_half

    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)
    elif model_config is not None and model_ckpt_path is not None:
        print("Creating model from config")
        model = create_model_from_config(model_config)
        print(f"Loading model checkpoint from {model_ckpt_path}")
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(
            load_ckpt_state_dict(pretransform_ckpt_path), strict=False
        )

    model.to(device).eval().requires_grad_(False)
    if in_model_half:
        model.to(torch.float16)
    model_half = in_model_half

    print("Done loading model")
    return model, model_config


def _prepare_init_audio(init_audio_input):
    """Convert a Gradio audio tuple (sr, np.ndarray) into the (sr, [c, n] tensor)
    format that `generate_diffusion_latent_flowedit` -> `prepare_audio` expects."""
    if init_audio_input is None:
        return None

    in_sr, audio = init_audio_input

    if audio.dtype == np.float32:
        audio = torch.from_numpy(audio)
    elif audio.dtype == np.int16:
        audio = torch.from_numpy(audio).float().div(32767)
    elif audio.dtype == np.int32:
        audio = torch.from_numpy(audio).float().div(2147483647)
    else:
        raise ValueError(f"Unsupported audio data type: {audio.dtype}")

    if model_half:
        audio = audio.to(torch.float16)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1)

    if in_sr != sample_rate:
        resample_tf = T.Resample(in_sr, sample_rate).to(audio.device).to(audio.dtype)
        audio = resample_tf(audio)

    if audio.shape[-1] > sample_size:
        audio = audio[:, :sample_size]

    return (sample_rate, audio)


def _sampled_to_outputs(
    sampled,
    save_path='/data/scratch-fast/snnithya/sat-zenon/gradio_debug.wav',
    spec_figsize=(3, 2),
):
    """Convert a [batch, channels, samples] tensor to (gradio_audio_tuple, [spectrogram]).

    `spec_figsize` controls matplotlib figure size for the rendered spectrogram.
    Smaller -> smaller PNG bytes -> faster Gradio Gallery rendering.
    """
    audio = rearrange(sampled, "b d n -> d (b n)").to(torch.float32).cpu()
    peak = audio.abs().max().clamp(min=1e-8)
    audio_int16 = audio.div(peak).clamp(-1, 1).mul(32767).to(torch.int16)

    if save_path is not None:
        torchaudio.save(save_path, audio_int16, sample_rate)

    spectrogram = audio_spectrogram_image(
        audio_int16, sample_rate=sample_rate, figsize=spec_figsize
    )
    audio_np = audio_int16.numpy().T  # (samples, channels)
    return (sample_rate, audio_np), [spectrogram]


def _run_flowedit(
    src_conditioning,
    tar_conditioning,
    init_audio,
    device,
    seed,
    src_inv_cfg_scale,
    tar_inv_cfg_scale,
    src_lfe_cfg_scale,
    tar_lfe_cfg_scale,
    inv_steps,
    lfe_steps,
    n_avg,
    noise_amt,
    deterministic_inverse,
    return_intermediate_latents,
    intermediate_latents_interval,
):
    """One sequential call to generate_diffusion_latent_flowedit.

    Builds fresh conditioning tensors each call, since the function deletes its
    local references at the end (and to be safe re: in-place dtype casts on views).
    """
    src_conditioning_tensors = model.conditioner(src_conditioning, device)
    tar_conditioning_tensors = model.conditioner(tar_conditioning, device)

    sampled, _intermediate_latents = generate_diffusion_latent_flowedit(
        model,
        src_inv_cfg_scale=float(src_inv_cfg_scale),
        tar_inv_cfg_scale=float(tar_inv_cfg_scale),
        src_lfe_cfg_scale=float(src_lfe_cfg_scale),
        tar_lfe_cfg_scale=float(tar_lfe_cfg_scale),
        src_conditioning_tensors=src_conditioning_tensors,
        tar_conditioning_tensors=tar_conditioning_tensors,
        n_avg=int(n_avg),
        batch_size=1,
        sample_size=sample_size,
        seed=int(seed),
        device=device,
        init_audio=init_audio,
        deterministic_inverse=bool(deterministic_inverse),
        noise_amt=float(noise_amt),
        inv_steps=int(inv_steps),
        lfe_steps=int(lfe_steps),
        return_intermediate_latents=return_intermediate_latents,
        intermediate_latents_interval=int(intermediate_latents_interval),
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return sampled, _intermediate_latents


def generate_edit(
    src_prompt,
    tar_prompt,
    init_audio_input,
    src_inv_cfg_scale,
    tar_inv_cfg_scale,
    src_lfe_cfg_scale,
    tar_lfe_cfg_scale,
    inv_steps,
    lfe_steps,
    n_avg,
    noise_amt,
    seed,
    deterministic_inverse,
    intermediate_latents_interval,
):
    """Run three variants sequentially with the same params/seed and return all
    six outputs (audio, spectrogram-gallery) for: inverse-only, FlowEdit-only,
    full pipeline."""
    if init_audio_input is None:
        raise gr.Error("Please provide an init audio file.")
    if not src_prompt or not tar_prompt:
        raise gr.Error("Please provide both a source and a target prompt.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    device = next(model.parameters()).device

    seed = int(seed)
    if seed == -1:
        seed = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
    print(f"[gradio_edit] using seed={seed}")

    seconds_total = sample_size // sample_rate
    src_conditioning = [{
        "prompt": src_prompt,
        "seconds_start": 0,
        "seconds_total": seconds_total,
    }]
    tar_conditioning = [{
        "prompt": tar_prompt,
        "seconds_start": 0,
        "seconds_total": seconds_total,
    }]

    init_audio = _prepare_init_audio(init_audio_input)

    common = dict(
        src_conditioning=src_conditioning,
        tar_conditioning=tar_conditioning,
        init_audio=init_audio,
        device=device,
        seed=seed,
        src_inv_cfg_scale=src_inv_cfg_scale,
        tar_inv_cfg_scale=tar_inv_cfg_scale,
        src_lfe_cfg_scale=src_lfe_cfg_scale,
        tar_lfe_cfg_scale=tar_lfe_cfg_scale,
        n_avg=n_avg,
        intermediate_latents_interval=intermediate_latents_interval,
    )

    # 1. Inverse-only: stochastic inverse + decode (no LFE)
    print("[gradio_edit] (1/3) inverse-only run (lfe_steps=0)")
    sampled_inv, _ = _run_flowedit(
        **common,
        lfe_steps=0,
        noise_amt=1,
        deterministic_inverse=True,
        inv_steps=inv_steps,
        return_intermediate_latents=False,
    )
    inv_audio, inv_specs = _sampled_to_outputs(sampled_inv, save_path="edit_output_inverse.wav")

    del sampled_inv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. FlowEdit-only: deterministic inverse (exactly reproduces source) + LFE
    print("[gradio_edit] (2/3) flowedit-only run")
    sampled_flow, intermediate_samples_flow = _run_flowedit(
        **common,
        lfe_steps=lfe_steps,
        inv_steps=0,
        deterministic_inverse=deterministic_inverse,
        noise_amt=0,
        return_intermediate_latents=True,
    )
    flow_audio_list = []
    flow_specs_list = []
    for ind, intermediate_sample in enumerate(intermediate_samples_flow[: NUM_COLS - 1]):
        flow_audio, flow_specs = _sampled_to_outputs(intermediate_sample, save_path=f"edit_output_flowedit_{ind}.wav")
        flow_audio_list.append(flow_audio)
        flow_specs_list.append(flow_specs)
    flow_audio, flow_specs = _sampled_to_outputs(sampled_flow, save_path="edit_output_flowedit.wav")
    flow_audio_list.append(flow_audio)
    flow_specs_list.append(flow_specs)
    del sampled_flow
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Full pipeline: stochastic inverse + LFE
    print("[gradio_edit] (3/3) LFE pipeline")
    sampled_full, intermediate_samples_full = _run_flowedit(
        **common,
        lfe_steps=lfe_steps,
        noise_amt=noise_amt,
        inv_steps=inv_steps,
        deterministic_inverse=deterministic_inverse,
        return_intermediate_latents=True,
    )
    print(len(intermediate_samples_full))
    full_audio_list = []
    full_specs_list = []
    for ind, intermediate_sample in enumerate(intermediate_samples_full[: NUM_COLS - 1]):
        full_audio, full_specs = _sampled_to_outputs(intermediate_sample, save_path=f"edit_output_full_{ind}.wav")
        full_audio_list.append(full_audio)
        full_specs_list.append(full_specs)
    full_audio, full_specs = _sampled_to_outputs(sampled_full, save_path="edit_output_full.wav")
    full_audio_list.append(full_audio)
    full_specs_list.append(full_specs)
    del sampled_full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pad to NUM_COLS so the function always returns 2 + 4 * NUM_COLS values,
    # matching the components allocated in create_edit_ui's outputs list.
    flow_audio_list = _pad(flow_audio_list, NUM_COLS)
    flow_specs_list = _pad(flow_specs_list, NUM_COLS)
    full_audio_list = _pad(full_audio_list, NUM_COLS)
    full_specs_list = _pad(full_specs_list, NUM_COLS)

    return (
        inv_audio,
        inv_specs,
        *flow_audio_list,
        *flow_specs_list,
        *full_audio_list,
        *full_specs_list,
    )


def create_edit_ui(gradio_title=""):
    with gr.Blocks(theme=gr.themes.Base()) as ui:
        if gradio_title:
            gr.Markdown(f"### {gradio_title}")

        with gr.Row():
            with gr.Column(scale=6):
                src_prompt = gr.Textbox(
                    label="Source prompt",
                    placeholder="Describe the input audio",
                )
                tar_prompt = gr.Textbox(
                    label="Target prompt",
                    placeholder="Describe the desired edit",
                )
            generate_button = gr.Button("Generate", variant="primary", scale=1)

        with gr.Row(equal_height=False):
            with gr.Column():
                init_audio_input = gr.Audio(
                    label="Init audio",
                    # waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                    # streaming=True,
                )
            with gr.Column():

                gr.Markdown("**CFG scales**")
                with gr.Row():
                    src_inv_cfg = gr.Slider(
                        0.0, 25.0, value=1.0, step=0.1, label="src_inv_cfg_scale"
                    )
                    tar_inv_cfg = gr.Slider(
                        0.0, 25.0, value=5.0, step=0.1, label="tar_inv_cfg_scale"
                    )
                with gr.Row():
                    src_lfe_cfg = gr.Slider(
                        0.0, 25.0, value=1.0, step=0.1, label="src_lfe_cfg_scale"
                    )
                    tar_lfe_cfg = gr.Slider(
                        0.0, 25.0, value=3.0, step=0.1, label="tar_lfe_cfg_scale"
                    )

                with gr.Accordion("Edit params", open=False):
                    with gr.Row():
                        inv_steps = gr.Slider(1, 200, value=20, step=1, label="inv_steps")
                        lfe_steps = gr.Slider(1, 200, value=20, step=1, label="lfe_steps")
                    with gr.Row():
                        n_avg = gr.Slider(1, 20, value=10, step=1, label="n_avg")
                        noise_amt = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.01, label="noise_amt"
                        )
                    with gr.Row():
                        deterministic_inverse = gr.Checkbox(label="deterministic_inverse", value=True)
                        seed_textbox = gr.Textbox(
                            label="Seed (-1 for random)", value="-1"
                        )
                        intermediate_latents_interval = gr.Slider(1, 10, value=5, step=1, label="intermediate_latents_interval")

                gr.Markdown(
                    "Three outputs are generated sequentially with the same params/seed:\n"
                    "1. **Inverse only** \u2014  \n"
                    "2. **FlowEdit only** \u2014 \n"
                    "3. **Full** \u2014 LFE"
                )

        # ---- Compact-row settings shared across the LFE rows ----
        # Gradio's default Column min_width is 320 which forces wrapping when 5
        # columns are placed in a row on typical screens. min_width=160 keeps
        # all 5 on one line at >=800px viewport widths.
        col_min_width = 160
        gallery_height = 120
        spec_figsize = (3, 2)  # passed through to _sampled_to_outputs at runtime

        gr.Markdown("### 1. Inverse only")
        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                inv_audio_output = gr.Audio(
                    label="Inverse-only audio",
                    interactive=False,
                    show_label=False,
                    container=False,
                )
                inv_spec_gallery = gr.Gallery(
                    label="Inverse-only spectrogram",
                    show_label=False,
                    columns=1,
                    height=gallery_height,
                    object_fit="contain",
                    container=False,
                    preview=False,
                )

        gr.Markdown("### 2. FlowEdit only")
        with gr.Row():
            flow_audio_outputs = []
            flow_spec_galleries = []
            for i in range(NUM_COLS):
                with gr.Column(scale=1, min_width=col_min_width):
                    flow_audio_outputs.append(gr.Audio(
                        label=f"int {i}",
                        interactive=False,
                        show_label=True,
                        container=False,
                    ))
                    flow_spec_galleries.append(gr.Gallery(
                        show_label=False,
                        columns=1,
                        height=gallery_height,
                        object_fit="contain",
                        container=False,
                        preview=False,
                    ))

        gr.Markdown("### 3. Full (inverse + LFE)")
        with gr.Row():
            full_audio_outputs = []
            full_spec_galleries = []
            for i in range(NUM_COLS):
                with gr.Column(scale=1, min_width=col_min_width):
                    full_audio_outputs.append(gr.Audio(
                        label=f"int {i}",
                        interactive=False,
                        show_label=True,
                        container=False,
                    ))
                    full_spec_galleries.append(gr.Gallery(
                        show_label=False,
                        columns=1,
                        height=gallery_height,
                        object_fit="contain",
                        container=False,
                        preview=False,
                    ))

        generate_button.click(
            fn=generate_edit,
            inputs=[
                src_prompt,
                tar_prompt,
                init_audio_input,
                src_inv_cfg,
                tar_inv_cfg,
                src_lfe_cfg,
                tar_lfe_cfg,
                inv_steps,
                lfe_steps,
                n_avg,
                noise_amt,
                seed_textbox,
                deterministic_inverse,
                intermediate_latents_interval,
            ],
            outputs=[
                inv_audio_output,
                inv_spec_gallery,
                *flow_audio_outputs,
                *flow_spec_galleries,
                *full_audio_outputs,
                *full_spec_galleries,
            ],
            api_name="generate_edit",
        )
    return ui


def main(args):
    torch.manual_seed(42)

    if args.model_config is not None:
        with open(args.model_config) as f:
            cfg = json.load(f)
    else:
        cfg = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model(
        model_config=cfg,
        model_ckpt_path=args.ckpt_path,
        pretrained_name=args.pretrained_name,
        pretransform_ckpt_path=args.pretransform_ckpt_path,
        in_model_half=args.model_half,
        device=device,
    )

    interface = create_edit_ui(gradio_title=args.title or "")
    interface.queue()
    interface.launch(
        share=args.share,
        auth=(args.username, args.password) if args.username is not None else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FlowEdit gradio interface")
    parser.add_argument("--pretrained-name", type=str, required=False)
    parser.add_argument("--model-config", type=str, required=False)
    parser.add_argument("--ckpt-path", type=str, required=False)
    parser.add_argument("--pretransform-ckpt-path", type=str, required=False)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--username", type=str, required=False)
    parser.add_argument("--password", type=str, required=False)
    parser.add_argument("--model-half", action="store_true", default=True)
    parser.add_argument("--title", type=str, required=False, default="FlowEdit")
    args = parser.parse_args()
    main(args)
