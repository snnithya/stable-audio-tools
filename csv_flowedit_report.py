"""Batch CSV -> inverse / FlowEdit-only / full-LFE WAVs plus one HTML comparison report.

Input layout: `--src-dir` contains `prompts.csv`. Each row lists a `path`, `src_prompt`, and `dst_prompt`.
The `path` column is interpreted relative to `src-dir` (absolute paths in the CSV are used as-is).

Loads the model like `gradio_edit.py`; does not launch Gradio.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from einops import rearrange

import gradio_edit as ge


def save_sampled_wav(sampled: torch.Tensor, path: Path) -> None:
    """Save model output tensor [batch, channels, samples] as normalized int16 WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = rearrange(sampled, "b d n -> d (b n)").to(torch.float32).cpu()
    peak = audio.abs().max().clamp(min=1e-8)
    audio_int16 = audio.div(peak).clamp(-1, 1).mul(32767).to(torch.int16)
    torchaudio.save(str(path), audio_int16, ge.sample_rate)


def save_prepared_src_wav(prepared_init: tuple, path: Path) -> None:
    """Save init waveform tensor before encode — same normalization as decoded outputs."""
    _sr, audio = prepared_init
    sr = _sr if isinstance(_sr, int) else int(_sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = audio.to(torch.float32).cpu()
    if a.dim() == 1:
        a = a.unsqueeze(0)
    peak = a.abs().max().clamp(min=1e-8)
    audio_int16 = a.div(peak).clamp(-1, 1).mul(32767).to(torch.int16)
    torchaudio.save(str(path), audio_int16, sr)


def load_init_audio_from_path(path: str) -> tuple:
    """Return (sr, np.float32 array) in Gradio-style layout for `_prepare_init_audio`."""
    waveform, sr = torchaudio.load(path)
    audio_np = waveform.numpy().astype(np.float32).T
    return (int(sr), audio_np)


def pad_audio_list(lst: list, n: int, fill=None) -> list:
    lst = list(lst)
    if len(lst) >= n:
        return lst[:n]
    return lst + [fill] * (n - len(lst))


def audio_tag(rel_path: str) -> str:
    return f'<audio controls preload="none" src="{html.escape(rel_path)}"></audio>'


def cell_audio(rel_path: str | None) -> str:
    if rel_path is None:
        return "—"
    return audio_tag(rel_path)


def run_row(
    row_idx: int,
    audio_path: str,
    src_prompt: str,
    dst_prompt: str,
    out_dir: Path,
    device: torch.device,
    seed: int,
    src_inv_cfg_scale: float,
    tar_inv_cfg_scale: float,
    src_lfe_cfg_scale: float,
    tar_lfe_cfg_scale: float,
    inv_steps: int,
    lfe_steps: int,
    n_avg: int,
    noise_amt: float,
    deterministic_inverse: bool,
    intermediate_latents_interval: int,
) -> dict:
    num_cols = intermediate_latents_interval + 1

    init_tuple = load_init_audio_from_path(audio_path)
    prepared_init = ge._prepare_init_audio(init_tuple)

    prefix = f"r{row_idx:04d}"
    rel = lambda p: str(Path(p).relative_to(out_dir))

    src_wav = out_dir / f"{prefix}_src.wav"
    save_prepared_src_wav(prepared_init, src_wav)
    rel_src = rel(src_wav)

    seconds_total = ge.sample_size // ge.sample_rate
    src_conditioning = [
        {
            "prompt": src_prompt,
            "seconds_start": 0,
            "seconds_total": seconds_total,
        }
    ]
    tar_conditioning = [
        {
            "prompt": dst_prompt,
            "seconds_start": 0,
            "seconds_total": seconds_total,
        }
    ]

    common = dict(
        src_conditioning=src_conditioning,
        tar_conditioning=tar_conditioning,
        init_audio=prepared_init,
        device=device,
        seed=int(seed),
        src_inv_cfg_scale=src_inv_cfg_scale,
        tar_inv_cfg_scale=tar_inv_cfg_scale,
        src_lfe_cfg_scale=src_lfe_cfg_scale,
        tar_lfe_cfg_scale=tar_lfe_cfg_scale,
        n_avg=n_avg,
        intermediate_latents_interval=intermediate_latents_interval,
    )

    # 1. Inverse only
    sampled_inv, _ = ge._run_flowedit(
        **common,
        lfe_steps=0,
        noise_amt=1,
        deterministic_inverse=True,
        inv_steps=inv_steps,
        return_intermediate_latents=False,
    )
    inv_path = out_dir / f"{prefix}_inv.wav"
    save_sampled_wav(sampled_inv, inv_path)
    del sampled_inv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 2. FlowEdit only
    sampled_flow, intermediate_flow = ge._run_flowedit(
        **common,
        lfe_steps=lfe_steps,
        inv_steps=0,
        deterministic_inverse=deterministic_inverse,
        noise_amt=0,
        return_intermediate_latents=True,
    )
    inter_paths_flow: list[str | None] = []
    for j, inter in enumerate(pad_audio_list(intermediate_flow, num_cols - 1, fill=None)):
        if inter is None:
            inter_paths_flow.append(None)
            continue
        p = out_dir / f"{prefix}_fe_int{j:02d}.wav"
        save_sampled_wav(inter, p)
        inter_paths_flow.append(rel(p))
    final_fe = out_dir / f"{prefix}_fe_final.wav"
    save_sampled_wav(sampled_flow, final_fe)
    rel_final_fe = rel(final_fe)
    del sampled_flow
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 3. Full (inverse + LFE)
    sampled_full, intermediate_full = ge._run_flowedit(
        **common,
        lfe_steps=lfe_steps,
        noise_amt=noise_amt,
        inv_steps=inv_steps,
        deterministic_inverse=deterministic_inverse,
        return_intermediate_latents=True,
    )
    inter_paths_lfe: list[str | None] = []
    for j, inter in enumerate(pad_audio_list(intermediate_full, num_cols - 1, fill=None)):
        if inter is None:
            inter_paths_lfe.append(None)
            continue
        p = out_dir / f"{prefix}_lfe_int{j:02d}.wav"
        save_sampled_wav(inter, p)
        inter_paths_lfe.append(rel(p))
    final_lfe = out_dir / f"{prefix}_lfe_final.wav"
    save_sampled_wav(sampled_full, final_lfe)
    rel_final_lfe = rel(final_lfe)
    del sampled_full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    flow_cols = inter_paths_flow + [rel_final_fe]
    flow_cols = flow_cols[:num_cols]
    flow_cols += [None] * max(0, num_cols - len(flow_cols))

    lfe_cols = inter_paths_lfe + [rel_final_lfe]
    lfe_cols = lfe_cols[:num_cols]
    lfe_cols += [None] * max(0, num_cols - len(lfe_cols))

    return {
        "idx": row_idx,
        "src_audio_rel": rel_src,
        "src_prompt": src_prompt,
        "dst_prompt": dst_prompt,
        "inv_rel": rel(inv_path),
        "flow_cols": flow_cols,
        "lfe_cols": lfe_cols,
        "seed_used": int(seed),
    }


def render_html(rows: list[dict], settings: dict, num_cols: int, title: str) -> str:
    def params_table(kv: dict) -> str:
        body = "".join(
            f"<tr><th>{html.escape(str(k))}</th><td><code>{html.escape(str(v))}</code></td></tr>"
            for k, v in kv.items()
        )
        return f"<table class='params'><tbody>{body}</tbody></table>"

    flow_headers = "".join(
        f"<th>FE step {i}</th>" if i < num_cols - 1 else "<th>FE final</th>"
        for i in range(num_cols)
    )
    lfe_headers = "".join(
        f"<th>LFE step {i}</th>" if i < num_cols - 1 else "<th>LFE final</th>"
        for i in range(num_cols)
    )

    inv_rows_html = ""
    for r in rows:
        inv_rows_html += (
            "<tr>"
            f"<td>{cell_audio(r['src_audio_rel'])}</td>"
            f"<td class='prompt'>{html.escape(r['src_prompt'])}</td>"
            f"<td class='prompt'>{html.escape(r['dst_prompt'])}</td>"
            f"<td>{cell_audio(r['inv_rel'])}</td>"
            "</tr>"
        )

    fe_rows_html = ""
    for r in rows:
        # Each sample must be in its own <td>; bare <audio> inside <tr> is invalid and gets hoisted out.
        cells = "".join(f"<td>{cell_audio(c)}</td>" for c in r["flow_cols"])
        fe_rows_html += (
            "<tr>"
            f"<td>{cell_audio(r['src_audio_rel'])}</td>"
            f"<td class='prompt'>{html.escape(r['src_prompt'])}</td>"
            f"<td class='prompt'>{html.escape(r['dst_prompt'])}</td>"
            f"{cells}"
            "</tr>"
        )

    lfe_rows_html = ""
    for r in rows:
        cells = "".join(f"<td>{cell_audio(c)}</td>" for c in r["lfe_cols"])
        lfe_rows_html += (
            "<tr>"
            f"<td>{cell_audio(r['src_audio_rel'])}</td>"
            f"<td class='prompt'>{html.escape(r['src_prompt'])}</td>"
            f"<td class='prompt'>{html.escape(r['dst_prompt'])}</td>"
            f"{cells}"
            "</tr>"
        )

    css = """
    body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #1a1a1a; }
    h1 { font-size: 1.25rem; }
    h2 { font-size: 1.05rem; margin-top: 2rem; }
    table.data { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
    table.data th, table.data td { border: 1px solid #ccc; padding: 0.35rem 0.5rem; vertical-align: top; }
    table.data th { background: #f4f4f4; text-align: left; }
    td.prompt { max-width: 22rem; font-size: 0.85rem; white-space: pre-wrap; }
    audio { width: 12rem; height: 2rem; display: block; }
    table.params th { text-align: left; padding-right: 1rem; vertical-align: top; }
    table.params td code { white-space: pre-wrap; word-break: break-all; }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{css}</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p>Settings used for all rows:</p>
{params_table(settings)}

<h2>1. Inverse only</h2>
<table class="data">
<thead><tr>
<th>Source audio</th><th>Source prompt</th><th>Dest prompt</th><th>Inverse</th>
</tr></thead>
<tbody>{inv_rows_html}</tbody>
</table>

<h2>2. FlowEdit only</h2>
<table class="data">
<thead><tr>
<th>Source audio</th><th>Source prompt</th><th>Dest prompt</th>
{flow_headers}
</tr></thead>
<tbody>{fe_rows_html}</tbody>
</table>

<h2>3. Full latent FlowEdit (inverse + LFE)</h2>
<table class="data">
<thead><tr>
<th>Source audio</th><th>Source prompt</th><th>Dest prompt</th>
{lfe_headers}
</tr></thead>
<tbody>{lfe_rows_html}</tbody>
</table>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV batch FlowEdit HTML report")
    parser.add_argument(
        "--src-dir",
        type=str,
        required=True,
        help="Directory containing prompts.csv; audio paths in the CSV are relative to this directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for WAVs and report.html",
    )
    parser.add_argument("--report-name", type=str, default="report.html")
    parser.add_argument("--pretrained-name", type=str, required=False)
    parser.add_argument("--model-config", type=str, required=False)
    parser.add_argument("--ckpt-path", type=str, required=False)
    parser.add_argument("--pretransform-ckpt-path", type=str, required=False)
    parser.add_argument("--model-half", action="store_true", default=True)
    parser.add_argument("--title", type=str, default="FlowEdit batch report")
    parser.add_argument("--seed", type=int, default=-1, help="-1: random per row")
    parser.add_argument("--src-inv-cfg", type=float, default=1.0)
    parser.add_argument("--tar-inv-cfg", type=float, default=5.0)
    parser.add_argument("--src-lfe-cfg", type=float, default=1.0)
    parser.add_argument("--tar-lfe-cfg", type=float, default=3.0)
    parser.add_argument("--inv-steps", type=int, default=20)
    parser.add_argument("--lfe-steps", type=int, default=20)
    parser.add_argument("--n-avg", type=int, default=10)
    parser.add_argument("--noise-amt", type=float, default=0.5)
    parser.add_argument(
        "--deterministic-inverse",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--intermediate-latents-interval",
        type=int,
        default=5,
        help="Table audio columns for FE/LFE = this + 1 (intermediates + final).",
    )
    args = parser.parse_args()

    if args.intermediate_latents_interval < 1:
        raise SystemExit("--intermediate-latents-interval must be >= 1")

    num_cols = args.intermediate_latents_interval + 1
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model_config is not None:
        with open(args.model_config) as f:
            cfg = json.load(f)
    else:
        cfg = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ge.load_model(
        model_config=cfg,
        model_ckpt_path=args.ckpt_path,
        pretrained_name=args.pretrained_name,
        pretransform_ckpt_path=args.pretransform_ckpt_path,
        in_model_half=args.model_half,
        device=device,
    )

    src_dir = Path(args.src_dir).resolve()
    csv_path = src_dir / "prompts.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing prompts.csv under src-dir: {csv_path}")

    rows_out: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"path", "src_prompt", "dst_prompt"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise SystemExit(
                f"CSV must include headers: {sorted(expected)}; got {reader.fieldnames!r}"
            )
        for row_idx, row in enumerate(reader):
            ap = (row.get("path") or "").strip()
            sp = (row.get("src_prompt") or "").strip()
            dp = (row.get("dst_prompt") or "").strip()
            if not ap or not sp or not dp:
                raise SystemExit(f"Row {row_idx}: empty path or prompt")

            audio_path_raw = Path(ap)
            audio_path_abs = (
                audio_path_raw
                if audio_path_raw.is_absolute()
                else (src_dir / audio_path_raw).resolve()
            )
            if not audio_path_abs.is_file():
                raise SystemExit(f"Row {row_idx}: audio file not found: {audio_path_abs}")

            seed = int(args.seed)
            if seed == -1:
                seed = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))

            row_data = run_row(
                row_idx=row_idx,
                audio_path=str(audio_path_abs),
                src_prompt=sp,
                dst_prompt=dp,
                out_dir=out_dir,
                device=device,
                seed=seed,
                src_inv_cfg_scale=args.src_inv_cfg,
                tar_inv_cfg_scale=args.tar_inv_cfg,
                src_lfe_cfg_scale=args.src_lfe_cfg,
                tar_lfe_cfg_scale=args.tar_lfe_cfg,
                inv_steps=args.inv_steps,
                lfe_steps=args.lfe_steps,
                n_avg=args.n_avg,
                noise_amt=args.noise_amt,
                deterministic_inverse=args.deterministic_inverse,
                intermediate_latents_interval=args.intermediate_latents_interval,
            )
            rows_out.append(row_data)
            print(f"[csv_flowedit_report] done row {row_idx} seed={row_data['seed_used']}")

    settings = {
        "src_dir": str(src_dir),
        "prompts_csv": str(csv_path),
        "out_dir": str(out_dir),
        "sample_rate": ge.sample_rate,
        "sample_size": ge.sample_size,
        "model_half": ge.model_half,
        "pretrained_name": args.pretrained_name,
        "ckpt_path": args.ckpt_path,
        "num_cols (FE/LFE)": num_cols,
        "intermediate_latents_interval": args.intermediate_latents_interval,
        "src_inv_cfg_scale": args.src_inv_cfg,
        "tar_inv_cfg_scale": args.tar_inv_cfg,
        "src_lfe_cfg_scale": args.src_lfe_cfg,
        "tar_lfe_cfg_scale": args.tar_lfe_cfg,
        "inv_steps": args.inv_steps,
        "lfe_steps": args.lfe_steps,
        "n_avg": args.n_avg,
        "noise_amt": args.noise_amt,
        "deterministic_inverse": args.deterministic_inverse,
        "seed_mode": "per-row random" if args.seed == -1 else f"fixed {args.seed}",
    }

    report_path = out_dir / args.report_name
    report_path.write_text(
        render_html(rows_out, settings, num_cols=num_cols, title=args.title),
        encoding="utf-8",
    )
    print(f"[csv_flowedit_report] wrote {report_path}")


if __name__ == "__main__":
    main()
