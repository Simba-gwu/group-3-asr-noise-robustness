"""
Data preparation for "Evaluating Speech Recognition Models Under Background Noise"
DATS 6312 – NLP project.  Owner: data preparation (LibriSpeech + MUSAN).

What this script does (all deterministic given --seed):
  1. Samples 100 utterances from LibriSpeech dev-clean and 300 from test-clean,
     spread across speakers (round-robin), excluding clips > 30 s.
  2. Builds the MUSAN noise pools:
       speech -> musan/speech/us-gov only   (LibriVox excluded: LibriSpeech comes from LibriVox)
       music  -> musan/music/*  with vocals == N in ANNOTATIONS
       noise  -> musan/noise/*  (free-sound + sound-bible)
     and splits each pool into disjoint dev / test noise files.
  3. For every utterance x noise type, picks one noise file and one crop
     (looped if shorter than the utterance) and mixes it at 20, 10 and 0 dB SNR.
     The same noise segment is reused across the three SNRs, so SNR is the only
     thing that changes. Mixtures are rescaled if needed to avoid clipping
     (a single gain on the whole mixture, so the SNR is unchanged).
  4. Writes manifests (CSV) + a summary JSON documenting counts, speakers,
     durations and the achieved SNR of every mixture.

Both ASR models then read exactly the same WAV files listed in the manifests.

Run it from the project root: all paths in the manifests are relative to it.

Usage:
  python prepare_data.py download --root data/raw            # optional, ~12 GB total
  python prepare_data.py build --librispeech data/raw/LibriSpeech \
                               --musan data/raw/musan --out data/prepared
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16_000
MAX_DUR_S = 30.0
SNRS_DB = (20, 10, 0)
NOISE_TYPES = ("speech", "music", "noise")
MIN_NOISE_RMS = 1e-4          # reject (near-)silent noise crops
PEAK_LIMIT = 0.99             # max |sample| after mixing

URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "musan": "https://www.openslr.org/resources/17/musan.tar.gz",
}


# --------------------------------------------------------------------------- utils
def rel(p) -> str:
    """Path relative to the folder you run the script from, so manifests are portable."""
    return os.path.relpath(p).replace(os.sep, "/")


def stable_rng(seed: int, *keys: str) -> np.random.Generator:
    """RNG that depends only on seed + keys (not on iteration order)."""
    return np.random.default_rng([seed] + [zlib.crc32(k.encode()) for k in keys])


def load_audio(path: str | Path, start: int = 0, frames: int = -1) -> np.ndarray:
    """Read audio as float32 mono at 16 kHz (partial reads when already 16 kHz)."""
    info = sf.info(str(path))
    if info.samplerate == SR:
        x, _ = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
        return x.mean(axis=1)
    import librosa  # only needed for non-16k files
    x, _ = librosa.load(str(path), sr=SR, mono=True)
    end = None if frames < 0 else start + frames
    return x[start:end].astype(np.float32)


def num_frames_16k(path: str | Path) -> int:
    info = sf.info(str(path))
    return int(round(info.frames * SR / info.samplerate))


# ----------------------------------------------------------------- LibriSpeech
def scan_librispeech(split_dir: Path) -> pd.DataFrame:
    rows = []
    for trans in sorted(split_dir.glob("*/*/*.trans.txt")):
        speaker, chapter = trans.parent.parent.name, trans.parent.name
        for line in trans.read_text().splitlines():
            utt_id, text = line.split(" ", 1)
            flac = trans.parent / f"{utt_id}.flac"
            info = sf.info(str(flac))
            rows.append(dict(utt_id=utt_id, speaker=speaker, chapter=chapter,
                             path=rel(flac), duration_s=info.frames / info.samplerate,
                             transcript=text.strip()))
    if not rows:
        raise FileNotFoundError(f"No LibriSpeech transcripts under {split_dir}")
    return pd.DataFrame(rows)


def sample_across_speakers(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Round-robin over shuffled speakers so every speaker contributes ~equally."""
    df = df[df.duration_s <= MAX_DUR_S]
    rng = np.random.default_rng(seed)
    speakers = sorted(df.speaker.unique())
    rng.shuffle(speakers)
    queues = {s: list(rng.permutation(df.index[df.speaker == s])) for s in speakers}
    picked = []
    while len(picked) < n and any(queues.values()):
        for s in speakers:
            if queues[s] and len(picked) < n:
                picked.append(queues[s].pop())
    if len(picked) < n:
        raise ValueError(f"Only {len(picked)} utterances available, need {n}")
    return df.loc[picked].sort_values("utt_id").reset_index(drop=True)


# ----------------------------------------------------------------------- MUSAN
def read_music_annotations(music_dir: Path) -> dict[str, str]:
    """utt_id -> 'Y'/'N' vocals flag from musan/music/*/ANNOTATIONS."""
    vocals = {}
    for ann in music_dir.glob("*/ANNOTATIONS"):
        for line in ann.read_text(errors="ignore").splitlines():
            tok = line.split()
            if len(tok) >= 3 and tok[2].upper() in ("Y", "N"):
                vocals[tok[0]] = tok[2].upper()
    return vocals


def scan_musan(musan: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((musan / "speech" / "us-gov").glob("*.wav")):
        rows.append(dict(noise_type="speech", source="us-gov", noise_id=f.stem, path=rel(f)))

    vocals = read_music_annotations(musan / "music")
    n_vocal = n_unannotated = 0
    for f in sorted((musan / "music").glob("*/*.wav")):
        flag = vocals.get(f.stem)
        if flag == "N":
            rows.append(dict(noise_type="music", source=f.parent.name, noise_id=f.stem, path=rel(f)))
        elif flag == "Y":
            n_vocal += 1
        else:
            n_unannotated += 1
    print(f"[musan] music: dropped {n_vocal} tracks with vocals, {n_unannotated} without annotation")

    for f in sorted((musan / "noise").glob("*/*.wav")):
        rows.append(dict(noise_type="noise", source=f.parent.name, noise_id=f.stem, path=rel(f)))

    df = pd.DataFrame(rows)
    if df.empty or set(df.noise_type) != set(NOISE_TYPES):
        raise FileNotFoundError(f"Missing MUSAN categories under {musan}: found {sorted(set(df.get('noise_type', [])))}")
    df["duration_s"] = [num_frames_16k(p) / SR for p in df.path]
    return df


def split_noise(df: pd.DataFrame, seed: int, dev_frac: float = 0.5) -> pd.DataFrame:
    """Disjoint dev/test noise files, split separately within each noise type."""
    df = df.copy()
    df["split"] = "test"
    for t in NOISE_TYPES:
        idx = df.index[df.noise_type == t].to_numpy()
        idx = stable_rng(seed, "noise-split", t).permutation(idx)
        df.loc[idx[: int(round(len(idx) * dev_frac))], "split"] = "dev"
    return df


# ---------------------------------------------------------------------- mixing
def noise_segment(path: str, length: int, rng: np.random.Generator, max_tries: int = 20):
    """Random crop of `length` samples; loop the file if it is shorter."""
    total = num_frames_16k(path)
    if total >= length:
        for _ in range(max_tries):
            offset = int(rng.integers(0, total - length + 1))
            seg = load_audio(path, start=offset, frames=length)
            if np.sqrt(np.mean(seg ** 2)) > MIN_NOISE_RMS:
                return seg, offset, False
        return None
    full = load_audio(path)
    if np.sqrt(np.mean(full ** 2)) <= MIN_NOISE_RMS:
        return None
    reps = int(np.ceil(length / len(full)))
    return np.tile(full, reps)[:length], 0, True


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float):
    p_s = np.mean(speech ** 2)
    p_n = np.mean(noise ** 2)
    gain = np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))
    scaled_noise = gain * noise
    mix = speech + scaled_noise
    peak = np.max(np.abs(mix))
    rescale = min(1.0, PEAK_LIMIT / peak) if peak > 0 else 1.0
    achieved = 10 * np.log10(p_s / np.mean(scaled_noise ** 2))
    return (mix * rescale).astype(np.float32), gain, rescale, achieved


def build_mixtures(utts: pd.DataFrame, noise: pd.DataFrame, split: str, out: Path, seed: int) -> pd.DataFrame:
    pool = {t: noise[(noise.split == split) & (noise.noise_type == t)].reset_index(drop=True) for t in NOISE_TYPES}
    rows = []
    for i, u in utts.iterrows():
        speech = load_audio(u.path)
        for t in NOISE_TYPES:
            rng = stable_rng(seed, split, u.utt_id, t)
            seg = None
            while seg is None:  # skip noise files that are silent everywhere
                nrow = pool[t].iloc[int(rng.integers(len(pool[t])))]
                seg = noise_segment(nrow.path, len(speech), rng)
            noise_sig, offset, looped = seg
            for snr in SNRS_DB:
                mix, gain, rescale, achieved = mix_at_snr(speech, noise_sig, snr)
                dst = out / "mixtures" / split / t / f"snr{snr}" / f"{u.utt_id}.wav"
                dst.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(dst), mix, SR, subtype="PCM_16")
                rows.append(dict(split=split, utt_id=u.utt_id, speaker=u.speaker, noise_type=t,
                                 snr_db=snr, path=rel(dst), clean_path=u.path,
                                 noise_id=nrow.noise_id, noise_source=nrow.source,
                                 noise_offset=offset, noise_looped=looped,
                                 noise_gain=gain, mix_rescale=rescale, achieved_snr_db=achieved,
                                 transcript=u.transcript))
        if (i + 1) % 25 == 0 or i + 1 == len(utts):
            print(f"[mix] {split}: {i + 1}/{len(utts)} utterances")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ main
def cmd_download(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for name in args.only or URLS:
        tgz = root / f"{name}.tar.gz"
        if not tgz.exists():
            print(f"downloading {URLS[name]} ...")
            urllib.request.urlretrieve(URLS[name], tgz)
        print(f"extracting {tgz} ...")
        with tarfile.open(tgz) as tf:
            tf.extractall(root, filter="data")


def summarize(df: pd.DataFrame) -> dict:
    return dict(n_utterances=int(len(df)), n_speakers=int(df.speaker.nunique()),
                total_duration_min=round(float(df.duration_s.sum()) / 60, 2),
                mean_duration_s=round(float(df.duration_s.mean()), 2),
                max_duration_s=round(float(df.duration_s.max()), 2))


def cmd_build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ls = Path(args.librispeech)

    splits = {"dev": ("dev-clean", args.n_dev), "test": ("test-clean", args.n_test)}
    utts = {}
    for split, (folder, n) in splits.items():
        full = scan_librispeech(ls / folder)
        utts[split] = sample_across_speakers(full, n, stable_rng(args.seed, split).integers(2**31))
        utts[split].to_csv(out / f"librispeech_{split}.csv", index=False)
        print(f"[librispeech] {split}: {summarize(utts[split])}")

    noise = split_noise(scan_musan(Path(args.musan)), args.seed)
    noise.to_csv(out / "musan_noise_files.csv", index=False)

    mixes = pd.concat([build_mixtures(utts[s], noise, s, out, args.seed) for s in splits], ignore_index=True)
    mixes.to_csv(out / "mixtures.csv", index=False)

    summary = dict(
        seed=args.seed, sample_rate=SR, max_duration_s=MAX_DUR_S, snrs_db=list(SNRS_DB),
        librispeech={s: summarize(utts[s]) for s in splits},
        musan={f"{t}/{s}": dict(n_files=int(len(g)), hours=round(float(g.duration_s.sum()) / 3600, 2))
               for (t, s), g in noise.groupby(["noise_type", "split"])},
        mixtures=dict(
            n_files=int(len(mixes)),
            n_rescaled_to_avoid_clipping=int((mixes.mix_rescale < 1).sum()),
            n_looped_noise=int(mixes.noise_looped.sum()),
            max_abs_snr_error_db=round(float((mixes.achieved_snr_db - mixes.snr_db).abs().max()), 4),
        ),
        overlap_checks=dict(
            utterances_dev_test=len(set(utts["dev"].utt_id) & set(utts["test"].utt_id)),
            noise_files_dev_test=len(set(noise[noise.split == "dev"].path) & set(noise[noise.split == "test"].path)),
        ),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--root", default="data/raw")
    d.add_argument("--only", nargs="*", choices=list(URLS))
    b = sub.add_parser("build")
    b.add_argument("--librispeech", required=True, help="folder containing dev-clean/ and test-clean/")
    b.add_argument("--musan", required=True, help="folder containing speech/ music/ noise/")
    b.add_argument("--out", default="data/prepared")
    b.add_argument("--n-dev", type=int, default=100)
    b.add_argument("--n-test", type=int, default=300)
    b.add_argument("--seed", type=int, default=6312)
    args = p.parse_args()
    {"download": cmd_download, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
