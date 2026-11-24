from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

MANIFEST = pd.read_csv("data/prepared/mixtures.csv")
OUTPUT = Path("reports/audio_processing")
OUTPUT.mkdir(parents=True, exist_ok=True)


def plot_spectrogram(ax, audio, sr, title):
    frequencies, times, power = spectrogram(
        audio, fs=sr, nperseg=512, noverlap=384
    )
    ax.pcolormesh(
        times, frequencies, 10 * np.log10(power + 1e-12),
        shading="auto"
    )
    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (seconds)")


for noise_type in ["speech", "music", "noise"]:
    row = MANIFEST[
        (MANIFEST["split"] == "dev") &
        (MANIFEST["noise_type"] == noise_type) &
        (MANIFEST["snr_db"] == 0)
    ].iloc[0]

    clean, sr = sf.read(row["clean_path"])
    noisy, _ = sf.read(row["path"])

    denoised_path = (
        Path("data/denoised/noisy") / "dev" /
        noise_type / "snr0" / f"{row['utt_id']}.wav"
    )
    denoised, _ = sf.read(denoised_path)

    signals = [
        ("Original clean", clean),
        ("Noisy mixture", noisy),
        ("Denoised noisy", denoised)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for col, (title, audio) in enumerate(signals):
        time_axis = np.arange(len(audio)) / sr

        axes[0, col].plot(time_axis, audio)
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("Time (seconds)")
        axes[0, col].set_ylabel("Amplitude")

        plot_spectrogram(axes[1, col], audio, sr, title)

    fig.suptitle(
        f"{noise_type.title()} Noise | 0 dB SNR | {row['utt_id']}"
    )
    fig.tight_layout()

    output_file = OUTPUT / f"{noise_type}_comparison.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_file}")

print("\nAll comparison plots generated.")
