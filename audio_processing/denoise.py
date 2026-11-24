
"""
Audio denoising pipeline for the NLP speech recognition project.

Processes clean and noisy audio without modifying original files.
"""

from pathlib import Path

import numpy as np
import noisereduce as nr
import soundfile as sf


SAMPLE_RATE = 16000

# Denoising configuration
PROP_DECREASE = 0.8
STATIONARY = False


def validate_audio(audio, sample_rate):
    """Check audio shape, sample rate and numerical validity."""

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, received {sample_rate} Hz"
        )

    if audio.ndim != 1:
        raise ValueError("Expected mono audio.")

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numerical values.")


def denoise_file(input_path, output_path):
    """Denoise one audio file and save a separate output."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Read original audio
    audio, sample_rate = sf.read(
        input_path,
        dtype="float32"
    )

    validate_audio(audio, sample_rate)

    # Apply spectral-gating noise reduction
    denoised_audio = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=STATIONARY,
        prop_decrease=PROP_DECREASE
    )

    # Preserve output as float32 WAV
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        output_path,
        denoised_audio.astype(np.float32),
        sample_rate,
        subtype="PCM_16"
    )

    print(f"Processed: {input_path.name}")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "sample_rate": sample_rate,
        "duration_seconds": len(audio) / sample_rate
    }


if __name__ == "__main__":
    print("Audio denoising module is ready.")
