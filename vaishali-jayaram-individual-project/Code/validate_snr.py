from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


MANIFEST = Path("data/prepared/mixtures.csv")
NOISE_MANIFEST = Path("data/prepared/musan_noise_files.csv")
OUTPUT = Path("reports/audio_processing")
OUTPUT.mkdir(parents=True, exist_ok=True)

SR = 16000


def load_audio(path, start=0, frames=-1):
    """Reproduce prepare_data.py audio loading logic."""
    info = sf.info(str(path))

    if info.samplerate == SR:
        audio, _ = sf.read(
            str(path),
            start=start,
            frames=frames,
            dtype="float32",
            always_2d=True
        )
        return audio.mean(axis=1)

    import librosa
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    end = None if frames < 0 else start + frames
    return audio[start:end].astype(np.float32)


def calculate_snr(speech, noise):
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)

    if speech_power <= 0 or noise_power <= 0:
        return np.nan

    return 10 * np.log10(speech_power / noise_power)


def main():
    mixtures = pd.read_csv(MANIFEST)
    noises = pd.read_csv(NOISE_MANIFEST)

    noise_lookup = noises.set_index("noise_id")["path"].to_dict()

    results = []

    print(f"Independently validating {len(mixtures)} mixtures...")

    for i, row in enumerate(mixtures.itertuples(index=False), 1):

        speech = load_audio(row.clean_path)
        noise_path = noise_lookup[row.noise_id]

        if row.noise_looped:
            full_noise = load_audio(noise_path)
            repetitions = int(np.ceil(len(speech) / len(full_noise)))
            noise = np.tile(full_noise, repetitions)[:len(speech)]
        else:
            noise = load_audio(
                noise_path,
                start=int(row.noise_offset),
                frames=len(speech)
            )

        # Reconstruct the noise component using the recorded gain.
        scaled_noise = float(row.noise_gain) * noise

        # Theoretical SNR from independently loaded source signals.
        reconstructed_snr = calculate_snr(speech, scaled_noise)

        # Recover the noise component from the saved PCM-16 mixture.
        saved_mix = load_audio(row.path)
        recovered_mix = saved_mix / float(row.mix_rescale)
        recovered_noise = recovered_mix - speech

        actual_snr = calculate_snr(speech, recovered_noise)

        results.append({
            "split": row.split,
            "utt_id": row.utt_id,
            "noise_type": row.noise_type,
            "target_snr_db": row.snr_db,
            "stored_snr_db": row.achieved_snr_db,
            "reconstructed_snr_db": reconstructed_snr,
            "actual_snr_db": actual_snr,
            "absolute_error_db": abs(actual_snr - row.snr_db)
        })

        if i % 500 == 0:
            print(f"Validated {i}/{len(mixtures)} mixtures")

    results_df = pd.DataFrame(results)

    results_path = OUTPUT / "snr_validation_results.csv"
    results_df.to_csv(results_path, index=False)

    summary = (
        results_df
        .groupby(["noise_type", "target_snr_db"])
        .agg(
            count=("utt_id", "count"),
            mean_actual_snr=("actual_snr_db", "mean"),
            mean_absolute_error=("absolute_error_db", "mean"),
            max_absolute_error=("absolute_error_db", "max")
        )
        .round(5)
    )

    print("\nSNR VALIDATION SUMMARY")
    print(summary.to_string())

    print("\nOverall mean absolute SNR error:",
          round(results_df["absolute_error_db"].mean(), 5), "dB")

    print("Overall maximum absolute SNR error:",
          round(results_df["absolute_error_db"].max(), 5), "dB")

    print("\nDetailed results saved to:", results_path)


if __name__ == "__main__":
    main()
