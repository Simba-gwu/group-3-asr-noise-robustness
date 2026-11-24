"""
Batch denoising pipeline for the NLP ASR project.

Processes each unique clean recording once and all noisy mixtures.
Original files remain unchanged.
"""

from pathlib import Path
import time

import pandas as pd

from audio_processing.denoise import denoise_file

MANIFEST = Path("data/prepared/mixtures.csv")
OUTPUT_ROOT = Path("data/denoised")
METADATA_OUTPUT = OUTPUT_ROOT / "denoised_metadata.csv"


def main(test_mode=False):
    df = pd.read_csv(MANIFEST)

    required = {
        "split", "utt_id", "noise_type", "snr_db",
        "path", "clean_path"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")

    records = []
    start = time.time()

    # Process each unique clean recording only once.
    clean_files = df[["split", "utt_id", "clean_path"]].drop_duplicates(
        subset=["split", "utt_id"]
    )

    if test_mode:
        clean_files = clean_files.head(2)
        df = df.groupby(["split", "utt_id"], sort=False).head(3).head(6)

    print(f"Processing {len(clean_files)} unique clean recordings...")

    for i, row in enumerate(clean_files.itertuples(index=False), 1):
        output = (
            OUTPUT_ROOT / "clean" / row.split /
            f"{row.utt_id}.wav"
        )

        result = denoise_file(row.clean_path, output)

        records.append({
            "split": row.split,
            "utt_id": row.utt_id,
            "audio_type": "clean",
            "noise_type": None,
            "snr_db": None,
            **result
        })

        if i % 25 == 0:
            print(f"Clean recordings: {i}/{len(clean_files)}")

    # Process every noisy mixture.
    print(f"\nProcessing {len(df)} noisy recordings...")

    for i, row in enumerate(df.itertuples(index=False), 1):
        output = (
            OUTPUT_ROOT / "noisy" / row.split /
            row.noise_type / f"snr{row.snr_db}" /
            f"{row.utt_id}.wav"
        )

        result = denoise_file(row.path, output)

        records.append({
            "split": row.split,
            "utt_id": row.utt_id,
            "audio_type": "noisy",
            "noise_type": row.noise_type,
            "snr_db": row.snr_db,
            **result
        })

        if i % 100 == 0:
            print(f"Noisy recordings: {i}/{len(df)}")

    metadata = pd.DataFrame(records)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(METADATA_OUTPUT, index=False)

    print("\nBatch processing completed.")
    print(f"Total files processed: {len(metadata)}")
    print(f"Clean files: {(metadata.audio_type == 'clean').sum()}")
    print(f"Noisy files: {(metadata.audio_type == 'noisy').sum()}")
    print(f"Metadata saved to: {METADATA_OUTPUT}")
    print(f"Total time: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    import sys
    main(test_mode="--test" in sys.argv)
