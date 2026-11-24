#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Installing dependencies ==="
python -m pip install -r requirements.txt

echo "=== Checking datasets ==="
if [[ ! -d data/raw/LibriSpeech/dev-clean || \
      ! -d data/raw/LibriSpeech/test-clean || \
      ! -d data/raw/musan ]]; then
    echo "Downloading LibriSpeech and MUSAN..."
    python prepare_data.py download --root data/raw
else
    echo "Raw datasets already exist. Skipping download."
fi

echo "=== Checking prepared mixtures ==="
if [[ ! -f data/prepared/mixtures.csv ]] || \
   [[ "$(($(wc -l < data/prepared/mixtures.csv) - 1))" -ne 3600 ]]; then
    echo "Building dataset..."
    python prepare_data.py build \
        --librispeech data/raw/LibriSpeech \
        --musan data/raw/musan \
        --out data/prepared \
        --seed 6312
else
    echo "Prepared mixtures already exist. Skipping."
fi

echo "=== Checking denoised audio ==="
WAV_COUNT=$(find data/denoised -type f -name '*.wav' 2>/dev/null | wc -l)

if [[ "$WAV_COUNT" -eq 4000 && -f data/denoised/denoised_metadata.csv ]]; then
    echo "All 4,000 denoised files already exist. Skipping denoising."
else
    echo "Running denoising pipeline..."
    python -m audio_processing.process_dataset
fi

echo "=== Pipeline complete ==="
echo "Denoised WAV files: $(find data/denoised -type f -name '*.wav' | wc -l)"
echo "Metadata: data/denoised/denoised_metadata.csv"
