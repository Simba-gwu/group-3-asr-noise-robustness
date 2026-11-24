# Audio Processing and Denoising

This module handles audio validation, noise reduction, dataset processing,
visualization, and independent SNR validation for the ASR noise robustness project.

## Methodology

- Sample rate: 16,000 Hz
- Audio format: Mono
- Denoising library: noisereduce
- Method: Non-stationary spectral noise reduction
- Noise reduction strength (prop_decrease): 0.8
- Output format: PCM 16-bit WAV

Original clean and noisy audio files are preserved.
Denoised files are written separately.

## Scripts

- denoise.py: Validates and denoises an individual audio file.
- process_dataset.py: Processes clean and noisy audio and creates metadata.
- validate_snr.py: Independently validates achieved SNR against target SNR.
- visualize_audio.py: Generates waveform and spectrogram comparisons.

## Run

From the project root:

    python -m audio_processing.process_dataset
    python -m audio_processing.validate_snr
    python -m audio_processing.visualize_audio

## Output locations

- Clean denoised audio: data/denoised/clean/
- Noisy denoised audio: data/denoised/noisy/
- Metadata: data/denoised/denoised_metadata.csv
- Validation results and plots: reports/audio_processing/

## Processing Results

- Clean audio files processed: 400
- Noisy mixtures processed: 3,600
- Total denoised files: 4,000
- Missing outputs: 0
- Invalid audio files: 0
- Incorrect sample rates: 0
- Non-mono outputs: 0

## Independent SNR Validation

All 3,600 noisy mixtures were independently checked.

- Mean absolute SNR error: 0.00014 dB
- Maximum absolute SNR error: 0.01573 dB

The small differences are consistent with numerical and PCM quantization effects.

SNR validation confirms the accuracy of noisy mixture construction.
It does not establish that denoising improves transcription accuracy;
that comparison belongs to the ASR evaluation stage.

Note: The audio dataset and generated denoised audio are stored locally
and are not committed to GitHub.

## Reproducing the Complete Pipeline

The project includes a setup script that automates dependency installation,
dataset preparation, and audio denoising.

### Requirements

- Python 3.13 or compatible Python version
- Approximately 12 GB for the original datasets, plus additional storage
  for prepared and denoised audio
- Internet connection for downloading LibriSpeech and MUSAN

### Setup

From the project root:

    python3 -m venv asr-env
    source asr-env/bin/activate
    ./setup_pipeline.sh

The script automatically:

1. Installs dependencies from requirements.txt.
2. Downloads LibriSpeech and MUSAN if the raw datasets are missing.
3. Builds the prepared dataset using random seed 6312.
4. Processes 400 clean files and 3,600 noisy mixtures.
5. Saves denoised audio and metadata under data/denoised/.

Existing datasets and complete denoised outputs are detected and skipped
to avoid unnecessary downloads and processing.

### Expected Results

- 400 clean audio files
- 3,600 noisy audio mixtures
- 4,000 denoised WAV files
- Metadata CSV containing 4,000 records

The raw datasets and generated audio files are stored locally and are
not committed to GitHub. Each team member can reproduce the pipeline
by running the setup script.
