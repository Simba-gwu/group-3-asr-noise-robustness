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
