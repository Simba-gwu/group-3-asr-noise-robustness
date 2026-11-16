# Data preparation — LibriSpeech + MUSAN

## Run it
```bash
pip install numpy pandas soundfile librosa
python prepare_data.py download --root data/raw          # dev-clean + test-clean (~0.7 GB) and MUSAN (~11 GB)
python prepare_data.py build --librispeech data/raw/LibriSpeech \
                             --musan data/raw/musan --out data/prepared
```
Default seed is `6312`. With the same seed and the same raw data, every run produces the same samples, noise crops and mixtures.

## What's produced (`data/prepared/`)
| File | Contents |
|---|---|
| `librispeech_dev.csv` / `librispeech_test.csv` | 100 / 300 sampled utterances: `utt_id, speaker, chapter, path, duration_s, transcript` |
| `musan_noise_files.csv` | Every noise file kept, its type (`speech`/`music`/`noise`), source subfolder and `dev`/`test` split |
| `mixtures/{split}/{noise_type}/snr{20,10,0}/{utt_id}.wav` | Noisy audio, 16 kHz mono, 16-bit PCM |
| `mixtures.csv` | One row per noisy file: the clean path, transcript, noise file and crop offset, whether the noise was looped, noise gain, clipping rescale, achieved SNR |
| `summary.json` | Clip counts, speakers, total duration, noise-pool sizes and sanity checks (for the report) |

**Conditions for the modeling team:**
- **Clean audio:** `clean_path` in `mixtures.csv` (or `path` in `librispeech_*.csv`)
- **Noisy audio:** `path` in `mixtures.csv`
- **Denoised noisy and denoised clean audio:** apply `noisereduce` to those same files

Both models must read exactly these files.

## Design decisions (match the proposal)
- **Sampling.** Clips longer than 30 s are dropped, then utterances are drawn round-robin across shuffled speakers, so each speaker contributes roughly equally.
- **Background speech** comes only from `musan/speech/us-gov`. LibriVox recordings are excluded because LibriSpeech also comes from LibriVox.
- **Music** includes only tracks whose `ANNOTATIONS` vocals flag is `N`.
- **Environmental noise** includes all of `musan/noise` (free-sound + sound-bible).
- **Dev and test noise files don't overlap.** Each noise type is split 50/50 at random.
- **Each utterance gets one noise file per noise type**, cropped at a random point or looped if shorter than the utterance. **The same noise segment is used at 20, 10 and 0 dB**, so SNR is the only thing that changes between those three conditions.
- **SNR** is measured from mean power over the whole utterance and the whole noise segment. Crops that are near-silent (RMS < 1e-4) are re-drawn.
- **Clipping.** If the peak goes above 0.99, the whole mixture is scaled down by a single factor, which leaves the SNR unchanged.
