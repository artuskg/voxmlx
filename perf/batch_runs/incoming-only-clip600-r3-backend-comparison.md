# Incoming 10-min Campaign (3 runs each): DFT vs FFT

Ground truth: `/tmp/ground_truth_mono_correctness_cb81d75.txt`
Audio: `/tmp/voxtral_ref_mono_10min.wav`, `/tmp/voxtral_ref_stereo_10min.wav`

## Timing means (mono+stereo total)

| version | DFT mean (s) | FFT mean (s) | delta (FFT-DFT) |
|---|---:|---:|---:|
| incoming_59735f8 | 315.152 | 324.734 | +9.583 (+3.041%) |
| incoming_cce41e1 | 305.053 | 309.128 | +4.076 (+1.336%) |
| incoming_dc994b1 | 304.737 | 313.851 | +9.114 (+2.991%) |

## Key observation

- Mono transcript is identical between DFT and FFT in tested runs (`sha256=c884ae4d...`, `1081` chars).
- Stereo transcript diverges strongly under FFT (example `incoming_dc994b1` run 2):
  - DFT stereo: `1081` chars, `sha256=13e3ad01...`
  - FFT stereo: `6615` chars, `sha256=785999f5...`
- Because only stereo changes, FFT campaign quality metrics improve numerically vs this reference (which is mono-derived) while timing is slower.

## Practical interpretation

- FFT path is slower here for all three incoming implementations.
- Metric shift is mostly from FFT changing stereo text behavior, not from mono improvement.
