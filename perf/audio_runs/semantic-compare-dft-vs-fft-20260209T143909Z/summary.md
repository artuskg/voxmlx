# DFT vs FFT Transcript Comparison

- reference: `perf/audio_runs/incremental-file-10min-20260209T135727Z/mono_incremental_transcript.txt`
- candidate: `perf/audio_runs/incremental-file-10min-fft-20260209T142155Z/mono_incremental_transcript_fft.txt`
- char distance: **14** (norm 0.001656)
- token distance: **14** (norm 0.008600)
- non-equal diff ops: **7**

## First differences
1. `replace` ref[752:754] vs fft[752:754]
   - ref: `attic and`
   - fft: `attic. And`
2. `replace` ref[762:764] vs fft[762:764]
   - ref: `me. Think`
   - fft: `me think`
3. `replace` ref[1443:1445] vs fft[1443:1445]
   - ref: `on. Machine`
   - fft: `on machine`
4. `replace` ref[1456:1458] vs fft[1456:1458]
   - ref: `on and`
   - fft: `on. And`
5. `replace` ref[1469:1471] vs fft[1469:1471]
   - ref: `Okay. So`
   - fft: `Okay, so`
6. `replace` ref[1501:1503] vs fft[1501:1503]
   - ref: `time and`
   - fft: `time. And`
7. `replace` ref[1512:1514] vs fft[1512:1514]
   - ref: `artifact from`
   - fft: `artifact. From`
