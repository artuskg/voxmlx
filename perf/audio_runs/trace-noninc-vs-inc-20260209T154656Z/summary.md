# Non-incremental vs Incremental Trace Comparison

- audio: `perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav`
- clip_seconds: `120`
- stft_backend: `dft`
- non_incremental elapsed: `14.564s`
- incremental elapsed: `33.299s`
- token distance: `384` (norm `0.254305`)
- char distance (KEEP): `3369` (norm `0.180498`)
- first token divergence index: `50`
- first meaningful divergence index: `50`
- meaningful span count: `64`
- focus indices: `[48, 49, 50, 51, 52]`
- non_incremental replay matches generate: `True`

## First diff spans
1. `delete` noninc[50:51] vs inc[50:50], meaningful=`True`
   - noninc: `[STREAMING_PAD]`
   - inc: ``
2. `insert` noninc[53:53] vs inc[52:53], meaningful=`True`
   - noninc: ``
   - inc: `[STREAMING_PAD]`
3. `insert` noninc[60:60] vs inc[60:62], meaningful=`True`
   - noninc: ``
   - inc: `[STREAMING_WORD] showing`
4. `delete` noninc[62:63] vs inc[64:64], meaningful=`True`
   - noninc: ` showing`
   - inc: ``
5. `delete` noninc[64:65] vs inc[65:65], meaningful=`True`
   - noninc: `[STREAMING_PAD]`
   - inc: ``
6. `insert` noninc[169:169] vs inc[169:170], meaningful=`True`
   - noninc: ``
   - inc: ` then`
7. `replace` noninc[170:172] vs inc[171:172], meaningful=`True`
   - noninc: `[STREAMING_WORD] then`
   - inc: `[STREAMING_PAD]`
8. `insert` noninc[204:204] vs inc[204:206], meaningful=`True`
   - noninc: ``
   - inc: `[STREAMING_WORD] Use`
9. `delete` noninc[206:207] vs inc[208:208], meaningful=`True`
   - noninc: ` Use`
   - inc: ``
10. `delete` noninc[208:209] vs inc[209:209], meaningful=`True`
   - noninc: `[STREAMING_PAD]`
   - inc: ``

## Focus Pairwise
- idx 48: token_equal=True noninc=32 inc=32 audio_diff_max_abs=0.250000
- idx 49: token_equal=True noninc=32 inc=32 audio_diff_max_abs=0.225792
- idx 50: token_equal=False noninc=32 inc=33 audio_diff_max_abs=0.124146
- idx 51: token_equal=False noninc=33 inc=4493 audio_diff_max_abs=0.125000
- idx 52: token_equal=False noninc=4493 inc=32 audio_diff_max_abs=0.125000
