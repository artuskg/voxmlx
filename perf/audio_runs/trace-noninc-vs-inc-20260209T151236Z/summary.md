# Non-incremental vs Incremental Trace Comparison

- audio: `perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav`
- clip_seconds: `120`
- stft_backend: `dft`
- non_incremental elapsed: `52.307s`
- incremental elapsed: `91.503s`
- token distance: `384` (norm `0.254305`)
- char distance (KEEP): `3369` (norm `0.180498`)
- first token divergence index: `50`
- first meaningful divergence index: `50`
- meaningful span count: `64`

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
