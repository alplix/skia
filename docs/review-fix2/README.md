# Finding 2 result files

`paired.jsonl` is the unchanged output of 200 fits; its `truth` field is the optimizer
vector (source amplitude/radius are natural-log coordinates). `paired-physical.jsonl`
adds an explicit copy of that original vector under `truth_packed_for_optimizer` and
replaces `truth` with correctly unpacked physical values. The fit outputs are
identical. The summary was recomputed from the physical-truth rows without rerunning
or changing any fit. `REPORT.md` and `summary.json` use the normalized values.

The preserved `paired.jsonl` contains all original per-lens outcomes. See
`../../P1_FIX2_PROTOCOL.md` for the predeclared population and assumptions.
