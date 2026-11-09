# SEA-RAFT, vendored

Source: https://github.com/princeton-vl/SEA-RAFT @ `main`, cloned 2026-09.
Licence: BSD-3-Clause, see `LICENSE`. Copyright (c) 2024, Princeton Vision &
Learning Lab.

> Yang, Jiang, Deng. *SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical
> Flow.* ECCV 2024.

**Files are verbatim upstream**, so a future update is a re-copy rather than a
merge. Nothing here is edited -- the adaptation lives in
`motion_estimator/methods/searaft/`.

## What was taken, and what was left

Taken: `core/{raft,corr,extractor,layer,update}.py`, `core/utils/utils.py`,
`config/parser.py`, `config/eval/*.json`.

Left out: `core/datasets.py`, `core/loss.py`, `core/utils/augmentor.py`,
`core/utils/flow_transforms.py`, `core/utils/frame_utils.py`, and the
top-level scripts. Those exist to train on FlyingChairs / Sintel / KITTI and
pull in dataset dependencies this project does not have. Training here is on
the simulated cohort through
`motion_estimator/methods/searaft/training.py`, against sparse scatterer
ground truth, which is a different loss to upstream's.

## The import quirk

`core/raft.py` uses flat imports (`from corr import CorrBlock`) because
upstream runs it with `sys.path.append('core')`. Rather than rewrite those --
which would make every upstream update a merge conflict -- the wrapper inserts
this directory on `sys.path` once, inside
`motion_estimator/methods/searaft/net.py`. One contained hack, in one place.

## Weights

`RAFT.from_pretrained("MemorySlices/<name>")` via `huggingface_hub`; the repo
is public and BSD-3, no token needed. The variant used here is
`Tartan-C-T-TSKH-spring540x960-M`, paired with `config/eval/spring-M.json`.
Note that config sets `"scale": -1`, so the model halves the input before
inference and rescales the flow afterwards -- not an optimisation we chose, and
worth remembering when reasoning about effective resolution on 500x500 frames.
