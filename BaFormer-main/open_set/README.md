# BaFormer open-set protocol construction

This directory contains only the new open-set protocol scripts.  The original
BaFormer files are not modified by the construction step.

The protocol follows Section IV-A and IV-B of *Learning Action Distribution
Flow for Open-Set Temporal Action Segmentation* (LADF).  Unknown means an
entire action class held out from training and encountered in test videos.  It
does not mean a frame missing from the ground truth, a generic background
region, or an action that is merely unrelated to the current activity.

For GTEA, the paper's nested settings are:

| unknown percentage | held-out actions |
| ---: | --- |
| 10% | `take` |
| 20% | `take`, `open` |
| 30% | `take`, `open`, `put` |
| 40% | `take`, `open`, `put`, `pour` |

`background` stays known at every level.  Original transition labels stay
unchanged; the paper follows the dataset's midpoint convention for assigning a
transition to its neighboring known action.  The percentages refer to classes,
not the percentage of frames.

The paper states that only known-class frames are used for network training; it
does not specify in the cited text how held-out frames are presented to the
backbone.  Here they remain in the input sequence with supervision masked out.
This is an implementation choice that preserves temporal alignment, not a
verified reproduction of an author-provided data loader.

Build all four settings from the TAS dataset used by this project:

```bash
./open_set/run_build_gtea_ladf.sh
```

To build again, set `OUTPUT_ROOT` to a new directory. Existing output is never
overwritten. The source dataset is read only.

The default output is `open_set_data/gtea_ladf_v2/pct10` through `pct40`.  Each
setting contains:

* `mapping_known.txt`: contiguous labels used by an open-set training model;
* `mapping_eval.txt`: known labels plus one `unknown` evaluation label;
* `labels/train`: known labels and `<ignore>` at held-out action frames;
* `targets/train`: known class ids and `-100` for held-out action frames;
* `supervision_masks/train`: one for supervised frames and zero for held-out frames;
* `labels/test` and `targets/test`: held-out actions collapsed to `unknown`;
* `features`: a relative symlink to the source feature directory;
* `metadata.json` and `manifest.json`: protocol and alignment checks.

This construction layer intentionally does not choose a rejection score or
modify BaFormer losses.  A later BaFormer wrapper can consume the generated
training targets/masks and evaluate the test targets under the same protocol.
The existing `main.py` cannot consume `-100` targets directly: its target
preparation, matching, mask loss, boundary loss, and metrics need to handle the
supervision mask. Its GTEA segment metrics hard-code background id 10, while
the generated contiguous mapping moves that id. Neither the `unknown` label nor
the masked target is BaFormer's unmatched-query `no-object` class.
