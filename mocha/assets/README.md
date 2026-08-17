# Third-party assets — obtain these three files yourself

The system uses one trained parameter file, `weights/head_jm_zscore_focal.npz` (14 KB), which
**is** in this repository because we trained it. Everything else it loads is third-party and is
**not redistributed here**. Place the three files below in this directory before running
`verify.py` or the training chain.

All three ship with the challenge organizers' own CARE-PD repository,
<https://github.com/TaatiTeam/CARE-PD>.

| file | md5 | where to get it |
|---|---|---|
| `motionagformer-s-h36m.pth.tr` | `2c0d06b6a60610f080661c46c6f5f231` | CARE-PD repo: `bash scripts/download_models.sh` → `assets/Pretrained_checkpoints/motionagformer/`. Original release: MotionAGFormer (Mehraban *et al.*, WACV 2024), Human3.6M-pretrained, **used frozen and unmodified**. |
| `J_regressor_h36m_correct.npy` | `57176924f79106f64dcafd64d4a1df6d` | CARE-PD repo, tracked at `data/preprocessing/common/J_regressor_h36m_correct.npy`. |
| `smpl_neutral_clean.npz` | `44a55a168cd56fbeaa1d28e9f2ee2389` | Derived from `SMPL_NEUTRAL.pkl` — see below. |

Verify what you obtained with `md5sum` against the table; the whole point of listing the digests is
that a reproduction can prove it is running the same bytes we did.

## `smpl_neutral_clean.npz`

This is the SMPL neutral body model (`v_template`, `shapedirs`, `posedirs`, `J_regressor`,
`kintree_table`, skinning `weights`) stored as a chumpy-free `.npz`, because `numpy` 2.x cannot
build chumpy and the original pickle needs it.

**We do not redistribute it: the SMPL model is licensed by the Max Planck Institute and its
license does not permit redistribution.** Obtain `SMPL_NEUTRAL.pkl` under that license — from
<https://smpl.is.tue.mpg.de/> after registering, or from the CARE-PD repository, which tracks it at
`data/preprocessing/common/body_models/smpl/SMPL_NEUTRAL.pkl` — then convert it with:

```sh
python assets/extract_smpl_npz.py \
    --pkl /path/to/SMPL_NEUTRAL.pkl \
    --out assets/smpl_neutral_clean.npz
```

The converter is deterministic; the resulting file should match the md5 in the table above.

## Why these are not in the repository

Beyond the SMPL license, this follows the convention of the other entries here: pretrained
third-party weights are referenced by name and digest rather than mirrored. It also keeps the
entry at ~135 KB, so cloning it costs nothing.

The archive submitted to CodaBench (submission 882979) is `code/` plus these three files and the
trained head, flattened into one directory and zipped. At 72 MB, and carrying the SMPL model, it is
likewise not published here — `verify.py` assembles exactly that layout in a temporary directory
once the three files are in place, and checks each one's md5 first, so a run of it confirms you are
executing the same bytes the evaluation server did.
