# Frozen execution log

All commands were run from `C:\T1D Meta Learning\code\python\meta-diffusion` with
`FITZPATRICK_ROOT=C:\T1D Meta Learning\dataset\fitzpatrick17k_verified` and
`PYTHONUTF8=1`. Debug runs are explicitly named `debug` and are not analysed.

## P1

- Built manifests with `scripts/build_fitz_family_folds.py` from
  `configs/p1_family.json`; validated each with `scripts/check_fitz_split.py`.
- Trained `p1_family_fold{0,1,2}_basis` using `runner.train`, 20,000 steps,
  `--scheme fitzpatrick --seed 12345 --m-source 32 --k 16 --base-channels 32
  --transport-kind linear --device cuda`.
- Validation-only selection (`scripts/fitz_select.py`) chose fold checkpoints
  step 16,000, 10,000 and 10,000.
- Evaluated each selected checkpoint twice with `scripts/c4_scarcity.py`: zero-shot and
  `--k-shots 1 2 5 10 20`, in both cases `--label-field nine_partition_label
  --headline-no-refine --instrument-split-only --repeats 8 --n-samples 96
  --ddim-steps 50 --seed 4321`.
- Aggregated using `scripts/aggregate_p1_family.py`.

## P2 and P3

- P2 used `checkpoints/fitz_full_basis_m32_k16_step11000.pt`, the original clean disease
  split, `configs/p2_wrong_source_maps.json`, and the same zero/few-shot sampling options as
  P1 except the semantic label remained disease `label`.
- P2 was aggregated with `scripts/aggregate_p2_hierarchical.py`.
- P3 used `scripts/p3_fitz_separability.py`; official torchvision ResNet-18
  `IMAGENET1K_V1` weights, seed 4321, and source images only.

## P4

- Built and validated five V--VI manifests from `configs/p4_phenotype_v_vi.json` before
  viewing P1 results. P4 was triggered only after P1 was classified inconclusive.
- A 20-step `p4_v_vi_debug_fold0` run verified the data/model path and is excluded.
- Trained `p4_v_vi_fold{0,1,2,3,4}_basis` for 20,000 steps with the same frozen
  architecture/options as P1 and source I--III, target V--VI.
- Validation-only selection chose steps 16,000, 4,000, 4,000, 4,000 and 10,000.
- Evaluated every selected checkpoint with `scripts/c4_scarcity.py`, first `--zero-shot`
  and then `--k-shots 1 2 5 10 20`, using `--label-field nine_partition_label
  --headline-no-refine --instrument-split-only --repeats 8 --n-samples 96
  --ddim-steps 50 --seed 4321`.
- Aggregated five cyclic outer folds with `scripts/aggregate_p1_family.py --prefix p4_v_vi
  --folds 5`; each family's two outer appearances were averaged before the family-level CI.

## Independent P5 audit

P5 was not rerun. The inspected saved artifacts and conclusions are enumerated in
`reports/p5_cifar_completion.md`.
