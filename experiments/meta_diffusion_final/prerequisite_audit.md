# P0 prerequisite and reproducibility audit

Status: **P0 passed for P1**; all three locked manifests pass the structural, on-disk,
population-side, taxonomy, duplicate and capacity checks. Semantic-instrument validation is
an outcome validity gate, not a data prerequisite. No new family-level Meta-Diffusion result
had been inspected when the protocol lock was written.

## Repository and environment

- Branch / HEAD: `jinjian` / `6de5187426b4ebfb4545846a49ca4d30f02987cb`.
- Working tree: dirty before this plan (133 paths); existing changes are preserved.
- OS: Windows 10 build 19045; Python 3.12.13.
- PyTorch 2.11.0+cu128; CUDA 12.8; NumPy 2.5.2; Pillow 12.3.0.
- GPU: NVIDIA GeForce RTX 5080, 16,303 MiB; driver 595.95. The GPU was idle at audit.
- No Python process was visible through the permitted process inventory at audit time.

## Dataset integrity and schema

- Root: `C:\T1D Meta Learning\dataset\fitzpatrick17k_verified`.
- Official CSV SHA-256: `851DFAABF1791C2A8B1C2957619F3E0673CF0002F4CE924A5C02D185C7D4A1CD`.
- Verification manifest SHA-256: `4AA4FC1C4B7F1888E06F132B4A621EC8086076C32CF234840C2870D583458DF3`.
- 16,577 official rows and 16,577 manifest rows; 11,394 are byte-verified and retained by
  the Fitzpatrick17k-C allow-list; 11,098 retained rows have Fitzpatrick type I--VI.
- `nine_partition_label` exists with exactly nine non-missing values. Disease `label` and
  phenotype labels are also present. No subject/patient identifier exists in the supplied
  schema, so patient-level leakage cannot be checked directly. Image hashes and the supplied
  near-duplicate annotations are used instead.
- Pre-dedup family counts are recorded below; the split manifests record post-dedup counts.

| Family | Source I--III | Target IV--VI | Target V--VI | Full P1 eligible |
|---|---:|---:|---:|---|
| inflammatory | 5,344 | 1,979 | 854 | yes |
| genodermatoses | 419 | 270 | 143 | yes |
| malignant epidermal | 786 | 188 | 61 | yes |
| benign epidermal | 411 | 164 | 58 | yes |
| benign dermal | 600 | 132 | 49 | yes |
| malignant melanoma | 332 | 84 | 38 | yes for IV--VI only |
| malignant dermal | 53 | 43 | 17 | no |
| benign melanocyte | 118 | 38 | 18 | no |
| malignant cutaneous lymphoma | 99 | 38 | 14 | no |

After collapsing known near-duplicate components, P1 retains 10,539 images. Its six eligible
families have source/target counts: benign dermal 573/122, benign epidermal 393/151,
genodermatoses 402/240, inflammatory 5,118/1,828, malignant epidermal 766/176, and malignant
melanoma 321/78. Each P1 fold allocates 10,168 distinct images and has passed
`scripts/check_fitz_split.py`. The shared image-set checksum is
`cda411e7069ffc427c74b655a28bd54825a1f8e72f199132ea58177197d9a8e5`; the complete
allocation checksums are stored inside each manifest.

The pre-outcome P4 feasibility check also passed. For target V--VI, five families remain
eligible after duplicate removal: benign dermal 574/47, benign epidermal 395/56,
genodermatoses 404/129, inflammatory 5,144/798, and malignant epidermal 767/59. Five cyclic
manifests were frozen before P1 results, each with 8,373 distinct images and explicit
phototype sets `[1,2,3]` to `[5,6]`; all five pass the same split checker.

## Frozen assets and prior evidence

- Existing disease split SHA-256:
  `786510799E57BC3FB99F4A57341B1419E0DAACF35982855834A21B7218B32974`.
- Frozen disease additive checkpoint SHA-256:
  `8DB503683DA6496A28788CF50B46CBE816007ECF709931ABCA04B2354686C776`.
- Matched disease FiLM checkpoint SHA-256:
  `D9765A8B385D90748D5D002DE177A326456E86812249A3E13C5DDC1F46F6F425`.
- Validation-only selection records exist for the disease checkpoints. Saved evidence also
  exists for CIFAR source controls, K_T=0, FiLM source dependence, linear transport, full
  fine-tuning, Fitzpatrick task scarcity/compression, and CIFAR 50% and 25% task scarcity.
- The CIFAR 25% run finished on 2026-09-21 and has raw per-episode files plus paired results;
  the older report statement that it was ongoing is stale.

## P0 limitations

- There are only six families supporting the full P1 grid, hence task-level cross-validation
  is required and uncertainty will still be based on only six outer-test tasks.
- The family semantic evaluator has not yet passed its locked validity threshold. Failure
  produces an inconclusive semantic gate, not a substituted metric.
- Patient identity is unavailable. Hash verification, Fitzpatrick17k-C filtering and known
  near-duplicate removal are the strongest available leakage controls.
- There are no prior family-level checkpoints; P1 requires new frozen-recipe training.
- Official torchvision ResNet-18 `IMAGENET1K_V1` weights were cached before the P3 outcome
  analysis; P3 is therefore technically unblocked.
