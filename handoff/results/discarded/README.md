# Discarded results

Kept, not deleted. This project documents overturned readings rather than quietly dropping
them, because the reason a number was wrong is usually worth more than the number.

**Do not quote anything in this directory.**

## `C4_50k_overfit.*` — the first C4 run, 2026-09-18

Run on `c_fitz_step50000.pt`, which took stage A's configuration (128 channels, 50 000 steps)
without checking whether the data funds it. It does not. Over that run:

| | step 2000 | step 50000 |
|---|---|---|
| training target loss, 5 train conditions | 0.0316 | 0.0085 |
| **validation loss, 3 held-out conditions** | **0.0628** | **0.1527** |
| `r_basis` | 0.447 | **0.0024** |

The training side draws from 808 images behind five conditions, so 50 000 steps is about two
thousand epochs. The validation loss on held-out conditions rose by a factor of 2.4 while the
training loss fell by 3.7, and the basis residual ended at 0.24 % of the base prediction —
the task coordinate had been trained out of the model.

**That is the whole explanation of the table.** `no adaptation` (z = 0) beat every other arm,
and `transport` was indistinguishable from `wrong source`, because z = 0 *was* the model:
any nonzero coordinate injected noise into a network that no longer used one. The table
measures a broken checkpoint, not the method.

Replaced by a capacity sweep selecting on the validation conditions
(`../C_model_selection.txt`, `scripts/c_select_table.py`), which chose a 32-channel model.

**A test-split reading was spent here.** Nothing from this run informed the checkpoint
selection — that came from the validation curve — but the test conditions were touched before
the selection was made, and that belongs on the record.
