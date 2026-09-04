"""Refinement budget. **Every strategy shares one object**, so the matched
comparison the document requires is enforced structurally rather than by hand.

Validation sweep ranges from A.3 of the document:
    J     in {0, 5, 10, 25, 50, 100}
    eta_z in {1e-3, 3e-3, 1e-2, 3e-2}
    beta_0 in {0, 0.01, 0.1, 1, 10}
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptBudget:
    steps: int = 25              # J
    lr: float = 1e-2             # eta_z
    beta0: float = 1.0           # beta_0, prior pull (further divided by K_T)
    noise_batch: int = 32        # (t, eps) draws per step on the support set

    # Weight learning rate used only by the full fine-tuning comparison. The step
    # count J is **shared** with the coordinate method (matched budget), but
    # applying eta_z = 1e-2 directly to network weights would destroy the model,
    # so this rate is set separately and reported alongside.
    lr_weights: float = 1e-4

    def replace(self, **kw) -> "AdaptBudget":
        from dataclasses import replace
        return replace(self, **kw)
