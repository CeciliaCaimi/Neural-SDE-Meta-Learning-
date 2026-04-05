# Baseline Experiments

Comparing Meta-Learning vs Transfer Learning approaches.

## Baseline 1: Transfer Learning (Warm-Start)

**Theory**: Yosinski et al. (2014) - Good initialization beats random initialization.

**Question**: Does meta-learning's structured task learning beat naive transfer learning?

### Architecture
- Same models as meta-learning (Encoder + SDE + Head)
- BUT: Trains on UNION of all data (no task boundaries)
- Adaptation: Fine-tunes SDE+Head only (Encoder frozen)

### Why This Is a Strong Baseline
1. Uses identical architecture to meta-learning
2. Learns general dynamics from all tasks mixed together
3. Encoder learns to extract generic (not task-specific) representations
4. If meta-learning wins, it proves **Episodic Training > Joint Training**

### Files

| File | Purpose |
|------|---------|
| `train_transfer.py` | Train on union of all data (4 hours) |
| `adapt_transfer.py` | Test adaptation on new tasks (1 hour) |

### How to Run

**Step 1: Train Transfer Model**
```bash
python -m baselines.train_transfer
