# Experimental Soundness Analysis: MiniGrid BoT Sweep

## 1. Technical Soundness (High)
The underlying infrastructure of the experiment is robust and scientifically valid:
- **Seeding**: The use of `seed=42+i` for each episode ensures perfectly matched comparison. For any given episode ID, the 3B and 7B models face the exact same environment state.
- **Metric Isolation**: Token counts and latency are calculated as per-episode deltas, preventing accumulation errors.
- **Environment Consistency**: Hardware (NVIDIA A100) and environment builds are identical across model runs.

## 2. Experimental Design Constraints (Moderate)
While technically sound, some design choices limit the "strength" of the conclusions:
- **Statistical Significance**: $N=5$ episodes per condition is low for Reinforcement Learning tasks. A single "lucky" seed can swing results by 20%. 
    - *Recommendation*: Increase to $N=20+$ for publication-grade results.
- **Step Limit (50)**: This is quite restrictive for "exploration" tasks. Many failures in `DoorKey` and `BlockedUnlockPickup` are likely due to "wandering" rather than logical incapacity.
    - *Recommendation*: Increase to 100-200 steps for complex environments.

## 3. Findings on Model Behavior
- **Looping (The "4-Turn Trap")**: In the 3B model (and 7B without history), agents frequently get stuck in 4-action cycles: `turn_left` -> `turn_left` -> `turn_left` -> `turn_left`. 
- **History Window ($k$) Effectiveness**: Increasing $k$ from 1 to 3 significantly broke these loops in the 7B model, leading to its success in `LavaGap`. In the 3B model, history was less effective, suggesting a base reasoning floor.
- **Prompting Strategy**: The current history format (a raw list of observations) is very long. As the context fills up, the model might "forget" the reasoning aid or the goal.

## 4. Final Verdict: **SOUND**
The experiments are logically valid for a **pilot sweep**. The conclusions about 7B > 3B and History Window > Buffer Size are supported by the data, though the absolute success rates are still influenced by the low $N$ and strict step limits.
