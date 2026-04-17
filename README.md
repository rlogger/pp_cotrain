# pp_cotrain — Predator-Prey Co-Training with and without Opponent Modeling

This repository implements two-team independent Q-learning on the MPE `simple_tag_v3` environment (three predators versus one prey), along with an A/B comparison against an opponent-aware variant (OA-IQL) that introduces an auxiliary opponent-action prediction head.

---
![A/B test returns](plots/compare_test_returns.png)

OA-IQL does not uniformly outperform baseline IQL. Rather, it shifts the equilibrium toward the prey.

| Metric | Baseline | OA-IQL | Δ |
|---|---|---|---|
| Final predator greedy return (30-step episode) | **+33.12** | **+24.69** | **−8.44** |
| Final prey greedy return (30-step episode) | **−49.05** | **−37.78** | **+11.27** |
| Predator peak return (training) | ~+103 @ update 2000 | **~+144 @ update 2000** | +41 |
| Opponent-action accuracy asymptote | — | ~0.57 (both teams) | — |
| Wall-clock per run | 150 s | 158 s | +5 % |

During early training, the OA predator peaks higher, indicating improved early sample efficiency. In later stages, the prey closes the gap more rapidly because its auxiliary task — predicting three predators over five actions — carries greater informational content than the predator's task of predicting a single prey over five actions. In a zero-sum-like game, the agent whose opponent model is more informative pulls the equilibrium toward itself.

---

## Repository Layout

```
src/
  iql_teams.py            # baseline two-team IQL
  iql_teams_oa.py         # opponent-aware IQL (auxiliary opponent-action head)
  compare_plots.py        # A/B plots from two metrics npz files
  plot_metrics.py         # per-run training curves, loss and Q plots
  visualize_rollout.py    # greedy rollout GIF from baseline parameters
  visualize_rollout_oa.py # greedy rollout GIF from OA parameters

configs/
  config.yaml                       # top-level Hydra config
  alg/
    ql_teams_simple_tag.yaml        # baseline hyperparameters
    ql_teams_oa_simple_tag.yaml     # OA-IQL hyperparameters (adds OPP_AUX_COEF)

docs/
  01_guide.md       # technical walkthrough
  02_demo.md        # 10-minute talk script
  03_qa.md          # anticipated Q&A
  04_quiz.md        # self-check quiz
  05_cheatsheet.md  # numbers, glossary, and code-reference table

plots/
  compare_test_returns.png    # A/B, principal result plot
  compare_opp_modeling.png    # OA auxiliary-head accuracy and cross-entropy
  compare_summary.png         # final-return bars
  train_curves.png            # baseline per-run curves
  loss_q.png                  # baseline TD loss and mean Q
  rollout.gif                 # baseline seed-7 greedy (5 captures / 60 steps)
  rollout_oa.gif              # OA-IQL seed-7 greedy (2 captures / 60 steps)

logs/
  MPE_simple_tag_v3/
    iql_teams_*              # baseline parameters and metrics
    iql_teams_oa_*           # OA-IQL parameters and metrics
```

---

## Quickstart

```bash
# 1. Environment
conda create -n pp_cotrain python=3.11 -y
conda activate pp_cotrain
pip install -e JaxMARL/
pip install "flax==0.10.2"         # JaxMARL pins jax<=0.4.38; the default flax is incompatible
pip install hydra-core flashbax wandb matplotlib

# 2. Train both variants (approximately 5 minutes total on an M4 Pro CPU)
python src/iql_teams.py
python src/iql_teams_oa.py alg=ql_teams_oa_simple_tag

# 3. Compare
python src/compare_plots.py \
  --baseline logs/MPE_simple_tag_v3/iql_teams_MPE_simple_tag_v3_seed0_metrics.npz \
  --oa       logs/MPE_simple_tag_v3/iql_teams_oa_MPE_simple_tag_v3_seed0_metrics.npz

# 4. Rollout GIFs
python src/visualize_rollout.py \
  --pred_params logs/MPE_simple_tag_v3/iql_teams_MPE_simple_tag_v3_pred_seed0_vmap0.safetensors \
  --prey_params logs/MPE_simple_tag_v3/iql_teams_MPE_simple_tag_v3_prey_seed0_vmap0.safetensors \
  --seed 7 --steps 60 --out plots/rollout.gif

python src/visualize_rollout_oa.py \
  --pred_params logs/MPE_simple_tag_v3/iql_teams_oa_MPE_simple_tag_v3_pred_seed0_vmap0.safetensors \
  --prey_params logs/MPE_simple_tag_v3/iql_teams_oa_MPE_simple_tag_v3_prey_seed0_vmap0.safetensors \
  --seed 7 --steps 60 --out plots/rollout_oa.gif
```

---

## OA-IQL Architecture

```
             obs ──► Dense(64) ──► ReLU ──► GRU(64) ──┬──► Dense(action_dim) ──► Q-values (the policy)
                                                      │
                                                      └──► Dense(opp_n × action_dim) ──► opponent-action logits
                                                            │
                                                            └─ CE versus opponent actions in replay buffer
                                                                (shapes the trunk; discarded at evaluation)
```

Per-team loss:

```
L = L_Q  +  OPP_AUX_COEF · CE(opp_action_pred, opp_action_true)
```

`OPP_AUX_COEF = 0.5` in the configuration. The Bellman target, target-network update, optimizer, replay buffer, and evaluation loop are identical to those of the baseline. Only the network carries a second head and the loss carries a second term.

---

## Three-Phase A/B Analysis

![opponent modeling](plots/compare_opp_modeling.png)

1. **Updates 0–2000 — onset of co-adaptation.** Both variants track one another. Prey return climbs from −247 to approximately −50 (boundary-penalty learning). Predator return climbs from +15 to approximately +50.
2. **Updates ~2000 — peak divergence.** The OA-IQL predator peaks at **+144**; the baseline peaks at +103. The opponent-aware trunk enables faster credit assignment early in training, because the opponent cross-entropy supplies a dense, stable gradient that complements the sparse Bellman backups.
3. **Updates 4000–9615 — equilibrium shift.** The prey likewise benefits from OA (its auxiliary task is richer) and recovers more quickly. By the end of training, **OA-prey exceeds baseline-prey by 11 points, while OA-predator falls 8 points below baseline-predator.**

![final returns](plots/compare_summary.png)

Co-training is zero-sum-like: a method that aids both sides may still disadvantage one side if it aids the other to a greater degree.

---

## Scope and Caveats

- OA-IQL is a representation-learning auxiliary, not a planner. It involves no tree search, no MCTS, and no Bayesian posterior — only DRON-style opponent-action prediction at a weight of 0.5.
- The opponent head is active only during training. Inference cost and policy-network size at test time are identical to the baseline.
- A single seed is used per variant. Multi-seed execution is available via a flag (`NUM_SEEDS: 5` via `jax.vmap`) at a budget of approximately 12 minutes.
- `OPP_AUX_COEF = 0.5` is a reasonable default drawn from the auxiliary-loss literature and has not been tuned.
- The results are nuanced rather than a uniform win, which is the principal point of interest.

---

## Future Work (Engineering, not Research)

- **Asymmetric auxiliary weights.** Setting `OPP_AUX_COEF_PRED = 0.5, OPP_AUX_COEF_PREY = 0.0` should yield the predator benefit without the offsetting prey gain. This requires a single flag change.
- **Auxiliary-weight sweep** across {0.1, 0.25, 0.5, 1.0, 2.0}, at a total cost of approximately 12 minutes.
- **Opponent head as a one-step planner.** At action selection, sample k opponent actions from the head, evaluate Q under each, and select the action maximizing either the robust minimum or the expectation. This incurs no additional training cost.
- **Substitute the auxiliary task with opponent-reward prediction.** This is more policy-invariant (approaching BIRL), at the cost of a harder supervised signal but greater stability under non-stationarity.

---

## Further Documentation

- [`docs/01_guide.md`](docs/01_guide.md) — technical walkthrough, line references, and the dtype bug
- [`docs/02_demo.md`](docs/02_demo.md) — 10-minute talk script with exact slides and pull-ups
- [`docs/03_qa.md`](docs/03_qa.md) — 17 anticipated questions with concise and detailed answers
- [`docs/04_quiz.md`](docs/04_quiz.md) — self-check quiz across six sections, with answers appended
- [`docs/05_cheatsheet.md`](docs/05_cheatsheet.md) — numbers, glossary, and code-reference table
