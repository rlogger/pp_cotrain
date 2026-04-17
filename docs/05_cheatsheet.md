# Cheat sheet

Keep on a second monitor during the demo.

---

## Numbers never to get wrong

| metric | baseline IQL | OA-IQL |
|---|---|---|
| training time | **150 s** | **158 s** |
| throughput | ~13 300 env-steps/s | ~12 650 env-steps/s |
| total env steps | 2 000 000 | 2 000 000 |
| update steps | 9 615 | 9 615 |
| final pred greedy return (30-step ep) | **+33.12** | **+24.69** |
| final prey greedy return (30-step ep) | **−49.05** | **−37.78** |
| pred return peak (during training) | ~+103 @ update 2000 | **~+144 @ update 2000** |
| prey return start | −247.46 | −248.33 |
| opp-action accuracy asymptote (pred / prey) | — | **~0.57 / ~0.57** |
| random-guess accuracy | 0.20 | 0.20 |

### A/B delta summary

| team | Δ (OA − baseline) |
|---|---|
| pred | **−8.44** |
| prey | **+11.27** |
| sum | **+2.83** |

---

## One-paragraph pitch

> "Two-team independent Q-learning on MPE simple_tag, with an A/B comparison against a variant that adds an auxiliary opponent-action prediction head to each team's Q-network. The aux head predicts the opponent team's next action from the current observation, trained with cross-entropy loss at 0.5× weight. The shared trunk is forced to encode opponent-aware features; the Q-head reads off a richer representation without any change to the Bellman target. Result after 2 M env-steps: OA-IQL shifts the equilibrium toward prey — pred loses 8 return points, prey gains 11. Prey's aux task (predicting 3 predators) is richer than pred's (predicting 1 prey), so prey benefits more from opponent modeling in a zero-sum-like co-training loop. Early in training, OA pred *peaks* +40 points above baseline pred before prey catches up — sample-efficiency lift is real but asymmetric. 150 s per training run on M4 Pro CPU."

---

## Reward arithmetic

- Per-predator per capture: **+10**
- Predator team per capture: **+30** (3 × 10)
- Prey per capture: **−10**, plus smooth `map_bounds_reward` if out of arena
- 30-step eval episode pred return +33 ≈ **1.1 captures**
- 30-step eval episode pred return +25 ≈ **0.8 captures**

---

## Code reference table

| thing | file : line |
|---|---|
| two-team split | `src/iql_teams.py:99-105` |
| per-team train state init (baseline) | `src/iql_teams.py:186-194` |
| per-team learn phase (baseline) | `src/iql_teams.py:254-290` |
| **two-head Q-net (OA-IQL)** | `src/iql_teams_oa.py:71-95` |
| **per-team learn phase (OA-IQL)** | `src/iql_teams_oa.py:241-310` |
| opp-action CE + accuracy | `src/iql_teams_oa.py:269-294` |
| auxiliary loss weight | `configs/alg/ql_teams_oa_simple_tag.yaml:23` |
| stock IQL wrong-shared-net pattern | `JaxMARL/baselines/QLearning/iql_rnn.py:166-171, 263-264` |
| simple_tag pred reward | `JaxMARL/jaxmarl/environments/mpe/simple_tag.py:163` |
| simple_tag prey + boundary reward | `simple_tag.py:157-160` |

---

## MARL / MARL-adjacent terms (1-liner each)

- **IQL** — Independent Q-Learning: per-agent Q with per-agent reward. No shared critic.
- **VDN** — team Q = sum of agent Qs. Cooperative only.
- **QMIX** — team Q = monotonic-mixer(agent Qs, global state). Cooperative only.
- **MAPPO** — Multi-Agent PPO: policy-gradient with shared centralized critic.
- **CTDE** — Centralized Training, Decentralized Execution.
- **IGM** — Individual-Global-Max. argmax(team Q) = joint argmax(individual Qs). Required by VDN/QMIX.
- **Opponent modeling** — any method that explicitly represents the opponent's policy, action distribution, or reward. OA-IQL does the action-distribution version.
- **BIRL** — Bayesian Inverse RL. Posterior over opponent reward hypotheses. Strictly more expensive than OA-IQL.
- **DRON** — Deep Reinforcement Opponent Network (He et al. 2016). Original auxiliary opponent-policy-prediction head for DQN. OA-IQL is DRON adapted to recurrent IQL in JaxMARL.
- **Self-play** — agent trains against a copy of itself. Requires symmetric game. Doesn't apply here (asymmetric teams).
- **Non-stationarity (in MARL)** — each agent's environment is non-stationary because other agents' policies change during training. OA-IQL partly absorbs this into the aux-loss representation.
- **MCTS / dec-MCTS** — tree-based planning with / without centralized tree. Not in this repo but the opp-head could plug into a leaf-value estimator.

---

## If pressed for time — show only these 3 artifacts

1. `plots/compare_test_returns.png` — the A/B evidence (45 s)
2. `plots/compare_opp_modeling.png` — the auxiliary head is learning (20 s)
3. `plots/rollout_oa.gif` — the behaviour (20 s)

Everything else is Q&A material.

---

## Common audience traps

- "So OA wins?" → "Not uniformly. It shifts equilibrium toward prey."
- "Why not use the opp head to plan?" → "Next step. Current scope is the representation-learning aux only."
- "Is this Bayesian?" → "No. Point-estimate classifier. Bayesian version would need BIRL."
- "Why 0.5?" → "Untuned. Sensible default from aux-loss literature. 5-second sweep is trivial."
- "Variance?" → "Single seed. Multi-seed is a flag; ~12 min for 5-seed."
