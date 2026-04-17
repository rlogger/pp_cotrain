# Cheat sheet — numbers and terms to have at your fingertips

Keep this open on a second monitor during the talk.

---

## The numbers to never get wrong

| question | answer |
|---|---|
| training time | **150 s** on M4 Pro CPU |
| throughput | ~**13 300** env-steps/sec |
| total env steps | 2 000 000 |
| update steps | 9 615 |
| final pred test return | **+33.125** per 30-step ep |
| final prey test return | **−49.05** per 30-step ep |
| starting pred test return | +14.375 |
| starting prey test return | −247.46 |
| pred return peak (train) | ~**+5** around update ~2000 |
| prey return trough (train) | ~**−8** around update ~2000 |
| pred return end (train) | ~+1.4 |
| prey return end (train) | ~−2.0 |
| converged Q (pred / prey) | ~**+5 / −5** |
| seed-7 rollout captures | **5 in 60 steps** (~1 every 12 steps) |

---

## Reward arithmetic — keep this straight

- **Per-predator per capture**: +10
- **Predator team per capture** (sum of 3 agents): +30
- **Prey per capture**: −10 (plus smooth boundary penalty)
- **Per 30-step eval episode**, +33 pred return ≈ 1.1 capture events.

---

## MARL term glossary (1-liner each)

- **IQL** — Independent Q-Learning: each agent has its own Q, trained with its own reward. No shared critic.
- **VDN** — Value Decomposition Network: team Q = sum of agent Qs. Cooperative only.
- **QMIX** — team Q = monotonic-mixer(agent Qs, global state). Cooperative only.
- **MAPPO** — Multi-Agent PPO: policy-gradient with a shared centralized critic.
- **CTDE** — Centralized Training, Decentralized Execution. Access to global info at train time, local at test.
- **IGM** — Individual-Global-Max. The assumption that argmax of team Q = joint argmax of individual Qs. VDN/QMIX require it.
- **Self-play** — agent trains against a copy of itself. Works for symmetric zero-sum games.
- **League training** — keep a population of past policies; train each new policy against a mix. What DeepMind used for AlphaStar.
- **BIRL** — Bayesian Inverse RL. Posterior over reward hypotheses given observed trajectories, assuming the actor is (approximately) optimal under the reward.
- **MCTS** — Monte-Carlo Tree Search. Build a tree of future states; evaluate by simulation or bootstrap.
- **dec-MCTS** — decentralized MCTS, one tree per agent, with coordination via communication / learned teammate models.
- **Non-stationarity (in MARL)** — the environment each agent sees is non-stationary because other agents' policies change during training.

---

## The one-paragraph pitch (memorize this)

> "The project is about model-based co-training for asymmetric adversarial multi-agent RL. Self-play and behavior cloning both fail when the opponent is learning — self-play assumes symmetric games with known rules, BC chases a moving target. Our contribution is to learn a model of the *opponent's reward function* using Bayesian inverse RL on trajectories observed during co-training, and feed that model to an MCTS planner. Before we build that, we need a strong model-free co-training baseline to beat — which is what this repo is: two independent IQL learners on predator-prey, with clear co-adaptation dynamics, 150 seconds to train on a laptop."

---

## Specific code references (if asked to show)

| thing | file : line |
|---|---|
| two-team split | `src/iql_teams.py:99-105` |
| per-team train state init | `src/iql_teams.py:186-194` |
| env step with per-team actions | `src/iql_teams.py:231-248` |
| per-team learn phase | `src/iql_teams.py:254-290` |
| per-team target update | `src/iql_teams.py:312-326` |
| metrics split by team | `src/iql_teams.py:340-358` |
| stock IQL's shared-net wrong pattern | `JaxMARL/baselines/QLearning/iql_rnn.py:166-171, 263-264` |
| simple_tag reward (predator) | `JaxMARL/jaxmarl/environments/mpe/simple_tag.py:163` |
| simple_tag reward (prey + boundary) | `simple_tag.py:157-160` |
| obs preprocessing (pad + ID one-hot) | `JaxMARL/jaxmarl/wrappers/baselines.py:328-335` |

---

## If pressed for time during demo

Show **only these 3 artifacts, in this order**:

1. `plots/train_curves.png` — the co-adaptation story (30 sec)
2. `plots/rollout.gif` — the behavior (20 sec)
3. `plots/loss_q.png` — convergence sanity check (20 sec)

Everything else is Q&A material.
