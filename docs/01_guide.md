# Technical Guide — pp_cotrain

A self-contained walkthrough of **what was built**, **why**, and **how it works**, with specific line references and numbers from the actual 2M-step run.

---

## 1. The problem

Adversarial multi-agent RL in an **asymmetric** setting: 3 predators vs 1 prey. Unlike zero-sum symmetric games (Chess, Go), self-play is not applicable — the two teams have different state spaces, different action dynamics, and different reward functions. Each team's optimal policy depends on the opponent's policy, which is *also* changing. This is the non-stationarity that our paper targets with **model-based co-training**: each team learns a model of the *opponent's reward/objective* and feeds it to a planner.

This repo is the **model-free co-training baseline** the proposed method will be compared against.

---

## 2. The environment — MPE `simple_tag_v3`

| field | value | source |
|---|---|---|
| agents | `adversary_0/1/2` + `agent_0` | `JaxMARL/jaxmarl/environments/mpe/simple_tag.py:28-30` |
| predator radius / accel / max-speed | 0.075 / 3.0 / 1.0 | `simple_tag.py:51,58,64` |
| prey radius / accel / max-speed | 0.05 / 4.0 / 1.3 | `simple_tag.py:52,59,65` |
| landmarks (obstacles) | 2, radius 0.2 | `simple_tag.py:53,66` |
| obs dim | 16 (adversary), 14 (prey) | `simple_tag.py:36,39` |
| action space | Discrete(5): {no-op, N, S, E, W} | default `action_type=DISCRETE_ACT` |
| predator reward | `+10 × #collisions_with_prey` per predator | `simple_tag.py:163` |
| prey reward | `-10 × #collisions − boundary_penalty` | `simple_tag.py:157-160` |
| episode length | 25 steps (default MPE) | `mpe/default_params.py` |

Key asymmetries:
- **Speed**: prey is 30% faster (1.3 vs 1.0). Predators must coordinate to corner it.
- **Rewards**: each predator independently earns +10 per collision (so a single capture yields +30 to the team). Prey pays −10 for being caught, plus a smooth `map_bounds_reward` penalty for leaving the [-1, 1]² arena.
- **Observation**: different dimensionalities; the `CTRolloutManager` wrapper pads obs to the max (16) and appends a 4-dim agent-id one-hot, yielding a uniform **20-dim observation** for every network forward.

---

## 3. Why the stock JaxMARL IQL is wrong for this task

`JaxMARL/baselines/QLearning/iql_rnn.py` instantiates **one** `RNNQNetwork` and **one** `TrainState`, then `jax.vmap`s it across the agent dimension with `in_axes=(None, 0, 0, 0)` (see `iql_rnn.py:263-264, 356`). `None` on `params` means **the same parameters are shared across every agent, both teams**.

In a cooperative task (e.g. `simple_spread`) that's fine — all agents minimize the same loss. In an adversarial task:

- predator transitions contribute gradients that **increase** the Q of `chase_prey` actions,
- prey transitions contribute gradients that **decrease** the Q of `be_chased` actions,
- they share the same network → **directly conflicting gradients on the same parameters**.

You end up training a confused Q-function that predicts neither team's return well. The first thing I did was read that script carefully to confirm this — not a bug, just the wrong architecture for our setting.

---

## 4. What `src/iql_teams.py` does differently

Two independent IQL learners, trained simultaneously:

| component | per-team | notes |
|---|---|---|
| `RNNQNetwork` | 1 each (`pred`, `prey`) | shared params **within** a team (the 3 predators share one net); `src/iql_teams.py:161-168` |
| `CustomTrainState` | 1 each | each carries its own params + target + optimizer state; `src/iql_teams.py:186-194` |
| flashbax trajectory buffer | 1 shared | buffer stores all-agent trajectories; teams split the minibatch by agent name at learn time; `src/iql_teams.py:206-216` |
| target network | 1 each, hard update every 200 updates (`TAU=1.0`) | `src/iql_teams.py:312-326` |
| ε-greedy exploration | shared schedule, independent action sampling per team | `src/iql_teams.py:218-238` |
| loss | team-local: `(r_team + γ (1−d) Q_target_team) − Q_team)²` | `src/iql_teams.py:255-290` |

Team split is by agent-name prefix (`src/iql_teams.py:99-105`): `adversary_*` → `pred`, `agent_*` → `prey`. So the 3 predators share a single `RNNQNetwork`, vmapped across the team dimension, while the lone prey has its own network.

### Why share params within team but not across teams?
- **Within** team: predators are interchangeable — same role, same reward, same action space. Parameter sharing cuts sample complexity by 3× and has a long track record in cooperative MARL.
- **Across** teams: predator and prey have different roles, different rewards, and (logically) different Q-functions. Sharing here is what breaks stock IQL.

### The data flow at one update
```
rollout (26 steps × 8 envs)
    → Timestep {obs, actions, rewards, dones, avail_actions}  (keys = agent names)
    → shared flashbax buffer
      (sample: 32 trajectories × 1 time-chunk)
    → split minibatch by team
      → pred: stack 3 adversary agents → (3, time, batch, 20)
      → prey: stack 1 good agent     → (1, time, batch, 20)
    → two independent TD losses, two independent .apply_gradients()
    → hard target update every 200 updates
```

---

## 5. Hyperparameters — what they do and why

All in `configs/alg/ql_teams_simple_tag.yaml`. The numbers match the JaxMARL MPE defaults (`JaxMARL/baselines/QLearning/config/alg/ql_rnn_mpe.yaml`) except where noted.

| key | value | meaning |
|---|---|---|
| `TOTAL_TIMESTEPS` | 2 000 000 | env steps |
| `NUM_ENVS` | 8 | parallel envs (vectorized) |
| `NUM_STEPS` | 26 | steps per rollout; one update consumes `26×8=208` env transitions |
| `NUM_UPDATES` | 9615 | derived: `TOTAL_TIMESTEPS / NUM_STEPS / NUM_ENVS` |
| `BUFFER_SIZE` | 5000 | trajectories in flashbax buffer |
| `BUFFER_BATCH_SIZE` | 32 | sampled per learn step |
| `HIDDEN_SIZE` | 64 | GRU hidden + embedding |
| `EPS_START / EPS_FINISH / EPS_DECAY` | 1.0 → 0.05 over 10% of updates (≈962 updates) | exploration schedule |
| `LR` | 0.005 | with linear decay to 1e-10 (`LR_LINEAR_DECAY=True`) |
| `GAMMA` | 0.9 | short horizon — MPE eps are 25 steps, 0.9^25 ≈ 0.07 |
| `MAX_GRAD_NORM` | 25 | clip |
| `TARGET_UPDATE_INTERVAL` | 200 | every 200 updates |
| `TAU` | 1.0 | hard target update (`τ=1` means copy, not Polyak) |
| `LEARNING_STARTS` | 10 000 | timesteps of random exploration before learning |
| `TEST_NUM_ENVS` | 128 | greedy eval envs |

---

## 6. What the results show (specific numbers from the run)

All from `logs/MPE_simple_tag_v3/iql_teams_MPE_simple_tag_v3_seed0_metrics.npz`.

### Test-time greedy returns (TEST_NUM_ENVS=128, TEST_NUM_STEPS=30)
| team | first eval | last eval |
|---|---|---|
| `pred` | +14.375 | **+33.125** |
| `prey` | −247.46 | **−49.05** |

A capture event yields +10 per predator = +30 total. Last-eval pred return of +33 per 30-step rollout ≈ **1.1 captures per rollout** (once exploration is off). Prey's −49 is the sum of −10 per capture and the boundary penalty; the massive improvement from −247 → −49 is the prey learning to **stop running into walls**.

### Training curves (`plots/train_curves.png`)
- Updates 0–2000: predators discover chasing → pred return peaks at **~+5**, prey troughs at **~−8**. This is the "predators figured it out first" phase.
- Updates 2000–4000: prey catches up, starts evading. Pred return drops from +5 to ~+1.4, prey recovers from −8 to ~−2.
- Updates 4000–9615: both settle into a noisy equilibrium around ±2. **This is the co-training signature you want to show.**

### Q-values (`plots/loss_q.png`)
Converge to **+5 for predator, −5 for prey** — symmetric around zero. With `GAMMA=0.9` and ~1 capture per 25-step episode, Q values of ±5 are consistent with a discounted expected return of about `γ^n_steps_to_capture × 30 ≈ 5` for predators.

### TD loss
Both teams: early spike to ~10 during the policy churn phase (updates 1500–3000), then monotonic decline to ~4. No divergence, no collapse. Clean Bellman regression.

### Rollout (`plots/rollout.gif`, seed 7)
60 greedy steps. Average **predator team reward 2.5 / step** = 150 total over 60 steps = **5 capture events**, one every ~12 steps. Prey reward is **−0.68 / step** (−10/capture plus boundary).

### Wall-clock
**150 seconds** for 2M timesteps on an Apple M4 Pro CPU via JAX. ≈ 13 300 env-steps/second. Roughly 624 updates/second, 1300 gradient steps/second (2 team losses per update).

---

## 7. Gotchas I hit

- **Flax ↔ JAX version mismatch.** JaxMARL pins `jax<=0.4.38` but leaves `flax` unpinned. `pip install` pulls flax 0.10.4, which imports `jax.api_util.debug_info` (doesn't exist until jax 0.5). Fix: `pip install "flax==0.10.2"`. See `README.md`.
- **int32 / float32 rewards.** `simple_tag.py` computes adversary rewards as `10 * jnp.sum(collisions)` — int scalar × int-cast bool sum → int32. The prey reward is int + float (boundary penalty) → float32. If you set `REW_SCALE=1.0` (float) the multiplication promotes adversary rewards to float32; the flashbax buffer was initialized from a random-policy sample trajectory (no REW_SCALE), so its stored dtype was int32 for adversaries. On the first real `.add` after learning starts, chex fails with `types: float32 != int32`. Fix: cast rewards to float32 explicitly in both the sample path (`src/iql_teams.py:201`) and the real rollout (`src/iql_teams.py:269-272`).

---

## 8. How this sets up the paper

- **Model-free, independent-Q co-training** is the baseline curve. Our proposed method — BIRL-based opponent-reward estimation feeding an MCTS planner — should beat this on:
  1. **Sample efficiency**: IQL here needs 2M steps. A planner with a learned opponent-reward model should match performance in far fewer interactions.
  2. **Transfer**: if the map changes (e.g. obstacles move), stock IQL needs to retrain. A planner using a learned reward model + known physics should adapt immediately.
  3. **Non-stationarity handling**: IQL implicitly assumes a stationary opponent; our method explicitly models the opponent's reward as something that *changes*.
- What this baseline does **not** try to do: opponent modeling, partial observability handling, or explicit exploration beyond ε-greedy.
