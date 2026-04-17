# Technical guide — baseline IQL + Opponent-Aware IQL

This repo trains two variants of two-team independent Q-learning on MPE `simple_tag_v3` (3 predators vs 1 prey) and compares them A/B:

1. **Baseline IQL** (`src/iql_teams.py`) — each team has its own Q-network, optimized against its own rewards. No explicit opponent modeling.
2. **Opponent-Aware IQL / OA-IQL** (`src/iql_teams_oa.py`) — same as above, plus an auxiliary head on each Q-network that predicts the opponent team's actions from the current observation. Trained with a joint loss:

    `L_total = L_Q  +  OPP_AUX_COEF · CrossEntropy(opp_action_pred, opp_action_true)`

The auxiliary loss forces the shared trunk (dense → GRU) to encode opponent-relevant features. The Q-head then reads Q-values off a representation that is already opponent-aware — no change to the Bellman target.

---

## 1. Why MPE simple_tag

- 3 **predators** (speed 1.0) chase 1 **prey** (speed 1.3, 30% faster).
- Discrete actions per agent: `[no-op, up, down, left, right]`.
- Reward: per capture, each predator gets +10 (team +30); prey gets −10 plus a smooth `map_bounds_reward` penalty if it leaves the arena.
- Episode length: 25 steps (fixed horizon).
- Asymmetric: different speeds, different obs sizes (pred 16, prey 14), different reward structures.

This is the smallest environment that is simultaneously discrete-action, continuous-state, and adversarial — so iteration is fast (2 M env-steps in ~2.5 min on CPU).

---

## 2. Baseline IQL (two-team fork)

Stock JaxMARL `iql_rnn.py` shares a **single** Q-network across all 4 agents:

```python
jax.vmap(net.apply, in_axes=(None, 0, 0, 0))(params, hs, obs, dones)
#                              ^^^^  "None" shares params across agents
```

Fine for cooperative tasks; broken for adversarial ones — predator gradients (maximize capture) and prey gradients (minimize capture) both update the same weights, cancelling.

Our fix: split into two `TrainState`s, two Q-networks, two target networks, shared replay buffer (split per-team at loss time). See `src/iql_teams.py:99-105` for the team split and `:186-194` for per-team init.

Within the predator team, the 3 agents share params (same role, same reward). A 4-dim agent-ID one-hot appended by `CTRolloutManager` gives them distinguishing identity inside a shared obs.

---

## 3. Opponent-Aware IQL

### Architecture change

The Q-network grows from one head to two:

```
Dense(64) → ReLU → GRU(64) ─┬─ Dense(action_dim)                = Q-values
                             └─ Dense(opp_n_agents * action_dim) = opp-action logits
```

- For the **predator network** (3 shared-param agents), each predator outputs a 5-way distribution over the single prey's action.
- For the **prey network** (1 agent), it outputs three 5-way distributions — one per predator.

See `src/iql_teams_oa.py:71-95` (`RNNQOppNetwork`).

### Loss change

At learn time, the minibatch already contains all agents' actions, so the opponent target is free:

```python
_opp_actions = batchify_team(minibatch.actions, opp_ags)   # (opp_n, B, T)
# opp_logits: (team_n, B, T, opp_n, action_dim)
ce   = -take_along_axis(log_softmax(opp_logits), opp_tgt).mean()
loss = q_loss + 0.5 * ce           # OPP_AUX_COEF = 0.5 by default
```

`OPP_AUX_COEF = 0.5` is set in `configs/alg/ql_teams_oa_simple_tag.yaml`. The Bellman target and target-network update are untouched — only the trunk gets an extra gradient path.

### What this buys you

- **Representation learning**: the trunk is forced to encode "where is the prey likely to go next" (or, for prey, "which of the three predators is about to charge me").
- **Zero inference cost**: at eval time you just run the Q-head as before. The opponent head is discarded (though you could use it — see §6).
- **Drop-in compatibility**: same buffer, same optimizer, same ε-schedule, same eval loop. The only changes are the network's second head and the loss.

---

## 4. Results from the 2 M-step run

Both variants trained for 2 000 000 env-steps, 1 seed, on M4 Pro CPU.

| metric | baseline IQL | OA-IQL | Δ |
|---|---|---|---|
| wall-clock | 150 s | 158 s | +5 % |
| throughput | 13 300 env-steps/s | 12 650 env-steps/s | −5 % |
| final pred greedy return (30-step ep) | **+33.12** | **+24.69** | −8.44 |
| final prey greedy return (30-step ep) | **−49.05** | **−37.78** | +11.27 |
| pred test-return peak | ~+103 @ update 2000 | **~+144 @ update 2000** | +41 |
| opp-action accuracy (pred head) | — | **~0.57** (asymptote) | — |
| opp-action accuracy (prey head) | — | **~0.57** (asymptote) | — |
| random-guess accuracy | 0.20 | 0.20 | — |

### The three-phase story

1. **Exploration → co-adaptation (0–2000 updates)**. Both variants follow the same trajectory: predators figure out chasing, prey is still learning boundaries.
2. **Peak divergence (updates ~2000)**. OA-IQL predators *peak significantly higher* than baseline (+144 vs +103). This is the sample-efficiency story: opponent modeling accelerates the pred's initial policy improvement.
3. **Equilibrium shift (updates 4000–9615)**. Prey catches up. By the end, **OA-IQL prey is +11 better than baseline prey, and OA-IQL pred is −8 worse than baseline pred**. The opponent-modeling benefit is asymmetric: predicting 3 predators (prey's task) gives more exploitable information than predicting 1 prey (pred's task), and in a zero-sum-like co-training loop, the side with the stronger opponent model pulls the equilibrium its way.

This is a nuanced result — honest rather than a clean "OA wins" claim. It's the right framing for adversarial MARL: symmetric changes have asymmetric effects.

---

## 5. Opponent head as a diagnostic

The auxiliary accuracy curve (see `plots/compare_opp_modeling.png`) is a free training-time instrument:

- Both teams climb from 20 % (random) to ~57 % by update 5000.
- Pred's accuracy is briefly higher than prey's in the early phase, matching the early pred return peak — the pred trunk learns opponent dynamics faster.
- Cross-entropy loss tracks the accuracy story: converges around 1.1 nats. Floor is bounded below by policy entropy of the opponent (still ε = 0.05 exploring).

When this accuracy plateaus below ~0.35, training is stuck; when it climbs steadily, representation learning is happening.

---

## 6. Code references (if asked during demo)

| thing | file : line |
|---|---|
| two-team split | `src/iql_teams.py:99-105` |
| per-team train state init (baseline) | `src/iql_teams.py:186-194` |
| per-team learn phase (baseline) | `src/iql_teams.py:254-290` |
| **RNNQOppNetwork** (two-head Q-net) | `src/iql_teams_oa.py:71-95` |
| **per-team learn with opp-aux loss** | `src/iql_teams_oa.py:241-310` |
| opp-action CE + accuracy | `src/iql_teams_oa.py:269-294` |
| auxiliary-loss weight | `configs/alg/ql_teams_oa_simple_tag.yaml:23` (`OPP_AUX_COEF: 0.5`) |
| metrics split by team | `src/iql_teams_oa.py:369-378` |
| stock IQL's shared-net wrong pattern | `JaxMARL/baselines/QLearning/iql_rnn.py:166-171, 263-264` |
| simple_tag pred reward | `JaxMARL/jaxmarl/environments/mpe/simple_tag.py:163` |
| simple_tag prey + boundary reward | `simple_tag.py:157-160` |
| obs preprocessing (pad + ID one-hot) | `JaxMARL/jaxmarl/wrappers/baselines.py:328-335` |

---

## 7. Quick dtype gotcha (kept from baseline)

`simple_tag.py:163` computes `adversary_reward = 10 * jnp.sum(collisions)` → int32, but prey reward (with smooth boundary penalty) is float32. Buffer-init traj is int32, rollout with `REW_SCALE = 1.0` becomes float32, and flashbax chokes with `types: float32 != int32` on the first `buffer.add`. Fix: explicit `.astype(jnp.float32)` on rewards in both sample and rollout paths. See `src/iql_teams.py:201, 269-272` (same pattern in `iql_teams_oa.py`).

---

## 8. Repro

```bash
conda activate pp_cotrain
# baseline
python src/iql_teams.py
# opponent-aware
python src/iql_teams_oa.py alg=ql_teams_oa_simple_tag
# compare
python src/compare_plots.py \
  --baseline logs/MPE_simple_tag_v3/iql_teams_MPE_simple_tag_v3_seed0_metrics.npz \
  --oa       logs/MPE_simple_tag_v3/iql_teams_oa_MPE_simple_tag_v3_seed0_metrics.npz
# rollout GIF (OA)
python src/visualize_rollout_oa.py \
  --pred_params logs/MPE_simple_tag_v3/iql_teams_oa_MPE_simple_tag_v3_pred_seed0_vmap0.safetensors \
  --prey_params logs/MPE_simple_tag_v3/iql_teams_oa_MPE_simple_tag_v3_prey_seed0_vmap0.safetensors \
  --seed 7 --steps 60 --out plots/rollout_oa.gif
```

Both training runs finish in about 2.5 minutes. End-to-end reproduction including plots and GIFs is under 10 minutes on a MacBook.
