# Self-check quiz — pp_cotrain

Work through these **without looking at the code or guide**. Target: >80% correct before presenting. Answers and pointers at the bottom.

---

## Section A — the research problem (conceptual)

**A1.** Why does AlphaZero-style self-play *not* apply to our predator-prey setup? Give two specific reasons.

**A2.** Name the failure mode of using behavior cloning to model a learning opponent.

**A3.** What is the latent variable our proposed method estimates about the opponent, and why is that choice better than estimating their policy directly?

**A4.** What does "asymmetric" mean in this context? Give two specific asymmetries in `simple_tag_v3`.

**A5.** Classify each with M (model-free) / B (model-based): (a) AlphaZero, (b) stock IQL, (c) MuZero, (d) EfficientZero, (e) our proposed method, (f) our current baseline.

---

## Section B — the environment

**B1.** In `MPE_simple_tag_v3`, how many predators, how many prey, how many landmarks?

**B2.** What is the action space dimension for each agent?

**B3.** Obs dim for predator vs prey — and *why* are they different?

**B4.** What is the reward a **predator team** gets for a single capture event? (Careful: not the reward one predator gets.)

**B5.** Is the prey allowed to leave the arena? What happens if it does?

**B6.** What's the max speed of prey vs predator, and what's the ratio?

---

## Section C — architecture and training

**C1.** In stock JaxMARL `iql_rnn.py`, how many Q-networks are there across the 4 agents? What's the consequence for adversarial tasks?

**C2.** In `iql_teams.py`, how many Q-networks, and how is their use split?

**C3.** Name three things that are duplicated per-team in `iql_teams.py`.

**C4.** What's shared across teams (not duplicated)?

**C5.** Why share parameters within the predator team but not across teams?

**C6.** At learn time, we sample one minibatch and then... describe what happens next in 1–2 sentences.

**C7.** What does `CTRolloutManager` do to the observations before they enter the network?

**C8.** Describe the role of the agent-ID one-hot in the observation.

---

## Section D — hyperparameters

**D1.** With `TOTAL_TIMESTEPS=2e6, NUM_STEPS=26, NUM_ENVS=8`, how many update steps does training run?

**D2.** What does `LEARNING_STARTS=10000` actually *do*?

**D3.** What does `TAU=1.0` mean for target-network updates? How does that differ from `TAU=0.005`?

**D4.** What fraction of training is spent with high exploration (ε > 0.1)? (Hint: `EPS_DECAY=0.1`, `EPS_FINISH=0.05`.)

**D5.** Why is `GAMMA=0.9` appropriate for this env? Reference the episode length in your answer.

---

## Section E — results (must know specific numbers)

**E1.** Training wall-clock on M4 Pro CPU for 2M timesteps?

**E2.** Final greedy-eval predator return per 30-step episode?

**E3.** Final greedy-eval prey return per 30-step episode?

**E4.** Predator return improvement from first to last eval (multiplicative)?

**E5.** Predator return peaks around update \_\_\_\_ at value \_\_\_\_ , then settles around \_\_\_\_ by the end.

**E6.** Converged Q-values are about \_\_\_\_ for pred and \_\_\_\_ for prey.

**E7.** In the seed-7 rollout, how many captures occurred in 60 steps?

**E8.** Why doesn't the predator return monotonically increase?

---

## Section F — gotchas / debugging

**F1.** Why did I need to downgrade flax from 0.10.4 to 0.10.2?

**F2.** Describe the int32/float32 reward-dtype bug in one sentence — *where* it originates and *how* I fixed it.

**F3.** Why did setting `REW_SCALE=1.0` (rather than `1`) trigger the bug?

**F4.** What's the consequence of not fixing the dtype bug — does training silently degrade or does it hard-fail?

---

## Section G — positioning the work

**G1.** In one sentence each: what does the baseline *show*, and what does it *not* claim?

**G2.** Name two axes on which our proposed method should beat this baseline.

**G3.** If the PI says "train MAPPO instead" — give a principled reason to stay with IQL for the baseline.

**G4.** If asked "what's next?", give a specific two-track plan.

**G5.** What would happen to this IQL baseline if the arena geometry changed at test time?

---

## Answers

### Section A
- **A1.** (a) The task is *asymmetric*: predator and prey have different action dynamics, speeds, and reward structures — self-play requires role symmetry. (b) The rules are *not* of a turn-based game; it's simultaneous-move continuous-state. Self-play assumes a closed-form game model with known rules (Go, chess). Also acceptable: non-zero-sum components in the reward (boundary penalty for prey only).
- **A2.** BC fits a model of the opponent's actions. If the opponent is still learning, the BC target moves under you — you're forever one step behind. Guide §1, §7.
- **A3.** The opponent's **reward function**. Rewards are the underlying objective that drives behavior, so they're more invariant to policy changes than the policy itself is. Guide §8.
- **A4.** Asymmetric = teams have *different* dynamics / rewards / obs. Examples: prey is 30% faster (1.3 vs 1.0); prey obs = 14 dims vs predator 16; prey pays −10 plus boundary penalty, predators earn +10 per collision *each*; only prey can leave the arena.
- **A5.** (a) B (b) M (c) B (d) B (e) B (hybrid: model-based planner + model-free bootstrap) (f) M.

### Section B
- **B1.** 3 predators, 1 prey, 2 landmarks. `simple_tag.py:14-16`.
- **B2.** 5 (no-op + 4 cardinal directions).
- **B3.** 16 for predator (includes prey velocity), 14 for prey (omits predator velocity). Prey has strictly less information — a baked-in asymmetry.
- **B4.** +30. Each of the 3 predators gets +10 independently, and the "team reward" is the sum.
- **B5.** Yes, but it incurs `map_bounds_reward` — a smooth penalty. Most of the prey's training improvement (−247 → −49) is learning to stay in.
- **B6.** Prey max 1.3, predator max 1.0. Ratio 1.3×. Guide §2.

### Section C
- **C1.** One Q-network, shared across all 4 agents via `vmap(..., in_axes=(None, 0, 0, 0))`. For adversarial tasks, predator and prey gradients conflict on the same parameters. Guide §3.
- **C2.** Two networks: one for the 3 predators (shared within team), one for the lone prey.
- **C3.** `TrainState`, target-network params, optimizer state. Also: two losses per learn step, two `apply_gradients` calls, two target-update checks. Guide §4 table.
- **C4.** The flashbax buffer (stores all-agent trajectories), the ε-greedy schedule, the rollout env.
- **C5.** Within team, agents are interchangeable (same role, same reward) — sharing cuts sample complexity ~3×. Across teams they have opposite rewards — sharing would produce conflicting gradients. Guide §4.
- **C6.** Split the minibatch by agent name into per-team tensors; run each team's network forward through its own target to compute the TD target; compute team-local MSE loss; apply gradients independently.
- **C7.** Pads obs to the max agent's obs dim (16), then appends a 4-dim agent-ID one-hot → 20-dim uniform obs. Guide §2, §6.
- **C8.** Lets the shared (within-team) network distinguish between homogeneous team members — so the same network can produce different outputs for "predator 0" vs "predator 2" if state asymmetries demand.

### Section D
- **D1.** 2e6 / 26 / 8 = **9615** updates.
- **D2.** For the first 10 000 env steps (= 10000 / 8 ≈ 1250 rollouts), no gradient updates happen. The buffer fills with random-policy transitions so the first Bellman targets aren't computed on a tiny or biased sample.
- **D3.** `TAU=1.0` = **hard target update**: every `TARGET_UPDATE_INTERVAL` updates, copy online params → target params exactly. `TAU=0.005` would be Polyak (soft) averaging — target = 0.005 × online + 0.995 × target per update.
- **D4.** ε starts at 1.0 and decays linearly to 0.05 over `EPS_DECAY × NUM_UPDATES = 0.1 × 9615 ≈ 962` updates. After that, ε stays at 0.05. So ~10% of training has non-trivial exploration; the remaining 90% is ε=0.05.
- **D5.** Episodes are 25 steps. γ=0.9 → γ^25 ≈ 0.07 → rewards beyond the episode horizon are essentially discounted out. γ=0.99 would weight phantom post-episode rewards that don't exist.

### Section E
- **E1.** **150 seconds**.
- **E2.** **+33.125** per 30-step episode.
- **E3.** **−49.05** per 30-step episode.
- **E4.** 14.375 → 33.125 = **~2.3×** (or +130%).
- **E5.** Peaks ~**update 2000** at **~+5**, settles around **~+1.5** by the end.
- **E6.** **+5 pred, −5 prey** (symmetric).
- **E7.** **5 captures in 60 steps** (predator avg team reward 2.5/step × 60 steps = 150 / 30-per-capture).
- **E8.** Because prey is also learning. After pred initially gets good at chasing, prey's gradients get a sharp learning signal and it starts evading — pred return drops back. The plateau around ±2 is the co-adaptation equilibrium. Guide §6.

### Section F
- **F1.** JaxMARL pins `jax<=0.4.38` but does not pin flax. Default flax 0.10.4 uses `jax.api_util.debug_info`, introduced in jax 0.5. 0.10.2 is the last flax that works with jax 0.4.38.
- **F2.** `simple_tag.py` line 163 computes adversary reward as `10 * jnp.sum(collisions)` → int32. Prey reward is int + float boundary → float32. The flashbax buffer was initialized from a no-REW_SCALE sample trajectory (int32), but the real rollout path multiplies by `REW_SCALE=1.0` (float) → promotes to float32 → dtype mismatch on `buffer.add`. Fix: `.astype(jnp.float32)` on rewards in both paths (`src/iql_teams.py:201, 269-272`).
- **F3.** In Python, `1 * int32_array = int32_array` but `1.0 * int32_array = float32_array`. So using an int constant would have left dtypes matching by accident.
- **F4.** Hard-fail. `chex.assert_trees_all_equal_dtypes` raises `AssertionError: types: float32 != int32` on the first `buffer.add` after `LEARNING_STARTS`.

### Section G
- **G1.** *Shows*: independent-Q learners can co-train stably on an asymmetric adversarial task, with clear co-adaptation dynamics. *Does not claim*: sample efficiency, opponent modeling, transfer, robustness to env changes, or centralized-critic benefits — all of which are target improvements for the proposed method.
- **G2.** Sample efficiency (should match IQL performance in far fewer env steps); transfer (when arena geometry changes, planner with env-physics model should adapt immediately — IQL must retrain).
- **G3.** MAPPO changes the value-function architecture entirely. With IQL, any improvement in the proposed method is attributable to opponent-reward modeling, not to "better credit assignment via centralized critic". Baseline's job is to isolate variables.
- **G4.** (1) BIRL-based opponent-reward estimator (small reward MLP / GP, trained on observed opponent trajectories with BIRL loss). (2) MCTS planner that rolls out trajectories under the estimated opponent reward; use the IQL Q-network as leaf-node value estimator.
- **G5.** It would have to retrain from scratch. The policy has baked in the current obstacle layout as raw observations; it has no explicit model of arena physics to transfer. That's precisely where the proposed method should shine.
