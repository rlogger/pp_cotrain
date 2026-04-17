# Self-check quiz — baseline + OA-IQL

Work through these without looking at code or guide. Answers at bottom. Target: >80 % before the demo.

---

## A. The setup

**A1.** How many predators and how many prey in `simple_tag_v3`? Action dim per agent?

**A2.** Give two asymmetries between the predator and prey sides.

**A3.** What does a single "capture event" pay out to each team?

**A4.** Why is γ = 0.9 appropriate for this env?

---

## B. Baseline IQL

**B1.** In stock JaxMARL `iql_rnn.py`, how many Q-networks exist across the 4 agents? What's the consequence for adversarial tasks?

**B2.** In `iql_teams.py`, how many Q-networks? How are they split?

**B3.** What is duplicated per-team? What is shared across teams?

**B4.** Why share within the predator team?

---

## C. OA-IQL architecture

**C1.** What does the OA-IQL Q-network output, beyond the Q-values?

**C2.** For the **predator** network (3 agents), what's the shape of the opponent-action head output (per agent, per step)?

**C3.** For the **prey** network (1 agent), what's the shape of the opponent-action head output?

**C4.** What is `OPP_AUX_COEF` and what does it control? What happens at 0? At 10?

**C5.** Is the opponent head used at inference time?

**C6.** Does the Bellman target change in OA-IQL?

---

## D. Results — numbers

**D1.** Final greedy pred return: baseline vs OA-IQL?

**D2.** Final greedy prey return: baseline vs OA-IQL?

**D3.** Peak pred return during training: baseline vs OA-IQL?

**D4.** Asymptotic opponent-action prediction accuracy for each team?

**D5.** Wall-clock cost of adding OA?

---

## E. Interpretation

**E1.** "OA-IQL beats baseline." True or false? Defend your answer.

**E2.** Why does pred peak higher with OA-IQL but settle lower?

**E3.** Why is the equilibrium shift *towards prey* rather than pred?

**E4.** What would happen if `OPP_AUX_COEF` were scaled to 10?

**E5.** You observe opp-action accuracy plateaued at 0.22 throughout training. What does that tell you?

---

## F. What's next

**F1.** Propose a modification that would make OA-IQL benefit pred instead of prey.

**F2.** How would you move from OA-IQL to a planning-based method using the opp-head?

**F3.** What would OA-IQL with opponent-reward prediction (instead of opponent-action) buy you?

---

## Answers

### A
- **A1.** 3 pred, 1 prey. Action dim = 5 (no-op, 4 cardinal).
- **A2.** Speed (pred 1.0 vs prey 1.3), obs dim (pred 16 vs prey 14), reward (pred +10/capture per agent, prey −10 + smooth boundary penalty), boundary (only prey can leave arena).
- **A3.** Pred team: +30 (each of 3 preds gets +10). Prey: −10, plus `map_bounds_reward` if it's out.
- **A4.** Episodes are 25 steps; γ²⁵ ≈ 0.07 means rewards beyond horizon are already discounted out. γ = 0.99 would weight phantom post-episode value.

### B
- **B1.** One Q-net, shared across all 4 agents via `vmap(..., in_axes=(None, 0, 0, 0))`. In adversarial tasks, pred and prey gradients update the same weights in opposite directions — they cancel.
- **B2.** Two. One shared across the 3 predators; one for the lone prey.
- **B3.** Per-team: `TrainState`, online params, target params, optimizer state, loss computation. Shared: replay buffer, ε-schedule, rollout env.
- **B4.** The 3 predators are interchangeable — same role, same reward. Sharing cuts sample complexity ~3× and is standard in cooperative MARL (VDN, MAPPO, QMIX all share within-team).

### C
- **C1.** An auxiliary head of logits over the opponent team's actions. Shape `(opp_n_agents, action_dim)`.
- **C2.** Each predator outputs a **(1, 5)** prediction over the single prey's action.
- **C3.** The prey outputs **(3, 5)** — three independent 5-way distributions, one per predator.
- **C4.** The weight on the auxiliary cross-entropy loss. At 0: OA-IQL reduces exactly to baseline (aux head still exists but gets no gradient). At 10: aux loss dominates, Q-learning breaks, returns tank.
- **C5.** No. Eval uses only the Q-head. The opp-head is training-time only.
- **C6.** No. `L = L_Q + 0.5 · L_opp`. Bellman target is unchanged. The aux loss only reshapes the shared trunk's representation.

### D
- **D1.** Baseline **+33.12**; OA-IQL **+24.69**. Δ = −8.44.
- **D2.** Baseline **−49.05**; OA-IQL **−37.78**. Δ = +11.27.
- **D3.** Baseline ≈ +103 at update 2000; OA-IQL ≈ +144 at update 2000.
- **D4.** Both teams asymptote around 0.57 (random baseline is 0.20).
- **D5.** 150 s → 158 s (+5 %). Throughput drops from ~13 300 to ~12 650 env-steps/s.

### E
- **E1.** False. OA-IQL shifts the equilibrium toward prey. Pred loses 8 return points, prey gains 11. The *sum* is +3 — the prey gain outweighs the pred loss, but neither side strictly improves. This is a nuanced result, not a uniform win.
- **E2.** Sample efficiency vs equilibrium: the opp-aware trunk gives pred a sharper early-training signal (faster credit assignment), so its Q-values climb faster and peak higher. But prey is *also* getting the aux-loss benefit, and prey benefits more, so eventually prey's improved policy pulls pred's return back down.
- **E3.** Information asymmetry. Prey predicts 3 actors × 5 actions = 15 dof; pred predicts 1 actor × 5 actions = 5 dof. The prey's aux task is harder but richer, and once it solves that task, it has more information to condition evasion on.
- **E4.** Aux loss dominates. Trunk would optimize for opp-prediction; Q-values would stay noisy; returns would collapse. Classic degenerate-aux-task failure.
- **E5.** Either (a) the opponent's policy is near-random (ε very high, or policy is stuck at init), or (b) the trunk capacity is too small to learn opp dynamics, or (c) `OPP_AUX_COEF` is too low relative to Q-loss scale. Inspect Q-loss first; if that's fine, raise the aux coef.

### F
- **F1.** Asymmetric aux weights: `OPP_AUX_COEF_PRED = 0.5, OPP_AUX_COEF_PREY = 0.0` — pred gets the benefit, prey stays baseline. Or: train pred with OA against a baseline prey (exploiter setup).
- **F2.** Use the opp-head at action-selection: for each candidate self-action a, sample k opponent-actions from the opp-head's posterior, evaluate `Q(s, a)` conditioned on those, pick the a that maximizes robust-min or expected value. That's 1-step planning with a learned model. Extend to MCTS for multi-step lookahead.
- **F3.** A reward model is more policy-invariant — as the opponent's policy shifts during co-training, a reward model stays closer to a fixed target (the opponent's objective). That's the core of inverse-RL. Trade-off: rewards carry less bit-rate than actions (often sparse) and harder to train supervised. OA-IQL is the cheap-fast version; OR-IQL (opponent-reward) is the slower-but-stabler version.
