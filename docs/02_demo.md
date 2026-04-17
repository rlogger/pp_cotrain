# Demo script — 10-minute walk-through

Aim: leave 3 min for Q&A. Numbers are from the actual 2 M-step runs in `logs/`.

---

## 0. Title (15 s)

> "Two-team IQL on MPE simple_tag, with an A/B against opponent-aware IQL. 2 M training steps per variant, 150 s each on M4 Pro CPU."

---

## 1. The setup (60 s)

> "Three predators, one prey. Prey is 30 % faster, so predators have to coordinate; prey has to evade without leaving the arena (smooth boundary penalty). Discrete 5-action space per agent. The task is asymmetric — different speeds, different obs dims (pred 16, prey 14), different reward structures.
>
> Predator reward is +10 per collision per predator, so a team capture is +30. Prey gets −10 per capture plus the boundary penalty."

**Pull up** `plots/rollout_oa.gif`.

---

## 2. The baseline (60 s)

> "Stock JaxMARL IQL shares **one** Q-network across all 4 agents. For cooperation that's fine. For adversarial tasks, predator gradients and prey gradients pull the same weights in opposite directions — they cancel.
>
> Fix: two independent Q-networks, two target nets, shared replay buffer split per-team at loss time. The 3 predators share within-team weights (they're interchangeable); the prey has its own net."

**Pull up** `src/iql_teams.py:99-105` and `:186-194`.

---

## 3. Adding opponent modeling (90 s)

> "Now OA-IQL. Same architecture, plus a second head on each Q-network that predicts the opponent team's actions from the current observation. Trained with a cross-entropy loss at weight 0.5:
>
>     L = L_Q + 0.5 * CE(opp_pred, opp_actual)
>
> The shared trunk — a dense layer plus GRU — is forced to encode opponent-relevant features. The Q-head reads values off a representation that is already opponent-aware, without any change to the Bellman target.
>
> Why this is cheap: the opponent actions are already in the replay buffer. No extra rollouts, no extra network forward passes in inference — the opponent head gets dropped at test time."

**Pull up** `src/iql_teams_oa.py:71-95` (two-head net) and `:269-294` (CE loss).

---

## 4. Results (3 min)

**Pull up** `plots/compare_test_returns.png`.

> "Left panel, predator greedy test return over training. Right panel, prey.
>
> Three phases.
>
> *First*, updates 0–2000, both variants track each other: predators figure out chasing, prey is still learning boundaries. Prey return climbs from −247 to around −50 by update 1500 — this is almost entirely boundary-penalty reduction, not capture avoidance.
>
> *Second*, around update 2000, predators peak. And here's the first real A/B: **OA-IQL predator peaks at +144; baseline peaks at +103**. Opponent modeling accelerates the pred's initial policy improvement — it makes sharper early decisions because the trunk already encodes prey dynamics.
>
> *Third*, after update 3000, prey catches up. By update 9615, OA-IQL prey is +11 above baseline prey; OA-IQL pred is −8 below baseline pred. **The opponent-modeling benefit is asymmetric** — prey learns to predict 3 predators (more information) while pred only learns to predict 1 prey (less information). In a zero-sum-like equilibrium, the side with the stronger opponent model pulls the equilibrium its way."

**Pull up** `plots/compare_opp_modeling.png`.

> "The auxiliary head itself is diagnostic. Accuracy climbs from random (0.20) to ~0.57 for both teams by update 5000. Cross-entropy settles around 1.1 nats. When this curve plateaus low, your representation isn't learning. When it climbs steadily, it is."

**Pull up** `plots/compare_summary.png`.

> "Final greedy-eval bars. Pred: 33 → 25 (−8). Prey: −49 → −38 (+11). OA-IQL shifts the equilibrium toward prey by about 19 return points."

---

## 5. The rollout (45 s)

**Pull up** `plots/rollout_oa.gif` alongside `plots/rollout.gif` (baseline).

> "60-step greedy rollouts, same seed (7). Baseline: 5 captures in 60 steps, clean pincer. OA-IQL: 2 captures, tighter evasion — prey uses predator-predicted moves to cut off closing angles. Same policy class, different equilibrium."

---

## 6. What's in the repo (15 s)

> "Two training scripts, one config each, one compare script. 150 seconds per training run. End-to-end reproduction under 10 minutes on a laptop."

---

## Backup slide — numbers table

| metric | baseline | OA-IQL |
|---|---|---|
| wall-clock | 150 s | 158 s |
| throughput | 13 300 / s | 12 650 / s |
| pred final return (30-step ep) | +33.12 | +24.69 |
| prey final return (30-step ep) | −49.05 | −37.78 |
| pred peak return | +103 | +144 |
| opp-action accuracy (final) | — | ~0.57 |

---

## Anti-patterns to avoid

- Don't claim OA-IQL "beats" the baseline — it doesn't, uniformly. It shifts equilibrium, and the shift happens to favor prey here.
- Don't say the opponent head is used at inference. It's a training-time auxiliary only; eval uses just the Q-head.
- Don't over-interpret the 2-vs-5 captures in the seed-7 rollouts. Single-seed rollouts are noisy; the curves and bars are the evidence.
- Don't call OA-IQL "Bayesian" — it isn't. It's a single point-estimate opponent classifier. Bayesian inverse RL is a separate, heavier object.

---

## If pressed for time, show only these 3 artifacts

1. `plots/compare_test_returns.png` — the A/B evidence (45 s)
2. `plots/compare_opp_modeling.png` — the auxiliary head is actually learning (20 s)
3. `plots/rollout_oa.gif` — the behaviour (20 s)

Everything else is Q&A material.
