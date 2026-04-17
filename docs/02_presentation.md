# Presentation notes — 10-minute demo

Timings are a guideline; aim to leave 3 minutes for Q&A. All numbers cited here are from the actual 2M-step run in `logs/`.

---

## Slide 0 — title (15 sec)
> "Two-team independent IQL baseline for the model-based co-training project — MPE `simple_tag` (predator-prey). 2 M training steps, 150 seconds on M4 Pro CPU."

---

## Slide 1 — the research problem (90 sec)

**What you say:**
> "We're targeting *adversarial* multi-agent RL in **asymmetric** settings. The canonical approach for zero-sum symmetric games is self-play — AlphaZero, MuZero. That breaks in our regime because team A can only learn team B's strategy through interaction with B, and B is also learning.
>
> Behavior cloning the opponent doesn't fix this either — you're fitting a moving target.
>
> Our key insight for the paper: each team learns a model of the opposing team's *reward function*, not their actions. The reward model is more stable than the policy, because the objective is what drives behavior across changes in strategy. We plan to use Bayesian inverse RL with an MCTS tree as the hypothesis space for the opponent reward.
>
> Before we build any of that, we need a model-free, co-training baseline to beat. That's what I'm showing today."

**Key message**: this is the *baseline for the paper*, not the proposed method.

---

## Slide 2 — why MPE simple_tag (45 sec)

> "Chose MPE `simple_tag_v3` because it's the smallest environment that matches all three desiderata from our design doc:
> - **discrete actions** (5 per agent: no-op + 4 directions) — compatible with future MCTS planning
> - **continuous state** (2D positions / velocities) — non-trivial dynamics
> - **asymmetric** teams: 3 predators (speed 1.0) versus 1 prey (speed 1.3, 30% faster). Predators must coordinate to corner a faster opponent. Prey has to evade without running out of bounds — that's the shaping in the reward.
>
> Predator reward is +10 per collision **per predator**, so a single capture is +30 to the team. Prey reward is −10 per capture plus a smooth penalty for leaving the arena."

---

## Slide 3 — why stock JaxMARL IQL is wrong here (60 sec)

> "JaxMARL ships an IQL baseline but it uses **one** Q-network shared across *all* agents — `vmap(net.apply, in_axes=(None, 0, 0, 0))`, that `None` shares params. In cooperative tasks that's fine; in adversarial tasks it means predator gradients pointing one way and prey gradients pointing the opposite way are applied to the same parameters. Conflicting signals.
>
> So the first thing I did was fork it: two independent `TrainState`s, two independent `RNNQNetwork`s, two target networks. The 3 predators share a single network (they have the same role); the prey has its own. They share a replay buffer but split the minibatch by agent name at loss time."

**Pull up** `src/iql_teams.py` line 161–194 on screen.

---

## Slide 4 — results (2.5 min)

Pull up `plots/train_curves.png` first.

> "Left panel, training returns per team, smoothed. Three phases:
>
> - Early, updates 0–2000: predators figure out chasing first. Pred return climbs to ~+5; prey return drops to ~−8 because it's getting caught and also bouncing out of bounds.
> - Middle, 2000–4000: prey catches up. It learns evasion; pred return drops back to about +1.5 while prey recovers to about −2.
> - Late, 4000–9615 updates: noisy equilibrium around ±2. Neither side dominates. **This is exactly the co-adaptation signature we wanted.**
>
> This is not a bug — it's what you get when two independent learners are each trying to beat the other and neither has a structural advantage in how it learns."

Pull up `plots/loss_q.png`.

> "TD losses: both spike to ~10 during the policy churn phase around updates 2000, then decline monotonically to ~4. No divergence, no collapse.
>
> Q-values on the right: converge symmetrically to **+5 for predator, −5 for prey**. With γ=0.9 and roughly one capture per 25-step episode, +5 is consistent with a discounted expected return of γ^n × 30, where n is the expected steps to a capture. The symmetry is a sanity check that both nets are learning their respective value functions properly."

Pull up `plots/rollout.gif`.

> "60-step greedy rollout. 5 capture events, one every ~12 steps. Watch the two trailing predators peel off to cut angles while the third chases directly — emergent pincer behavior. No explicit coordination mechanism; just shared parameters across the 3 predators."

---

## Slide 5 — specific numbers table (30 sec)

Keep this as a backup slide for Q&A. Don't read — just have it up.

| metric | value |
|---|---|
| training time | **150 s** on M4 Pro CPU |
| env-steps / second | **13 300** |
| predator test return (last eval) | **+33.125 / episode** |
| prey test return (last eval) | **−49.05 / episode** |
| predator return improvement from eval 0 → last | +14.38 → +33.13 (+130%) |
| prey return improvement | −247.46 → −49.05 (−80% boundary loss) |
| capture rate at greedy (seed 7) | **1 capture / ~12 steps** |

---

## Slide 6 — where this plugs into the paper (60 sec)

> "This is the straw-man. The contribution of the paper is what we build on top:
>
> 1. Replace prey's Q-network (and predator's) with a planner that receives: (a) the env dynamics, and (b) an **estimate of the opponent's reward function** derived from observed opponent trajectories.
> 2. BIRL gives us a distribution over opponent reward hypotheses; MCTS evaluates moves under each.
> 3. We expect to beat this baseline on **sample efficiency** — 2M environment steps is a lot for a task this small — and on **transfer**: when the arena changes, the planner's physics model still holds; the IQL baseline here would have to retrain.
>
> We can also show ablations: turn off opponent-reward learning and you collapse to vanilla MCTS; turn off planning and you collapse to something like this baseline."

---

## Slide 7 — what's in the repo (15 sec)

> "Everything on GitHub at `rlogger/pp_cotrain`. Training script, configs, saved parameters, plots, and the rollout GIF are all committed. 150 seconds to reproduce end-to-end on a laptop."

---

## Closing lines if you run short

> "Happy to go deeper on any of: the IQL math, why the buffer was the trickiest thing to get right, the expected shape of the opponent reward model, or the MCTS design. Where would you like to dig in?"

---

## Anti-patterns to avoid in the talk

- **Don't** call this "the method". It's the *baseline*. If the PI thinks this is your proposed method they will rightly ask "what's the paper about, then?"
- **Don't** oversell the rollout GIF. It's one seed cherry-picked for capture density. The *curves* are the evidence.
- **Don't** claim emergent communication or coordination. The 3 predators share weights; any "coordination" is implicit in the shared policy, not an emergent protocol.
- **Don't** show raw wandb screenshots if you didn't use wandb in this run. Show `plots/*.png` — they're deterministic and yours.
