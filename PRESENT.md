
- [ ] Terminal open at repo root: `cd ~/Documents/Projects/pp_cotrain`
- [ ] `conda activate pp_cotrain` — confirm prompt shows the env
- [ ] Open three image viewers side-by-side:
  - `plots/compare_test_returns.png`   ← the main plot
  - `plots/compare_opp_modeling.png`   ← the aux-head plot
  - `plots/compare_summary.png`        ← the bar chart
- [ ] Open two GIFs ready to loop:
  - `plots/rollout.gif`     (baseline, 5 captures)
  - `plots/rollout_oa.gif`  (OA-IQL, 2 captures)
- [ ] Editor open with two files:
  - `src/iql_teams.py` jumped to line **99** (two-team split)
  - `src/iql_teams_oa.py` jumped to line **71** (two-head net)
- [ ] `docs/05_cheatsheet.md` open on a second monitor if you have one

---

## The 90-second opener (verbatim, memorize)

> "I trained two variants of two-team independent Q-learning on MPE simple_tag — 3 predators versus 1 prey. The baseline is vanilla IQL. The second variant, OA-IQL, adds an auxiliary head on each team's Q-network that predicts the opponent's next action from the current observation, trained with a cross-entropy loss at half weight. Both runs are 2 million env-steps, about 150 seconds each on my M4 Pro CPU.
>
> The question I was testing: does adding cheap opponent modeling to a co-training loop help, and if so, which side of the adversarial game does it help more?"

---

## The results walk — 4 minutes

### 1. Show `plots/compare_test_returns.png` (90 seconds)

> "Left panel is the predator team's greedy test return over training. Right panel is the prey's. Dashed grey is baseline; solid colored is OA-IQL.
>
> Three phases.
>
> First, updates 0 to 2000 — both variants track each other. Predators learn chasing; prey learns to stop running out of bounds.
>
> Second, the peak around update 2000. **OA-IQL pred peaks at +144; baseline peaks at +103**. That's a real sample-efficiency lift — the opp-aware trunk gives pred a denser gradient signal early, so its Q-values sharpen faster.
>
> Third, updates 4000 through the end. Prey catches up. And here's the counter-intuitive result: **OA-IQL pred ends up *below* baseline pred. Prey ends up *above* baseline prey.**"

Pause. Then:

> "The explanation is information-theoretic. Prey's aux task is to predict 3 predators' actions — 15 degrees of freedom. Pred's aux task is to predict 1 prey's action — 5 degrees of freedom. Prey has more to gain from opponent modeling. In a zero-sum-like equilibrium, the side with the richer opponent model pulls the equilibrium its way."

### 2. Show `plots/compare_summary.png` (30 seconds)

> "The bars make it concrete. Pred: +33 drops to +25. Prey: −49 rises to −38. The sum is +2.8, but neither side strictly improves. This isn't a uniform win. It's an equilibrium shift."

### 3. Show `plots/compare_opp_modeling.png` (45 seconds)

> "The auxiliary head itself — this is a free training-time diagnostic. Both teams climb from 20% accuracy, which is random on a 5-way task, up to about **57%**. Pred gets there slightly faster because its task is easier.
>
> If this curve ever plateaus below 0.35, I know the trunk isn't learning opponent dynamics and something's broken. It's the single most informative signal I have during training."

### 4. Show `plots/rollout_oa.gif` (45 seconds)

> "60-step greedy rollout from the OA-IQL policy, seed 7. Watch the prey — it reads predator intent early and cuts angles. Two captures in 60 steps. Compare to baseline, same seed: 5 captures. The policy class is identical; the equilibrium is different."

Optional: flip to `plots/rollout.gif` for 10 seconds to show baseline contrast.

### 5. The close (30 seconds)

> "So the headline isn't 'OA-IQL wins' — it's 'symmetric auxiliary objectives produce asymmetric gains in adversarial co-training, and the structure of the aux task predicts which side benefits.'
>
> Next natural experiment is asymmetric aux weights — give pred OA-modeling, leave prey baseline — which should isolate the pred-side sample-efficiency gain without prey's offsetting lift. That's a one-line config change, ten minutes to run."

**Stop talking. Wait for questions.**

---

## If asked (the 5 you'll definitely get)

**"So does it help or not?"**
> "It helps prey, at the cost of pred. Equilibrium shift, not uniform gain."

**"Why those numbers?"**
> "Single seed on 2 million env-steps. Variance check is a flag — `NUM_SEEDS: 5` via `jax.vmap` — and adds about 10 minutes of wall-clock. Haven't run it yet but happy to."

**"Is the opponent head used at inference?"**
> "No. Training-time auxiliary only. At eval I discard it and run just the Q-head. Same inference cost as baseline."

**"Why 0.5 for the aux weight?"**
> "Untuned default from the aux-loss literature — UNREAL, CURL — that range is 0.05 to 1. Sweep would take twelve minutes. At zero, OA-IQL reduces exactly to baseline. At very large, the aux loss dominates and Q-learning breaks."

**"How is this different from BIRL or MCTS?"**
> "It isn't a planner and it isn't Bayesian. It's a point-estimate classifier on opponent actions, sharing a trunk with the Q-function. BIRL would give a posterior over opponent *rewards*, which is more policy-invariant. MCTS would use the opp-head at leaf nodes for lookahead. This is the cheap drop-in version — DRON adapted to recurrent IQL on JaxMARL."

---

## If asked (the 3 you might get)

**"What's the wall-clock cost?"**
> "Five percent. 150 seconds baseline, 158 seconds OA-IQL. Aux head is about 5% more parameters."

**"Why do you even need the two-team split? Why not stock JaxMARL IQL?"**
> "Stock IQL shares one Q-network across all 4 agents via vmap. Fine for cooperation. For adversarial, pred gradients and prey gradients update the same weights in opposite directions — they cancel. I split into two networks, two target nets, one shared buffer." (**show `src/iql_teams.py:99`**)

**"How do you know the aux head is actually shaping the trunk rather than just being independent?"**
> "Two signals. One: the trunk is shared — both heads read the same GRU output, so the aux gradient flows into the trunk by construction. Two: if it weren't shaping Q-learning, OA-IQL's Q-values and return curves would be identical to baseline. They aren't — peak returns diverge by 41 points at update 2000."

---

## Emergency fallbacks

**If matplotlib windows won't display** — open the PNGs in Finder/Preview directly. They're already in `plots/`.

**If someone asks to see a number not in the plots** — run:
```bash
/opt/anaconda3/envs/pp_cotrain/bin/python -c "
import numpy as np
m = np.load('logs/MPE_simple_tag_v3/iql_teams_oa_MPE_simple_tag_v3_seed0_metrics.npz')
for k in sorted(m.files): print(k, m[k].shape, m[k].min(), m[k].max(), m[k].mean())
"
```

**If someone challenges the result** — say "single seed, happy to launch multi-seed right now" and in another terminal:
```bash
# edit configs/config.yaml: NUM_SEEDS: 5
python src/iql_teams_oa.py alg=ql_teams_oa_simple_tag
```
Takes ~12 minutes.

**If someone asks to see the code** — have `src/iql_teams_oa.py` already open at line 71 (the `RNNQOppNetwork`) and line 269 (the CE loss). The whole file is 470 lines; the diff from baseline is ~30 lines.

---

## What NOT to do or say

- Do **not** call OA-IQL "better" without qualifying which team.
- Do **not** say "Bayesian" — it isn't.
- Do **not** say "planning" or "MCTS" — this is model-free, aux-loss only.
- Do **not** over-read the 5-vs-2 captures in the seed-7 rollouts. Curves are the evidence; rollouts are the vibe.
- Do **not** commit to future work ("we'll have MCTS by next month") — the next step is the one-line asymmetric-coef experiment.
- Do **not** read the code aloud. Point to the line, summarize, move on.

---

## Timing

| segment | time | cumulative |
|---|---|---|
| opener | 1:30 | 1:30 |
| compare_test_returns | 1:30 | 3:00 |
| compare_summary | 0:30 | 3:30 |
| compare_opp_modeling | 0:45 | 4:15 |
| rollout | 0:45 | 5:00 |
| close | 0:30 | **5:30** |
| Q&A buffer | 4:30 | 10:00 |

If running short, cut `rollout` — it's the most compressible piece.

If running long, cut the baseline rollout comparison.

---

## Numbers to drill before walking in

Flashcard style. Cover the right column and test yourself.

| question | answer |
|---|---|
| wall-clock per run | **150 s / 158 s** |
| total env-steps | **2 000 000** |
| baseline pred final return | **+33.12** |
| OA-IQL pred final return | **+24.69** |
| baseline prey final return | **−49.05** |
| OA-IQL prey final return | **−37.78** |
| pred peak — baseline / OA | **+103 / +144** |
| opp-action accuracy asymptote | **~0.57 both teams** |
| random-guess accuracy | **0.20** |
| `OPP_AUX_COEF` | **0.5** |

---

## One-breath pitch (memorize for the hallway)

> "Two-team IQL on predator-prey. Added a DRON-style opponent-action prediction head to each team's Q-net at half weight. Result: not a uniform win — shifts the equilibrium toward prey by about 19 return points, because prey's aux task of predicting 3 predators is richer than pred's task of predicting 1 prey. Early training, OA pred peaks 41 points higher, so there's a real sample-efficiency lift before prey catches up. 150 seconds per run, single seed, full A/B committed on GitHub."
