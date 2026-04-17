# pp_cotrain — Predator-Prey Co-training Baseline

Two-team independent IQL baseline on MPE `simple_tag_v3` for the model-based co-training paper.

## Why this exists

The stock JaxMARL `iql_rnn.py` trains **one shared Q-network across all agents**. That's fine for cooperative tasks but wrong for adversarial predator-prey — predator and prey have opposite rewards, so shared gradients conflict. This project forks it into **two independent IQL learners** (one per team), each with its own network, buffer, and target, trained simultaneously against each other (co-training).

## Layout

```
configs/
  config.yaml                       # top-level hydra config
  alg/ql_teams_simple_tag.yaml      # algorithm hyperparams
src/
  iql_teams.py                      # two-team IQL training script
  plot_metrics.py                   # training curves from saved npz
  visualize_rollout.py              # GIF of a greedy episode
logs/                               # saved params + metrics.npz
plots/                              # generated figures/gifs
```

## Setup

```bash
# one-time
conda create -n pp_cotrain python=3.11 -y
conda activate pp_cotrain
pip install -e "../JaxMARL[algs]"
pip install "flax==0.10.2"    # JaxMARL pins jax<=0.4.38 but not flax; newer flax breaks
```

## Train

```bash
conda activate pp_cotrain
# default: 2M timesteps, 1 seed, saves params and metrics npz into logs/
python src/iql_teams.py
# override anything on the CLI
python src/iql_teams.py alg.TOTAL_TIMESTEPS=500000 NUM_SEEDS=3
```

## Plot

```bash
python src/plot_metrics.py logs/MPE_simple_tag_v3/iql_teams_MPE_simple_tag_v3_seed0_metrics.npz
```

## Visualize a rollout

```bash
python src/visualize_rollout.py \
  --pred_params logs/MPE_simple_tag_v3/iql_teams_MPE_simple_tag_v3_pred_seed0_vmap0.safetensors \
  --prey_params logs/MPE_simple_tag_v3/iql_teams_MPE_simple_tag_v3_prey_seed0_vmap0.safetensors
```

## What a good demo shows

- Predator team return climbs over training (more captures)
- Prey team return (negative of capture, minus boundary penalties) also stabilizes or climbs as it learns to evade
- Co-training dynamics: neither side trivially dominates
- Rollout GIF: coordinated predators chasing a fleeing prey
