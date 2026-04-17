"""Plot training curves (predator vs prey) from metrics npz produced by iql_teams.py."""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(y, w=21):
    if y.ndim == 1:
        y = y[None]
    if y.shape[-1] < w:
        return y.mean(0), y.std(0)
    kernel = np.ones(w) / w
    sm = np.stack([np.convolve(yi, kernel, mode="valid") for yi in y])
    return sm.mean(0), sm.std(0)


def load(npz_path):
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="path to metrics npz")
    p.add_argument("-o", "--out", default="plots")
    p.add_argument("--smooth", type=int, default=21)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    m = load(args.npz)
    # keys look like 'pred__returned_episode_returns', 'test__pred__returned_episode_returns'
    # shape: (NUM_SEEDS, NUM_UPDATES) or scalar metrics

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # train-time per-team mean return
    for ax, key_suffix, title in [
        (axes[0], "returned_episode_returns", "Train return (per-team mean)"),
        (axes[1], "returned_episode_lengths", "Episode length"),
    ]:
        for team, color in [("pred", "tab:red"), ("prey", "tab:blue")]:
            key = f"{team}__{key_suffix}"
            if key not in m:
                continue
            y = np.asarray(m[key])
            if y.ndim == 1:
                y = y[None]
            mu, sd = smooth(y, w=args.smooth)
            x = np.arange(len(mu))
            ax.plot(x, mu, color=color, label=team, lw=1.8)
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.2)
        ax.set_xlabel("update step")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("MPE simple_tag  |  independent two-team IQL (co-training)")
    fig.tight_layout()
    out_path = os.path.join(args.out, "train_curves.png")
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")

    # loss + Q-values
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, key_suffix, title in [
        (axes2[0], "loss", "TD loss"),
        (axes2[1], "qvals", "Mean Q"),
    ]:
        for team, color in [("pred", "tab:red"), ("prey", "tab:blue")]:
            key = f"{team}__{key_suffix}"
            if key not in m:
                continue
            y = np.asarray(m[key])
            if y.ndim == 1:
                y = y[None]
            mu, sd = smooth(y, w=args.smooth)
            x = np.arange(len(mu))
            ax.plot(x, mu, color=color, label=team, lw=1.8)
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.2)
        ax.set_xlabel("update step")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig2.tight_layout()
    out_path2 = os.path.join(args.out, "loss_q.png")
    fig2.savefig(out_path2, dpi=140)
    print(f"wrote {out_path2}")

    print("\navailable metric keys:")
    for k in sorted(m):
        v = np.asarray(m[k])
        print(f"  {k}  shape={v.shape}")


if __name__ == "__main__":
    main()
