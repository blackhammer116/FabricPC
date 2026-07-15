"""Tiny Shakespeare comparison for BP, PC, NS-PC/HJ and end-to-end NS/HJ."""

from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from fabricpc.core.energy import SequenceNavierStokesEnergy
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph import initialize_params
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.training import train_autoregressive, evaluate_autoregressive
from fabricpc.training import (
    train_backprop_autoregressive,
    evaluate_backprop_autoregressive,
)
from fabricpc.training.hj_ot import hj_ot_optimizer
from fabricpc.training.train_autoregressive import compute_loss

NAMES = {
    "backprop": "Backprop + AdamW",
    "pc": "Standard PC + AdamW",
    "ns_pc_hj": "NS-PC + stabilized HJ-OT",
    "ns_e2e_hj": "End-to-end NS + stabilized HJ-OT",
}


class TextLoader:
    def __init__(
        self, data, vocab_size, seq_len, batch_size, max_batches, shuffle, seed
    ):
        self.data, self.vocab_size, self.seq_len = data, vocab_size, seq_len
        self.batch_size, self.max_batches, self.shuffle, self.seed = (
            batch_size,
            max_batches,
            shuffle,
            seed,
        )
        self.epoch = 0
        self.starts = np.arange(0, len(data) - seq_len - 1, seq_len, dtype=np.int32)

    def __len__(self):
        return min(self.max_batches, len(self.starts) // self.batch_size)

    def __iter__(self):
        starts = self.starts.copy()
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(starts)
        self.epoch += 1
        for b in range(len(self)):
            selected = starts[b * self.batch_size : (b + 1) * self.batch_size]
            x = np.stack([self.data[s : s + self.seq_len] for s in selected])
            yi = np.stack([self.data[s + 1 : s + self.seq_len + 1] for s in selected])
            yield {
                "x": x.astype(np.float32),
                "y": np.eye(self.vocab_size, dtype=np.float32)[yi],
            }


def load_text(path, seq_len, batch_size, train_batches, eval_batches, seed):
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    encoded = np.array([stoi[c] for c in text], dtype=np.int32)
    n = len(encoded)
    train, val, test = (
        encoded[: int(0.9 * n)],
        encoded[int(0.9 * n) : int(0.95 * n)],
        encoded[int(0.95 * n) :],
    )
    make = lambda data, batches, shuffle: TextLoader(
        data, len(chars), seq_len, batch_size, batches, shuffle, seed
    )
    return (
        len(chars),
        make(train, train_batches, True),
        make(val, eval_batches, False),
        make(test, eval_batches, False),
    )


def model(regime, args, vocab_size):
    ns = regime.startswith("ns_")
    energy = None
    if ns:
        energy = SequenceNavierStokesEnergy(
            viscosity=0.1,
            data_weight=1.0,
            latent_ns_weight=args.ns_weight,
            prediction_ns_weight=0.0,
            momentum_weight=1.0,
            divergence_weight=0.25,
            multi_field=True,
            multi_field_reduction="mean",
        )
    return create_deep_transformer(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        mlp_dim=args.mlp_dim,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        inference=InferenceSGDNormClip(
            eta_infer=0.03, infer_steps=args.infer_steps, max_norm=64
        ),
        weight_init={"type": "normal", "std": 0.04},
        residual_energy=energy,
    )


def stable_hj(args, steps):
    schedule = optax.cosine_decay_schedule(args.lr, steps, alpha=0.1)
    return hj_ot_optimizer(
        learning_rate=schedule,
        viscosity=0.9,
        viscosity_decay=0.9998,
        viscosity_min=0.5,
        transport_cost=1e-5,
        weight_decay=1e-5,
        stable_dynamics=True,
    )


def train_e2e_ns(params, structure, loader, optimizer, args, key):
    state = optimizer.init(params)

    def step(p, s, batch, k):
        def objective(q):
            clamps = {structure.task_map["x"]: batch["x"]}
            graph_state = initialize_graph_state(
                structure, batch["x"].shape[0], k, clamps=clamps, params=q
            )
            ce = compute_loss(
                graph_state, batch["y"], structure.task_map["y"], "cross_entropy"
            )
            physics = jnp.array(0.0)
            for name in structure.nodes:
                if name.endswith("_mha") or name.endswith("_mlp2"):
                    physics = physics + jnp.mean(graph_state.nodes[name].energy)
            return ce + physics, ce

        (loss, ce), grads = jax.value_and_grad(objective, has_aux=True)(p)
        updates, s = optimizer.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss, ce

    jit_step = jax.jit(step)
    for epoch in range(args.epochs):
        keys = jax.random.split(jax.random.fold_in(key, epoch), len(loader))
        for i, b in enumerate(loader):
            batch = {k: jnp.asarray(v) for k, v in b.items()}
            params, state, _, _ = jit_step(params, state, batch, keys[i])
    return params


def run(regime, args, vocab_size, train, val, test):
    structure = model(regime, args, vocab_size)
    pk, tk, ek = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    params = initialize_params(structure, pk)
    config = {"num_epochs": args.epochs, "use_causal_mask": False}
    history = []
    start = time.time()
    if regime in {"backprop", "pc", "ns_pc_hj"}:
        if regime == "backprop":
            train_fn, eval_fn, opt = (
                train_backprop_autoregressive,
                evaluate_backprop_autoregressive,
                optax.adamw(args.lr, weight_decay=1e-3),
            )
        elif regime == "pc":
            train_fn, eval_fn, opt = (
                train_autoregressive,
                evaluate_autoregressive,
                optax.adamw(args.lr, weight_decay=1e-3),
            )
        else:
            train_fn, eval_fn, opt = (
                train_autoregressive,
                evaluate_autoregressive,
                stable_hj(args, args.epochs * len(train)),
            )

        def callback(epoch, p, s, c, k):
            m = eval_fn(p, s, val, c, k)
            history.append(
                {"epoch": epoch + 1, "perplexity": m["perplexity"], "loss": m["loss"]}
            )
            print(
                f'[{regime}] epoch={epoch+1} val_ppl={m["perplexity"]:.2f}', flush=True
            )
            return m

        params, _, _ = train_fn(
            params,
            structure,
            train,
            opt,
            config,
            tk,
            epoch_callback=callback,
            verbose=True,
        )
    else:
        params = train_e2e_ns(
            params,
            structure,
            train,
            stable_hj(args, args.epochs * len(train)),
            args,
            tk,
        )
        m = evaluate_backprop_autoregressive(params, structure, val, config, ek)
        history = [
            {"epoch": args.epochs, "perplexity": m["perplexity"], "loss": m["loss"]}
        ]
    eval_fn = (
        evaluate_autoregressive
        if regime in {"pc", "ns_pc_hj"}
        else evaluate_backprop_autoregressive
    )
    metrics = eval_fn(params, structure, test, config, ek)
    return {
        "regime": regime,
        "name": NAMES[regime],
        "test": metrics,
        "history": history,
        "elapsed_seconds": time.time() - start,
    }


def plot(results, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in results:
        ax.plot(
            [x["epoch"] for x in r["history"]],
            [x["perplexity"] for x in r["history"]],
            marker="o",
            label=f'{r["name"]} (test {r["test"]["perplexity"]:.2f})',
        )
    ax.set(
        title="Tiny Shakespeare validation perplexity",
        xlabel="Epoch",
        ylabel="Perplexity (lower is better)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=Path, required=True)
    p.add_argument("--regimes", nargs="+", choices=tuple(NAMES), default=list(NAMES))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-batches", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--embed-dim", type=int, default=30)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--mlp-dim", type=int, default=60)
    p.add_argument("--infer-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ns-weight", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-json", type=Path, default=Path("shakespeare_ns_transformer.json")
    )
    p.add_argument(
        "--output-plot", type=Path, default=Path("shakespeare_ns_transformer.png")
    )
    args = p.parse_args()
    vocab, train, val, test = load_text(
        args.text,
        args.seq_len,
        args.batch_size,
        args.train_batches,
        args.eval_batches,
        args.seed,
    )
    results = []
    for regime in args.regimes:
        results.append(run(regime, args, vocab, train, val, test))
        args.output_json.write_text(
            json.dumps(
                {"config": vars(args), "vocab_size": vocab, "results": results},
                indent=2,
                default=str,
            )
            + "\n"
        )
    plot(results, args.output_plot)


if __name__ == "__main__":
    main()
