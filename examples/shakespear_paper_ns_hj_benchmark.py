"""Paper-scale Tiny Shakespeare benchmark for stable NS + HJ-OT.

This reproduces the common architecture in ``ngc vs py vs bp.pdf`` while
keeping evaluation strictly autoregressive: targets are shifted by one token,
attention is causal inside every MHA node, and validation/test targets are
never clamped into the graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from fabricpc.core.energy import SequenceNavierStokesEnergy
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph import initialize_params
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.training.hj_ot import hj_ot_optimizer

PAPER_RESULTS = {
    "Backprop": 355.85,
    "NGC PC": 1022.0,
    "PyTorch PC": 2.28,
}


class TokenLoader:
    def __init__(self, tokens, seq_len, batch_size, shuffle, seed, max_batches=None):
        self.tokens = np.asarray(tokens, dtype=np.int32)
        self.seq_len, self.batch_size = seq_len, batch_size
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        self.max_batches = max_batches
        self.starts = np.arange(0, len(tokens) - seq_len - 1, seq_len, dtype=np.int32)

    def __len__(self):
        available = len(self.starts) // self.batch_size
        return min(available, self.max_batches) if self.max_batches else available

    def __iter__(self):
        starts = self.starts.copy()
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(starts)
        self.epoch += 1
        usable = len(self) * self.batch_size
        for offset in range(0, usable, self.batch_size):
            selected = starts[offset : offset + self.batch_size]
            x = np.stack([self.tokens[s : s + self.seq_len] for s in selected])
            y = np.stack([self.tokens[s + 1 : s + self.seq_len + 1] for s in selected])
            yield {"x": x, "y": y}


def train_or_load_tokenizer(train_text, path, vocab_size):
    if path.exists():
        tokenizer = Tokenizer.from_file(str(path))
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=1,
            special_tokens=["[UNK]"],
            show_progress=True,
        )
        tokenizer.train_from_iterator([train_text], trainer=trainer)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(path))
    actual = tokenizer.get_vocab_size()
    if actual != vocab_size:
        raise ValueError(f"tokenizer vocabulary is {actual}, expected {vocab_size}")
    return tokenizer


def prepare_data(
    text_path,
    tokenizer_path,
    vocab_size,
    seq_len,
    batch_size,
    seed,
    train_batches=None,
    eval_batches=None,
):
    raw = text_path.read_text(encoding="utf-8")
    n = len(raw)
    train_text, val_text, test_text = (
        raw[: int(0.9 * n)],
        raw[int(0.9 * n) : int(0.95 * n)],
        raw[int(0.95 * n) :],
    )
    tokenizer = train_or_load_tokenizer(train_text, tokenizer_path, vocab_size)
    encode = lambda text: np.asarray(tokenizer.encode(text).ids, dtype=np.int32)
    train, val, test = encode(train_text), encode(val_text), encode(test_text)
    loaders = (
        TokenLoader(train, seq_len, batch_size, True, seed, train_batches),
        TokenLoader(val, seq_len, batch_size, False, seed, eval_batches),
        TokenLoader(test, seq_len, batch_size, False, seed, eval_batches),
    )
    return tokenizer, (train, val, test), loaders


def create_model(args):
    ns = SequenceNavierStokesEnergy(
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
        vocab_size=args.vocab_size,
        inference=InferenceSGDNormClip(
            eta_infer=args.inference_lr,
            infer_steps=args.infer_steps,
            max_norm=64,
        ),
        weight_init={"type": "normal", "std": args.init_std},
        residual_energy=ns,
    )


def nll_from_probs(probs, targets):
    selected = jnp.take_along_axis(probs, targets[..., None], axis=-1)[..., 0]
    return -jnp.mean(jnp.log(selected + 1e-10))


def evaluate(params, structure, loader, key):
    def step(p, x, y, k):
        clamps = {structure.task_map["x"]: x}
        state = initialize_graph_state(
            structure, x.shape[0], k, clamps=clamps, params=p
        )
        probs = state.nodes[structure.task_map["y"]].z_mu
        nll = nll_from_probs(probs, y)
        correct = jnp.sum(jnp.argmax(probs, axis=-1) == y)
        return nll, correct

    jit_step = jax.jit(step)
    nll_sum, correct, tokens = 0.0, 0, 0
    for index, batch in enumerate(loader):
        x, y = jnp.asarray(batch["x"]), jnp.asarray(batch["y"])
        nll, hits = jit_step(params, x, y, jax.random.fold_in(key, index))
        count = int(y.size)
        nll_sum += float(nll) * count
        correct += int(hits)
        tokens += count
    mean_nll = nll_sum / tokens
    return {
        "nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "accuracy": correct / tokens,
        "tokens": tokens,
        "batches": len(loader),
    }


def generate_text(
    params, structure, tokenizer, prompt, seq_len, max_new_tokens, temperature, key
):
    """Sample autoregressively from a rolling causal context window."""
    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        raise ValueError("generation prompt produced no tokens")

    # The graph has a fixed sequence shape. For a short prompt, repeat its first
    # token only as invisible left context; those padding IDs are not decoded.
    context = ([prompt_ids[0]] * max(0, seq_len - len(prompt_ids)) + prompt_ids)[
        -seq_len:
    ]
    generated = []

    def predict(p, token_ids, rng):
        x = jnp.asarray(token_ids, dtype=jnp.int32)[None, :]
        state = initialize_graph_state(
            structure,
            1,
            rng,
            clamps={structure.task_map["x"]: x},
            params=p,
        )
        return state.nodes[structure.task_map["y"]].z_mu[0, -1]

    predict = jax.jit(predict)
    for index in range(max_new_tokens):
        sample_key, model_key = jax.random.split(jax.random.fold_in(key, index))
        probabilities = predict(params, jnp.asarray(context), model_key)
        logits = jnp.log(probabilities + 1e-10) / temperature
        token = int(jax.random.categorical(sample_key, logits))
        generated.append(token)
        context = (context + [token])[-seq_len:]

    return {
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "generated_token_ids": generated,
        "text": prompt + tokenizer.decode(generated),
    }


def train(params, structure, loader, val_loader, args, key):
    total_steps = args.epochs * len(loader)
    schedule = optax.cosine_decay_schedule(args.learning_rate, total_steps, alpha=0.1)
    optimizer = hj_ot_optimizer(
        learning_rate=schedule,
        viscosity=0.9,
        viscosity_decay=0.9998,
        viscosity_min=0.5,
        transport_cost=1e-5,
        weight_decay=1e-5,
        stable_dynamics=True,
    )
    opt_state = optimizer.init(params)

    def step(p, state, x, y, k, ns_scale):
        def objective(q):
            clamps = {structure.task_map["x"]: x}
            graph_state = initialize_graph_state(
                structure, x.shape[0], k, clamps=clamps, params=q
            )
            probs = graph_state.nodes[structure.task_map["y"]].z_mu
            ce = nll_from_probs(probs, y)
            physics = jnp.array(0.0)
            for name in structure.nodes:
                if name.endswith("_mha") or name.endswith("_mlp2"):
                    # FeedforwardStateInit deliberately zeros energy after it
                    # sets z_latent=z_mu.  Recompute here so the NS residual is
                    # genuinely part of the end-to-end objective.
                    node = structure.nodes[name]
                    energy_obj = node.node_info.energy
                    prediction = graph_state.nodes[name].z_mu
                    physics += jnp.mean(
                        type(energy_obj).energy(
                            prediction,
                            prediction,
                            energy_obj.config,
                            context={"node_info": node.node_info},
                        )
                    )
            return ce + ns_scale * physics, (ce, physics)

        (loss, (ce, physics)), grads = jax.value_and_grad(objective, has_aux=True)(p)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss, ce, physics

    jit_step = jax.jit(step)
    history = []
    global_step = 0
    for epoch in range(args.epochs):
        started = time.time()
        ce_total = physics_total = 0.0
        for batch_index, batch in enumerate(loader):
            x, y = jnp.asarray(batch["x"]), jnp.asarray(batch["y"])
            warmup_steps = max(1, int(args.ns_warmup_fraction * total_steps))
            ns_scale = min(1.0, global_step / warmup_steps)
            params, opt_state, _, ce, physics = jit_step(
                params,
                opt_state,
                x,
                y,
                jax.random.fold_in(key, global_step),
                jnp.asarray(ns_scale),
            )
            ce_total += float(ce)
            physics_total += float(physics)
            global_step += 1
        metrics = evaluate(
            params, structure, val_loader, jax.random.fold_in(key, epoch)
        )
        row = {
            "epoch": epoch + 1,
            "train_nll": ce_total / len(loader),
            "train_perplexity": math.exp(ce_total / len(loader)),
            "mean_ns_energy": physics_total / len(loader),
            "val_nll": metrics["nll"],
            "val_perplexity": metrics["perplexity"],
            "val_accuracy": metrics["accuracy"],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    return params, history


def count_params(params):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))


def plot_results(history, test, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    epochs = [x["epoch"] for x in history]
    axes[0].plot(
        epochs, [x["train_perplexity"] for x in history], marker="o", label="Train"
    )
    axes[0].plot(
        epochs, [x["val_perplexity"] for x in history], marker="o", label="Validation"
    )
    axes[0].axhline(
        test["perplexity"],
        color="black",
        linestyle="--",
        label=f'Test {test["perplexity"]:.2f}',
    )
    axes[0].set(
        title="NS + stable HJ-OT learning curve",
        xlabel="Epoch",
        ylabel="Perplexity (lower is better)",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    names = list(PAPER_RESULTS) + ["NS-HJ-OT"]
    values = list(PAPER_RESULTS.values()) + [test["perplexity"]]
    colors = ["#4C78A8", "#F58518", "#E45756", "#54A24B"]
    axes[1].bar(names, values, color=colors)
    axes[1].set_yscale("log")
    axes[1].set(
        title="Test perplexity vs PDF reference", ylabel="Perplexity (log scale)"
    )
    axes[1].tick_params(axis="x", rotation=20)
    for i, value in enumerate(values):
        axes[1].text(i, value * 1.08, f"{value:.2f}", ha="center")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=Path, required=True)
    p.add_argument(
        "--tokenizer", type=Path, default=Path("tinyshakespeare_bpe_11711.json")
    )
    p.add_argument("--vocab-size", type=int, default=11711)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--mlp-dim", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--train-batches", type=int)
    p.add_argument("--eval-batches", type=int)
    p.add_argument("--infer-steps", type=int, default=26)
    p.add_argument("--inference-lr", type=float, default=7.21e-3)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--ns-weight", type=float, default=1e-5)
    p.add_argument("--ns-warmup-fraction", type=float, default=0.1)
    p.add_argument("--init-std", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prompt",
        default="First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n",
    )
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument(
        "--output-json", type=Path, default=Path("shakespeare_paper_ns_hj.json")
    )
    p.add_argument(
        "--output-plot", type=Path, default=Path("shakespeare_paper_ns_hj.png")
    )
    p.add_argument(
        "--output-sample", type=Path, default=Path("shakespeare_paper_ns_hj_sample.txt")
    )
    args = p.parse_args()

    tokenizer, token_splits, loaders = prepare_data(
        args.text,
        args.tokenizer,
        args.vocab_size,
        args.seq_len,
        args.batch_size,
        args.seed,
        args.train_batches,
        args.eval_batches,
    )
    train_loader, val_loader, test_loader = loaders
    structure = create_model(args)
    init_key, train_key, test_key, generation_key = jax.random.split(
        jax.random.PRNGKey(args.seed), 4
    )
    params = initialize_params(structure, init_key)
    started = time.time()
    params, history = train(
        params, structure, train_loader, val_loader, args, train_key
    )
    test = evaluate(params, structure, test_loader, test_key)
    generation = generate_text(
        params,
        structure,
        tokenizer,
        args.prompt,
        args.seq_len,
        args.max_new_tokens,
        args.temperature,
        generation_key,
    )
    args.output_sample.write_text(generation["text"], encoding="utf-8")
    result = {
        "method": "end-to-end NS residual regularization + stable HJ-OT",
        "strictly_causal": True,
        "target_clamped_during_evaluation": False,
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "parameter_count": count_params(params),
        "text_sha256": hashlib.sha256(args.text.read_bytes()).hexdigest(),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "split_tokens": {
            name: len(values)
            for name, values in zip(("train", "validation", "test"), token_splits)
        },
        "batches": {
            "train": len(train_loader),
            "validation": len(val_loader),
            "test": len(test_loader),
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(x) for x in jax.devices()],
        },
        "history": history,
        "test": test,
        "generation": generation,
        "paper_reference_test_perplexity": PAPER_RESULTS,
        "elapsed_seconds": time.time() - started,
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    plot_results(history, test, args.output_plot)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
