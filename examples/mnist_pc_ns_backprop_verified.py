"""Controlled MNIST comparison: backprop, standard PC, and NS-PC + HJ-OT.

All three regimes use the same architecture, input representation, seed,
training/test split, batch size, and epoch count. The script writes a JSON
record and a PNG/SVG accuracy curve.

Example:
    PYTHONPATH=. python examples/mnist_training_comparison.py \
        --data-dir /path/to/mnist-idx-files --epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import optax

from fabricpc.graph import initialize_params
from fabricpc.training import evaluate_pcn, train_pcn
from fabricpc.training.hj_ot import hj_ot_optimizer
from fabricpc.training.train_backprop import evaluate_backprop, train_backprop
from mnist_ns_hj_ot_verified import ArrayLoader, corrected_structure, load_mnist

DISPLAY_NAMES = {
    "backprop": "Backprop + AdamW",
    "pc": "Standard PC + AdamW",
    "ns_hj_ot": "NS-PC + stabilized HJ-OT",
}


def make_loaders(dataset, batch_size: int, seed: int, train_limit: int | None):
    train_x, train_y, test_x, test_y = dataset
    return (
        ArrayLoader(
            train_x,
            train_y,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            limit=train_limit,
        ),
        ArrayLoader(
            test_x,
            test_y,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        ),
    )


def run_regime(name: str, args, dataset):
    train_loader, test_loader = make_loaders(
        dataset, args.batch_size, args.seed, args.train_limit
    )
    use_ns = name == "ns_hj_ot"
    structure = corrected_structure(
        infer_steps=args.infer_steps,
        use_navier_stokes=use_ns,
    )
    master_key = jax.random.PRNGKey(args.seed)
    param_key, train_key = jax.random.split(master_key)
    params = initialize_params(structure, param_key)

    if name == "ns_hj_ot":
        schedule = optax.cosine_decay_schedule(
            init_value=args.learning_rate,
            decay_steps=args.epochs * len(train_loader),
            alpha=0.05,
        )
        optimizer = hj_ot_optimizer(
            learning_rate=schedule,
            viscosity=0.9,
            viscosity_decay=0.9998,
            viscosity_min=0.5,
            transport_cost=1e-5,
            weight_decay=1e-5,
            nesterov=False,
            stable_dynamics=True,
        )
        train_fn, evaluate_fn = train_pcn, evaluate_pcn
    else:
        optimizer = optax.adamw(
            learning_rate=args.learning_rate,
            weight_decay=1e-3,
        )
        if name == "backprop":
            train_fn, evaluate_fn = train_backprop, evaluate_backprop
        else:
            train_fn, evaluate_fn = train_pcn, evaluate_pcn

    history = []
    started = time.time()

    def callback(epoch, current_params, current_structure, config, rng_key):
        metrics = evaluate_fn(
            current_params,
            current_structure,
            test_loader,
            config,
            rng_key,
        )
        accuracy = float(metrics["accuracy"] * 100.0)
        history.append({"epoch": epoch + 1, "test_accuracy": accuracy})
        print(
            f"[{DISPLAY_NAMES[name]}] epoch={epoch + 1} "
            f"test_accuracy={accuracy:.2f}%",
            flush=True,
        )
        return accuracy

    trained_params, _, _ = train_fn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config={"num_epochs": args.epochs},
        rng_key=train_key,
        verbose=True,
        epoch_callback=callback,
    )
    del trained_params

    return {
        "regime": name,
        "display_name": DISPLAY_NAMES[name],
        "optimizer": "stabilized_hj_ot" if use_ns else "adamw",
        "energy": "navier_stokes" if use_ns else "gaussian",
        "best_test_accuracy": max(item["test_accuracy"] for item in history),
        "final_test_accuracy": history[-1]["test_accuracy"],
        "elapsed_seconds": time.time() - started,
        "history": history,
    }


def save_plot(results, output_png: Path, output_svg: Path):
    styles = {
        "backprop": {"color": "#4C78A8", "marker": "o"},
        "pc": {"color": "#F58518", "marker": "s"},
        "ns_hj_ot": {"color": "#54A24B", "marker": "^"},
    }
    fig, axis = plt.subplots(figsize=(10, 6.5))
    for result in results:
        history = result["history"]
        axis.plot(
            [item["epoch"] for item in history],
            [item["test_accuracy"] for item in history],
            linewidth=2.2,
            markersize=4,
            markevery=max(1, len(history) // 10),
            label=(
                f'{result["display_name"]} '
                f'(best {result["best_test_accuracy"]:.2f}%)'
            ),
            **styles[result["regime"]],
        )
    axis.axhline(99.0, color="#777777", linestyle="--", linewidth=1, label="99% target")
    axis.set_title("MNIST accuracy: identical architecture, three learning regimes")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Test accuracy (%)")
    axis.set_xticks(range(1, len(results[0]["history"]) + 1))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_svg)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--infer-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=tuple(DISPLAY_NAMES),
        default=list(DISPLAY_NAMES),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("mnist_training_comparison.json"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("mnist_training_comparison.png"),
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=Path("mnist_training_comparison.svg"),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from --output-json without training.",
    )
    parser.add_argument(
        "--reuse-ns-result",
        type=Path,
        help="Merge a compatible result from mnist_ns_hj_ot_verified.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.plot_only:
        comparison = json.loads(args.output_json.read_text(encoding="utf-8"))
        results = comparison["results"]
        if args.reuse_ns_result is not None:
            source = json.loads(args.reuse_ns_result.read_text(encoding="utf-8"))
            reused = {
                "regime": "ns_hj_ot",
                "display_name": DISPLAY_NAMES["ns_hj_ot"],
                "optimizer": "stabilized_hj_ot",
                "energy": "navier_stokes",
                "best_test_accuracy": source["best_test_accuracy"],
                "final_test_accuracy": source["history"][-1]["test_accuracy"],
                "elapsed_seconds": source["elapsed_seconds"],
                "history": source["history"],
            }
            results = [item for item in results if item["regime"] != "ns_hj_ot"]
            results.append(reused)
            comparison["results"] = results
            args.output_json.write_text(
                json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
            )
        save_plot(results, args.output_png, args.output_svg)
        print(f"saved {args.output_json}")
        print(f"saved {args.output_png}")
        print(f"saved {args.output_svg}")
        return

    dataset = load_mnist(args.data_dir)
    results = []
    for regime in args.regimes:
        result = run_regime(regime, args, dataset)
        results.append(result)
        partial = {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "infer_steps": args.infer_steps,
            "train_examples": (
                len(dataset[0]) if args.train_limit is None else args.train_limit
            ),
            "test_examples": len(dataset[2]),
            "results": results,
        }
        args.output_json.write_text(
            json.dumps(partial, indent=2) + "\n", encoding="utf-8"
        )

    save_plot(results, args.output_png, args.output_svg)
    print(f"saved {args.output_json}")
    print(f"saved {args.output_png}")
    print(f"saved {args.output_svg}")


if __name__ == "__main__":
    main()
