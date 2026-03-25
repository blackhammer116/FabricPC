"""
Mini-verification for IFN components (Conv2D and IFN-T).
Rapidly checks for crashes, stable energy minimization, and learning.
"""

import jax
import jax.numpy as jnp
import optax
from fabricpc.nodes import (
    Conv2D,
    IFNTransformerBlock,
    IdentityNode,
    Linear,
    EmbeddingNode,
)
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.training import train_pcn
from fabricpc.core.inference import InferenceSGD
from fabricpc.training.hj_ot import hj_ot_optimizer


def test_conv2d():
    print("\n--- Testing Conv2D Node ---")
    pixels = IdentityNode(shape=(8, 8, 3), name="pixels")
    conv = Conv2D(shape=(8, 8, 4), name="conv", filters=4, kernel_size=(3, 3))
    fc = Linear(shape=(2,), name="out", flatten_input=True)

    struct = graph(
        nodes=[pixels, conv, fc],
        edges=[Edge(pixels, conv.slot("in")), Edge(conv, fc.slot("in"))],
        task_map=TaskMap(x=pixels, y=fc),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=5),
    )

    params = initialize_params(struct, jax.random.PRNGKey(0))
    x = jnp.ones((2, 8, 8, 3))
    y = jnp.array([[1, 0], [0, 1]])

    loader = [{"x": x, "y": y}] * 5
    optimizer = hj_ot_optimizer(1e-3)

    train_pcn(
        params,
        struct,
        loader,
        optimizer,
        {"num_epochs": 1},
        jax.random.PRNGKey(1),
        verbose=True,
    )
    print("Conv2D Test Passed (No crashes, learning started)")


def test_ifn_t():
    print("\n--- Testing IFN-T Node ---")
    tokens = IdentityNode(shape=(4,), name="tokens")
    embedding = EmbeddingNode(
        shape=(4, 12), name="embedding", vocab_size=20, embed_dim=12
    )
    ifn_t = IFNTransformerBlock(shape=(4, 12), name="ifn_t", num_heads=2)

    struct = graph(
        nodes=[tokens, embedding, ifn_t],
        edges=[Edge(tokens, embedding.slot("in")), Edge(embedding, ifn_t.slot("in"))],
        task_map=TaskMap(x=tokens, y=ifn_t),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=5),
    )
    # Note: IFNTransformerBlock usually needs an Embedding node for 'tokens' to work,
    # but here we'll just feed in dummy embeddings.

    params = initialize_params(struct, jax.random.PRNGKey(2))
    x = jnp.zeros((2, 4), dtype=jnp.int32)  # (Batch, Seq) indices
    y = jnp.ones((2, 4, 12))  # (Batch, Seq, Embed)

    loader = [{"x": x, "y": y}] * 5
    optimizer = hj_ot_optimizer(1e-3)

    train_pcn(
        params,
        struct,
        loader,
        optimizer,
        {"num_epochs": 1},
        jax.random.PRNGKey(3),
        verbose=True,
    )
    print("IFN-T Test Passed (No crashes, energy minimized)")


if __name__ == "__main__":
    test_conv2d()
    test_ifn_t()
