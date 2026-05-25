import os
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Iterator

# 1. Environment Setup
from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

# set_jax_flags_before_importing_jax(jax_platforms="gpu") # Set to GPU for speed
os.environ.setdefault(
    "TFDS_DATA_DIR", os.path.join(tempfile.gettempdir(), "fabricpc_tfds")
)

# 2. Import FabricPC components
from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import SoftmaxActivation, FluidActivation, GeluActivation
from fabricpc.core.energy import CrossEntropyEnergy, NavierStokesEnergy
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc.training.hj_ot import hj_ot_optimizer
from fabricpc.graph.state_initializer import FeedforwardStateInit


# 3. Enhanced Data Loader Wrapper
class MnistNavierStokesLoader:
    def __init__(self, split: str, batch_size: int, n_channels=3, **loader_kwargs):
        self.loader = MnistLoader(
            split=split, batch_size=batch_size, tensor_format="NHWC", **loader_kwargs
        )
        self.n_channels = n_channels

    def __iter__(self):
        for images, labels in self.loader:
            batch_size = images.shape[0]
            # Map pixels to u, v channels; p is zeroed
            u = images
            v = images
            p = np.zeros_like(images)

            # Group into triplets
            triplet = np.concatenate([u, v, p], axis=-1)  # (B, 28, 28, 3)

            # Repeat triplets to fill n_channels
            n_triplets = (self.n_channels + 2) // 3
            uvp = np.concatenate([triplet] * n_triplets, axis=-1)
            uvp = uvp[..., : self.n_channels].astype(np.float32)

            yield {"x": uvp, "y": np.asarray(labels)}

    def __len__(self):
        return len(self.loader)


# 4. Define Optimized Network Architecture
def create_model(n_channels=15):  # Multiples of 3 work best for NS triplets
    pixels = IdentityNode(shape=(28, 28, n_channels), name="pixels")

    # Hidden Fluid Layer with Multi-Field support
    # We lower the weights for the physics residuals to allow learning to dominate
    fluid1_energy = NavierStokesEnergy(
        viscosity=0.5,
        latent_ns_weight=0.05,  # Conservative weight for physics prior
        prediction_ns_weight=0.05,
        momentum_weight=0.5,
        divergence_weight=0.5,
        data_weight=1.0,  # Keep data alignment strong
        multi_field=True,  # Applies NS to all channel triplets
    )

    fluid1 = Linear(
        shape=(28, 28, n_channels),
        # Using sharpening for better feature discrimination
        activation=FluidActivation(
            velocity_gain=1.5, pressure_gain=0.8, sharpening=0.1
        ),
        energy=fluid1_energy,
        flatten_input=False,
        name="fluid1",
    )

    # Output Classification Layer
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        name="class",
    )

    return graph(
        nodes=[pixels, fluid1, output],
        edges=[
            Edge(source=pixels, target=fluid1.slot("in")),
            Edge(source=fluid1, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        # CRITICAL: Increase inference steps for convergence in physics landscape
        inference=InferenceSGDNormClip(eta_infer=0.2, infer_steps=100, max_norm=1.0),
        graph_state_initializer=FeedforwardStateInit(),
    )


# 5. Training and Evaluation
def run_training():
    max_epochs = 30
    batch_size = 128
    n_channels = 15  # 5 fluid fields

    # Cosine Decay Learning Rate Schedule
    steps_per_epoch = 60000 // batch_size
    lr_schedule = optax.cosine_decay_schedule(
        init_value=1e-3, decay_steps=max_epochs * steps_per_epoch, alpha=0.1
    )

    print(f"Preparing loaders (n_channels={n_channels})...")
    train_loader = MnistNavierStokesLoader(
        "train", batch_size=batch_size, n_channels=n_channels, shuffle=True
    )
    test_loader = MnistNavierStokesLoader(
        "test", batch_size=batch_size, n_channels=n_channels, shuffle=False
    )

    print("Initializing Optimized IFN Model...")
    structure = create_model(n_channels=n_channels)
    params = initialize_params(structure, jax.random.PRNGKey(42))

    # Scaled HJ_OT Optimizer
    optimizer = hj_ot_optimizer(
        learning_rate=lr_schedule,
        viscosity=0.7,  # Lower viscosity for faster weight updates
        transport_cost=1e-5,  # Slight non-linear dissipation for stability
        weight_decay=1e-4,
    )

    def epoch_callback(epoch, params, structure, config, key):
        print(f"\n--- Epoch {epoch+1} Evaluation ---")
        metrics = evaluate_pcn(params, structure, test_loader, config, key)
        print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
        return metrics["accuracy"]

    print("Starting optimized training loop...")
    trained_params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config={"num_epochs": max_epochs},
        rng_key=jax.random.PRNGKey(42),
        epoch_callback=epoch_callback,
        verbose=True,
    )


if __name__ == "__main__":
    run_training()
