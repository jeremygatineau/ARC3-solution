from typing import Any, Callable, Dict, Optional, Tuple
import sys, datetime, os, time
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ARC-AGI-3-Agents'))
from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState

sys.path.append(os.path.dirname(__file__))  # Add current directory to path

from utils import setup_experiment_directory, setup_logging_for_experiment, get_environment_directory
from view_utils import save_action_visualization
try: 
    from torch.utils.tensorboard import SummaryWriter
    _SUMMARY_KWARG = "log_dir"
except ImportError:
    _SUMMARY_KWARG = "logdir"
    try: 
        from tensorboardX import SummaryWriter
    except ImportError as e:
        raise ImportError("Requires PyTorch or tensorboardX to use SummaryWriter.") from e


import jax
import jax.numpy as jnp
import flashbax as fb
import optax
from flax import linen as nn
from flax import struct
from flax.training import train_state

def create_summary_writer(base_dir: Path) -> Tuple[SummaryWriter, Path]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(**{_SUMMARY_KWARG: str(run_dir)})
    return writer, run_dir

@chex.dataclass(frozen=True)
class Transition:
    observation: jnp.ndarray
    action: Dict[str, jnp.ndarray] # 1-5 or 6 w/ coordinates
    reward: float
    next_observation: jnp.ndarray
    done: bool

@struct.dataclass
class FlashbaxBufferState:
    buffer_state: Any
    rollout: Optional[Dict[str, Any]] = None

@chex.dataclass(frozen=True)
class LearnerState:
    params: Any
    target_params: Any
    opt_state: Any
    step: int
@chex.dataclass(frozen=True)
class SetupArtifacts:
    learner_state: LearnerState
    agent: tAgent
    learn_step: Callable([[LearnerState, FlashbaxBufferState, chex.PRNGKey], Tuple[LearnerState, FlashbaxBufferState, Dict[str, float]]])

GRID_SIZE = 64
NUM_CHANNELS = 16
RPB_CAPACITY = 1000
BATCH_SIZE = 32


def _frame_to_obs(frame: FrameData) -> jnp.ndarray:
    """Convert FrameData to a JAX ndarray observation with one-hot encoding for colors."""
    frame_array = jnp.array(frame.data, dtype=jnp.int32) # Shape: (G, G), values 0-15 
    frame_array_one_hot = jax.nn.one_hot(frame_array, num_classes=16)  # Shape: (G, G, C)
    return frame_array_one_hot.reshape(NUM_CHANNELS, GRID_SIZE, GRID_SIZE)  # Shape: (C, 64, 64)

class tAgent(Agent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start_time = time.time()

        self.MAX_STEPS = float('inf')

        self.base_dir, log_file = setup_experiment_directory()
        setup_logging_for_experiment(log_file)
        env_dir = get_environment_directory(self.base_dir, self.game_id)
        
        self.writer, run_dir = create_summary_writer(os.path.join(env_dir, 'tensorboard'))

        # --- 

        self.current_score = -1


        self.rpb = fb.make_flat_buffer(
            max_length=RPB_CAPACITY,
            min_length=2,
            sample_batch_size=BATCH_SIZE,
            add_sequences=False,
            add_batch_size=None
        )
        
        self.needs_init = True

def setup(*, rng: chex.PRNGKey, 
    rollout_length: int, 
    train_steps_per_update: int,
    initial_experience: Transition,    
    ) -> SetupArtifacts:

    pass




class FrameEncoder(nn.Module):
    # visual attention module for encoding frames
    def __init__(self, encode_dim: int = 128, latent_dim, input_shape: Tuple[int, int, int] = (16, 64, 64), num_heads: int = 4):
        super().__init__()
        self.encode_dim = encode_dim
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.num_heads = num_heads
        # one token per pixel (64*64=4096 tokens)
        self.vocab_size = 1000 # semantic vocab size for visual tokens
        self.token_embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.encode_dim)

        
class model(nn.Module):
    def __init__(self, latent_dim: int = 128, action_structure: Dict[str, Any] = {"discrete": 6, "click": (64, 64)}):
        super().__init__()
        
        self.frame_encoder = 
        
        for key, value in action_structure.items():
            if key=="discrete":
                self.action_heads[key] = nn.Dense(value)
            elif key=="click":
                self.action_heads[key] = nn.Dense(value[0]*value[1])
        self.action_heads: Dict[str, nn.Module] = {}
        self.latent_dim = latent_dim

    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        encoded = self.frame_encoder(x)  # Shape: (latent_dim,)
        actions = {}
        for key, head in self.action_heads.items():
            action_logits = head(encoded)
            if key == "click":
                action_logits = action_logits.reshape(-1, 64, 64)  # Reshape for click action
            actions[key] = action_logits
        return actions
            