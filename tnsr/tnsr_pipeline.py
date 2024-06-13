"""
Transparent Neural Surface Refinement implementation
"""

from dataclasses import dataclass, field
from typing import Type, Literal, Optional, Dict, Any
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

@dataclass
class TNSRPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: TNSRPipeline)
    """target class to instantiate"""


class TNSRPipeline(VanillaPipeline):
    config: TNSRPipelineConfig

    def __init__(
        self,
        config: TNSRPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

    def load_pipeline(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.load_state_dict(state)
