import logging
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import hydra
from omegaconf import OmegaConf

from .train import train, record
from .train_utils import TopLevelConfig
from .triton_policy import replace_policy_mlp_with_triton

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def setup_distributed_training():
    """Configure distributed training with NCCL backend"""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    if world_size > 1:
        # Initialize process group
        log.info(f"Initializing distributed training: rank={rank}, world_size={world_size}")
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        # Set device
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        log.info(f"Using GPU: {torch.cuda.get_device_name(local_rank)}")
        
        return True, rank, local_rank
    
    return False, 0, 0

@hydra.main(config_path="configs", config_name="distributed", version_base="1.3")
def main(cfg: TopLevelConfig) -> None:
    is_distributed, rank, local_rank = setup_distributed_training()
    
    # Only the rank 0 process logs the config
    if rank == 0:
        log.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create unique experiment directory
    if rank == 0:
        from .train_utils import get_unique_experiment_dir
        exp_name = cfg.experiment_name or "distributed_exp"
        workspace = os.path.abspath(cfg.runtime.workspace_path)
        exp_base = os.path.join(workspace, exp_name)
        workspace_dir = get_unique_experiment_dir(exp_base)
        os.makedirs(workspace_dir, exist_ok=True)
        
        with open(os.path.join(workspace_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config=OmegaConf.structured(cfg), f=f.name)
        log.info(f"✅ Saved resolved config to {f.name}")
    else:
        # Other ranks wait for rank 0 to create the workspace
        if is_distributed:
            dist.barrier()
        # Get the workspace directory created by rank 0
        from glob import glob
        exp_name = cfg.experiment_name or "distributed_exp"
        workspace = os.path.abspath(cfg.runtime.workspace_path)
        exp_base = os.path.join(workspace, exp_name)
        dirs = sorted(glob(f"{exp_base}_*"))
        workspace_dir = dirs[-1] if dirs else exp_base
    
    # Update config with distributed settings
    cfg.training.device = f"cuda:{local_rank}"
    
    # Train the model
    result = train(cfg, workspace_dir)
    
    # Apply Triton optimization to policy network
    replace_policy_mlp_with_triton(result.model)
    
    # Only rank 0 records results to MLflow
    if rank == 0:
        record(cfg, result)
    
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
