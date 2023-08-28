from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger


@dataclass
class RenderConfig:
    """ Parameters for the Mesh Renderer """
    # Render size for training
    train_grid_H: int = 1200 # Note: please set to at least larger than 1000 to contain the whole object
    train_grid_W: int = 1600
    # Render size for evaluating
    eval_grid_H: int = 1200
    eval_grid_W: int = 1600
    # training camera radius range
    radius_range: Tuple[float, float] = (20.0, 20.5)# (1.0, 1.5)
    # Set [0,angle_overhead] as the overhead region
    angle_overhead: float = 30
    # Define the front angle region
    angle_front: float = 70
    # Which NeRF backbone to use
    backbone: str = 'texture-mesh'

@dataclass
class GuideConfig:
    """ Parameters defining the guidance """
    # Guiding text prompt
    text: str
    # The mesh to paint
    shape_path: str = '~/latent-nerf/shapes/animal.obj'
    # Append direction to text prompts
    append_direction: bool = False # True
    # A Textual-Inversion concept to use
    concept_name: Optional[str] = None
    # A huggingface diffusion model to use
    diffusion_name: str = 'CompVis/stable-diffusion-v1-4'
    # Scale of mesh in 1x1x1 cube
    shape_scale: float = 0.6
    # height of mesh
    dy: float = 0.25
    # texture image resolution
    texture_resolution=128
    # texture mapping interpolation mode from texture image, options: 'nearest', 'bilinear', 'bicubic'
    texture_interpolation_mode: str= 'nearest'


@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    # Seed for experiment
    seed: int = 0
    # Total iters
    iters: int = 1000# 5000
    # Learning rate
    lr: float = 1e-5
    # Resume from checkpoint
    resume: bool = False
    # Load existing model
    ckpt: Optional[str] = None
    # Use random view or random image view in dataset
    use_neus_view: bool = True

@dataclass
class GlobalConfig:
    gpu: str = 'cuda:0'
    half: bool = True
    mode: str = 'latent_paint' # 'latent_paint' for latent paint, 'train' for training Geo-NeuS, 'validate_mesh' for validating mesh, same as 'validate_image', 'eval_image' or 'interpolate'
    latent: bool = True

@dataclass
class LogConfig:
    """ Parameters for logging and saving """
    # Experiment name
    exp_name: str
    # Experiment output dir
    exp_root: Path = Path('experiments/')
    # How many steps between save step
    save_interval: int = 100
    # Run only test
    eval_only: bool = False
    # Number of angles to sample for eval during training
    eval_size: int = 10
    # Number of angles to sample for eval after training
    full_eval_size: int = 100
    # Export a mesh
    save_mesh: bool = True
    # Number of past checkpoints to keep
    max_keep_ckpts: int = 2
    # marching cube threshold
    mcube_threshold: float = 0.0

    @property
    def exp_dir(self) -> Path:
        return self.exp_root / self.exp_name

@dataclass
class NeusConfig:
    """Parameters for NeuS"""
    neus_cfg_path: str = './confs/womask.conf'
    load_from_neus: bool = True # load SDF network from NeuS or not
    neus_ckpt_path: str = 'neus_ckpt/ckpt_300000.pth'
    use_white_bkgd: bool = True
    mcube_threshold: float = 0.0
    is_continue: bool = False
    checkpoint: int = 0
    case: str = 'scan37'
    suffix: str = ''
    dilation: int = 15

@dataclass
class TrainConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)
    neus: NeusConfig = field(default_factory=NeusConfig)
    global_setting: GlobalConfig = field(default_factory=GlobalConfig)

    def __post_init__(self):
        if self.log.eval_only and (self.optim.ckpt is None and not self.optim.resume):
            logger.warning('NOTICE! log.eval_only=True, but no checkpoint was chosen -> Manually setting optim.resume to True')
            self.optim.resume = True

