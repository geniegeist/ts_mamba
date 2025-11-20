from dataclasses import dataclass

@dataclass 
class Model:
    model: str
    n_layers: int
    d_model: int
    d_intermediate: int
    headdim: int
    d_state: int
    expand: int
    d_conv: int
    activation: str
    rms_norm: bool
    norm_epsilon: float
    residual_in_fp32: bool
    attn_layer_idx: list[int]
    attn_cfg: dict
    vocab_size: int
    tie_embeddings: bool

@dataclass
class DatasetConfig:
    _name_: str
    data: str
    meta: str

@dataclass
class Dataset:
    sampling: DatasetConfig
    train: DatasetConfig
    validation: DatasetConfig

@dataclass
class Wandb:
    project: str
    group: str
    name: str
    job_type: str
    mode: str
    id: str

@dataclass 
class Train:
    num_last_tokens: int

@dataclass
class Config:
    dataset: Dataset
    model: Model
    wandb: Wandb
    train: Train

    device: str
    context_window_in_days: int
    num_workers: int
    batch_size: int
    loss: str

    validate_at_start: bool
