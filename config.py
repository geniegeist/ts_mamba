from dataclasses import dataclass

# -------------------- Model -------------------- #

@dataclass 
class Model:
    # either 'simple' or 'token'
    architecture: str

    # dimension of hidden layer
    d_model: int
    # the total number of mixer blocks
    n_layer: int 
    d_intermediate: int
    rms_norm: bool
    norm_epsilon: float
    residual_in_fp32: bool
    fused_add_norm: bool
    dropout: float

    # block config
    ssm_cfg: dict
    attn_layer_idx: list[int]
    attn_cfg: dict

    # init
    use_llm_init: bool

    # for default sequence models
    d_output: int

    # for token sequence models
    vocab_size: int
    # pad the vocab_size so that it is multiple of pad_vocab_multiple
    # use 1 to not pad the vocab_size
    pad_vocab_multiple: int
    # whether embedding and output projection layer should share the same weights
    tie_embeddings: bool


# -------------------- Dataset -------------------- #

@dataclass
class DatasetConfig:
    data: str
    meta: str
    stride: int

@dataclass
class Dataset:
    sample: DatasetConfig
    train: DatasetConfig
    validate: DatasetConfig
    use_covariates: bool
    context_window_in_days: int


# -------------------- Wandb -------------------- #

@dataclass
class Wandb:
    project: str
    group: str
    name: str
    job_type: str
    mode: str
    id: str

# -------------------- Train -------------------- #

@dataclass
class Loss:
    # mse | l1 | cross_entropy | quantile
    name: str

    # quantile config
    quantiles: list[float]

@dataclass 
class Train:
    loss: Loss

    num_iterations: int
    warmup_ratio: float
    total_batch_size: int # how many tokens to process per iteration
    device_batch_size: int # set as high as possible until oom, measured in (context_length, d_input)
    lr: float
    final_lr_frac: float
    weight_decay: float
    grad_clip: float



@dataclass
class Validate:
    # point, quantile, llm
    type: str
    batch_size: int
    validate_every: int

    quantile_10_idx: int
    quantile_90_idx: int
    quantile_point_forecast_idx: int

@dataclass
class Sample:
    # point, quantile, llm
    type: str
    batch_size: int
    sample_every: int

    quantile_10_idx: int
    quantile_90_idx: int
    quantile_point_forecast_idx: int

@dataclass
class Benchmark:
    checkpoint: str
    test_file: str
    test_meta: str
    time_zone: str

@dataclass
class Config:
    dataset: Dataset
    model: Model
    wandb: Wandb
    train: Train
    validate: Validate
    sample: Sample

    benchmark: Benchmark

    model_tag: str

    # checkpoint
    resume_from_step: int
    checkpoint_dir: str
    load_optimizer: bool
    save_every: int

    num_workers: int
    device: str
