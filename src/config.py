from omegaconf import OmegaConf
from pydantic import BaseModel


class DataConfig(BaseModel):
    data_path: str
    batch_size: int
    n_workers: int
    image_size: int
    processor_image_size: int
    dataset_name: str

class MlflowConfig(BaseModel):
    run_name: str
    experiment_name: str
    tracking_uri: str

class Config(BaseModel):
    project_name: str
    data_config: DataConfig
    mlflow_config: MlflowConfig
    n_epochs: int
    num_classes: int
    accelerator: str
    devices: str
    monitor_metric: str
    task: str
    box_detection_per_img: int
    hidden_layer: int
    mask_threshold: float
    min_score: float
    load_from_ckpt: str
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)