from typing import Optional
from dataset import COCODataset
from config import DataConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from augmentations import get_transform


class DataModule(pl.LightningDataModule):
  def __init__(self, config: DataConfig):
    super().__init__()
    self._config = config
    self.batch_size = self._config.batch_size
    self.n_workers = self._config.n_workers

  def setup(self, stage: Optional[str] = None) -> None:
    self.train_dataset = COCODataset(
      config=self._config,
      set_name='train',
      transform=get_transform(train=True),
      resize=True,
    )
    self.val_dataset = COCODataset(
      config=self._config,
      set_name='valid',
      transform=get_transform(train=False),
      resize=True,
    )

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        num_workers=self.n_workers,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )

  def val_dataloader(self):
    return DataLoader(
        self.val_dataset,
        batch_size=self.batch_size,
        num_workers=self.n_workers,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
    )
