import os
import argparse
from clearml import Task
from clearml_log import clearml_logging

from config import Config
from datamodule import DataModule

from lightning_module import MaskRCNNLightning
from predict_callback import PredictAfterValidationCallback

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()

def train(config: Config, config_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.devices

    task = Task.init(project_name=config.project_name, task_name=config.task)
    logger = task.get_logger()
    clearml_logging(config, logger)

    checkpoint_callback = ModelCheckpoint(
        filename='mrcnn-{epoch}-{step}-{train_loss:.4f}',
        save_top_k=1,
        every_n_epochs=1,
        verbose=True,
        mode='max',
        monitor=config.monitor_metric,
    )

    # early_stopping_callback = EarlyStopping(monitor=config.monitor_metric, patience=20, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    datamodule = DataModule(config.data_config)

    torch.set_float32_matmul_precision('high')

    trainer = pl.Trainer(
        max_epochs=config.n_epochs, 
        accelerator=config.accelerator, 
        devices=1, 
        log_every_n_steps=1,
        callbacks=[
            PredictAfterValidationCallback(logger=logger, config=config),
            checkpoint_callback,
            lr_monitor,
        ],
    )
    model = MaskRCNNLightning(config)

    task.upload_artifact(
        name='config_file',
        artifact_object=config_file,
    )

    trainer.fit(model=model, datamodule=datamodule)
    # trained_model = MaskRCNNLightning.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # model_weights_path = 'weights/model_weights.pt'
    # torch.save(trained_model.model.state_dict(), model_weights_path)


if __name__ == "__main__":
    args = arg_parse()
    pl.seed_everything(42, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config, args.config_file)
    