import torch
from config import Config
from train_utils import load_object
import pytorch_lightning as pl
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from metrics import evaluate_map, evaluate_map_mask
import torchmetrics


class MaskRCNNLightning(pl.LightningModule):
    def __init__(self, config: Config):
        super(MaskRCNNLightning, self).__init__()
        self._config = config
        self.model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1,
            box_detection_per_image=self._config.box_detection_per_img,
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            self._config.num_classes,
        )

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = self._config.hidden_layer

        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            self._config.num_classes,
        )
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Инициализация метрик
        self.bbox_metric = torchmetrics.detection.MeanAveragePrecision(iou_type='bbox')
        self.seg_metric = torchmetrics.detection.MeanAveragePrecision(iou_type='segm')
        self.train_loss = []

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.forward(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.train_loss.append(losses)
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        preds_map_bbox, targets_map_bbox = evaluate_map(outputs, targets)
        preds_map_mask, targets_map_mask = evaluate_map_mask(outputs, targets, self._config.mask_threshold)
        # Логирование метрики Mean Average Precision
        self.bbox_metric.update(preds_map_bbox, targets_map_bbox)
        self.seg_metric.update(preds_map_mask, targets_map_mask)

        return outputs
    
    def predict_step(self, batch):
        images, _ = batch
        outputs = self.forward(images)
        return outputs
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_loss).mean()
        self.log('train_loss', avg_loss, on_epoch=True)
    
    def on_validation_epoch_end(self):
        for key, value in self.bbox_metric.compute().items():
            if value.numel() > 0:
                self.log(f"val_bbox_{key}", value, on_epoch=True)
        for key, value in self.seg_metric.compute().items():
            if value.numel() > 0:
                self.log(f"val_mask_{key}", value, on_epoch=True)

        self.bbox_metric.reset()
        self.seg_metric.reset()

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self.model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }