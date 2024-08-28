import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def convert_onnx(
    model_path: str,
    height_dummy: int,
    width_dummy: int,
    num_classes: int,
    hidden_layer: int,
) -> None:
    """ Converting Mask-RCNN to ONNX format

    Args:
        model_path (str): path to model .pt or .bin
        height_dummy (int): height of dummy tensor
        width_dummy (int): width of dummy tensor
        num_classes (int): num classes of primal Mask-RCNN model
        hidden_layer (int): num of hidden layers in model
    """
    dummy_tensor = torch.randn(1, 3, height_dummy, width_dummy)

    model = maskrcnn_resnet50_fpn(
        num_classes=num_classes,
        pretrained=False,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )
    # Загрузка весов из файла .pt или .bin
    model.load_state_dict(torch.load(model_path))

    model_onnx_name = f'{model_path.rsplit(".", 1)[0]}.onnx'

    # Экспорт модели в формат ONNX
    torch.onnx.export(
        model,  # Модель
        dummy_tensor,  # Пример ввода (batch of images)
        model_onnx_name,  # Имя выходного файла
        opset_version=11,  # Выбор версии ONNX (11 и выше для поддержания всех операций Mask R-CNN)
        input_names=["input"],  # Имя входного тензора
        output_names=["boxes", "labels", "scores", "masks"],  # Имена выходных тензоров
        dynamic_axes={"input": {0: "batch_size"},  # Поддержка динамического размера батча
                    "boxes": {0: "batch_size"},
                    "labels": {0: "batch_size"},
                    "scores": {0: "batch_size"},
                    "masks": {0: "batch_size"}}
    )