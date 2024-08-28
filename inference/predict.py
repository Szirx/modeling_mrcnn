from PIL import Image
import torch
from torchvision import transforms


def predict(
    image_path: str,
    model: torch.nn.Module,
    device: str,
):
    # Загрузка и предобработка изображения
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Перевод модели в режим оценки
    model.eval()

    # Предсказание
    with torch.no_grad():
        predictions = model(image_tensor)
    
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_masks = predictions[0]['masks'].cpu().numpy()
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()

    return pred_boxes, pred_scores, pred_masks, pred_labels


def predict_batch(
    image_paths,
    model: torch.nn.Module,
    device: str,
):
    # Загрузка и предобработка изображений
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensors = [transform(Image.open(image_path).convert("RGB")) for image_path in image_paths]
    batch_tensor = torch.stack(image_tensors).to(device)

    # Перевод модели в режим оценки
    model.eval()

    # Предсказание
    with torch.no_grad():
        predictions = model(batch_tensor)

    
    # Постобработка предсказаний для всего батча
    batch_pred_boxes = []
    batch_pred_scores = []
    batch_pred_masks = []
    batch_pred_labels = []
    
    for prediction in predictions:
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        pred_masks = prediction['masks'].cpu().numpy()
        pred_labels = prediction['labels'].cpu().numpy()

        batch_pred_boxes.append(pred_boxes)
        batch_pred_scores.append(pred_scores)
        batch_pred_masks.append(pred_masks)
        batch_pred_labels.append(pred_labels)

    return batch_pred_boxes, batch_pred_scores, batch_pred_masks, batch_pred_labels