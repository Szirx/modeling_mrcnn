from PIL import Image
import torch
import onnxruntime as ort
from torchvision import transforms


def preprocess_image(image_path, device=None):
    """Загрузка и предобработка изображения."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.255)),
    ])
    image_tensor = transform(image).unsqueeze(0)
    if device:
        image_tensor = image_tensor.to(device)
    return image_tensor


def postprocess_predictions(predictions):
    """Постобработка предсказаний."""
    pred_boxes = predictions['boxes'].cpu().numpy()
    pred_scores = predictions['scores'].cpu().numpy()
    pred_masks = predictions['masks'].cpu().numpy()
    pred_labels = predictions['labels'].cpu().numpy()

    return pred_boxes, pred_scores, pred_masks, pred_labels


def predict(image_path: str, model: torch.nn.Module, device: str):
    """Предсказание для одного изображения с использованием PyTorch модели."""
    image_tensor = preprocess_image(image_path, device)

    # Перевод модели в режим оценки
    model.eval()

    # Предсказание
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    return postprocess_predictions(predictions)


def predict_batch(image_paths, model: torch.nn.Module, device: str):
    """Предсказание для батча изображений с использованием PyTorch модели."""
    image_tensors = [preprocess_image(image_path) for image_path in image_paths]
    batch_tensor = torch.cat(image_tensors).to(device)

    # Перевод модели в режим оценки
    model.eval()

    # Предсказание
    with torch.no_grad():
        predictions = model(batch_tensor)

    # Постобработка предсказаний для всего батча
    batch_results = [postprocess_predictions(pred) for pred in predictions]
    batch_pred_boxes, batch_pred_scores, batch_pred_masks, batch_pred_labels = zip(*batch_results)

    return batch_pred_boxes, batch_pred_scores, batch_pred_masks, batch_pred_labels


def predict_onnx(onnx_model_path, image_path):
    """Предсказание для одного изображения с использованием ONNX модели."""
    ort_session = ort.InferenceSession(onnx_model_path)

    # Предобработка изображения
    input_tensor = preprocess_image(image_path).numpy()

    # Выполнение предсказания
    outputs = ort_session.run(None, {"input": input_tensor})

    # Извлечение результатов
    pred_boxes = outputs[0]
    pred_labels = outputs[1]
    pred_scores = outputs[2]
    pred_masks = outputs[3]

    return pred_boxes, pred_scores, pred_masks, pred_labels
