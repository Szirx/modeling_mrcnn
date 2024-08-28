import torch


def evaluate_map(outputs, targets):
    preds = [{
        'boxes': output['boxes'],
        'scores': output['scores'],
        'labels': output['labels'],
    } for output in outputs]

    targets_ = [{
        'boxes': target['boxes'],
        'labels': target['labels']
    } for target in targets]

    return preds, targets_


def evaluate_map_mask(outputs, targets, threshold):
    preds = [{
        'masks': (output['masks'].squeeze(1) > threshold).to(torch.uint8),
        'scores': output['scores'],
        'labels': output['labels'],
    } for output in outputs]

    targets_ = [{
        'masks': target['masks'],
        'labels': target['labels']
    } for target in targets]

    return preds, targets_
