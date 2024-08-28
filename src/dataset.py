import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from config import DataConfig


class COCODataset(Dataset):
    def __init__(self, config: DataConfig, set_name: str, transform=None, resize=False):
        self._config = config
        self.root = self._config.data_path
        self.set_name = set_name
        self.transform = transform
        self.image_path = os.path.join(self.root, self.set_name)
        self.coco = COCO(os.path.join(self.image_path, '_annotations.coco.json'))
        
        self.should_resize = resize is not False
        if self.should_resize:
            self.height = int(self._config.processor_image_size * resize)
            self.width = int(self._config.processor_image_size * resize)
        else:
            self.height = self._config.processor_image_size
            self.width = self._config.processor_image_size
        
        self.image_info = []
        for img_id in self.coco.getImgIds():
            image_data = self.coco.loadImgs(img_id)[0]
            annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            self.image_info.append({
                'image_id': img_id,
                'file_name': image_data['file_name'],
                'annotations': annotations
            })
    
    def get_box(self, bbox):
        ''' Get the bounding box in COCO format '''
        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        ''' Get the image and the target'''
        
        info = self.image_info[idx]
        img_path = os.path.join(self.image_path, info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        if self.should_resize:
            img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        annotations = info['annotations']
        n_objects = len(annotations)
        
        masks = np.zeros((n_objects, self.height, self.width), dtype=np.uint8)
        boxes = []
        
        for i, annotation in enumerate(annotations):
            bbox = annotation['bbox']
            mask = self.coco.annToMask(annotation)
            
            if self.should_resize:
                mask = Image.fromarray(mask)
                mask = mask.resize((self.width, self.height), resample=Image.BILINEAR)
                mask = np.array(mask)
            
            masks[i, :, :] = mask > 0
            boxes.append(self.get_box(bbox))

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32).view(-1, 4)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = [ann['category_id'] for ann in annotations]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([info['image_id']])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.tensor([ann['iscrowd'] for ann in annotations], dtype=torch.int64)

        # This is the required target for the Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.image_info)
