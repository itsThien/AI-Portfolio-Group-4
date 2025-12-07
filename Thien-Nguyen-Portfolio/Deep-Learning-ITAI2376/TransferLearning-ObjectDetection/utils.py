import torch

def format_coco_targets(targets, images, device):
    formatted = []

    for t, img in zip(targets, images):
        boxes = []
        labels = []
        for obj in t:
            x, y, w, h = obj["bbox"]
            boxes.append([x, y, x+w, y+h])
            labels.append(obj["category_id"])

        formatted.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
            "labels": torch.tensor(labels, dtype=torch.int64).to(device)
        })

    return formatted
