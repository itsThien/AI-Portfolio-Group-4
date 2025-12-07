import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import utils

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset paths
    train_root = os.path.join(args.data, "train2017")
    ann_path = os.path.join(args.data, "annotations", "instances_train2017.json")

    dataset = CocoDetection(train_root, ann_path, transform=F.to_tensor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)

    model = get_model(num_classes=91)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.005,
                                momentum=0.9,
                                weight_decay=0.0005)

    for epoch in range(args.epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in loop:
            images = [img.to(device) for img in images]
            targets = utils.format_coco_targets(targets, images, device)

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "model.pth")
    print("Training complete. Model saved to model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="coco")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    main(args)
