import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import argparse

def load_model(weights="model.pth"):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 91)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model

def draw_boxes(image, boxes, scores, labels, threshold=0.5):
    img = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold: continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{label}:{score:.2f}",
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,0), 1)
    return img

def main(args):
    model = load_model(args.weights)

    image = Image.open(args.image).convert("RGB")
    input_tensor = F.to_tensor(image)

    with torch.no_grad():
        pred = model([input_tensor])

    pred = pred[0]
    result = draw_boxes(
        image=np.array(image),
        boxes=pred["boxes"].numpy(),
        scores=pred["scores"].numpy(),
        labels=pred["labels"].numpy(),
    )

    cv2.imwrite(args.output, result)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.jpg")
    parser.add_argument("--weights", type=str, default="model.pth")
    args = parser.parse_args()
    main(args)
