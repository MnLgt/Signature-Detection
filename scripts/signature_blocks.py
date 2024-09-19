import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torchvision import models
from torchvision import transforms as T
from torchvision.ops import nms
from typing import List, Any, Tuple

STATE_DICT = os.path.join(
    os.path.dirname(__file__), "..", "state_dicts", "signature_blocks_v14.pth"
)


def get_device():
    if torch.cuda.is_available():
        device = "cuda"

    # aten::hardsigmoid.out' is not currently implemented for the MPS device
    # setting fallback does not work either
    # elif torch.backends.mps.is_built():
    #     device = "mps"
    else:
        device = "cpu"
    return device


class ImgFactory:
    def serialize(self, img: Any) -> Any:
        serializer = self._get_serializer(img)
        return serializer(img)

    def _get_serializer(self, img: Any) -> Any:
        if isinstance(img, str):
            return self._serialize_string_to_image
        else:
            return self._serialize_image_to_image

    def _serialize_string_to_image(self, img):
        return Image.open(img)

    def _serialize_image_to_image(self, img):
        return img


class SignatureBlockModel(ImgFactory):
    def __init__(self, img, state_dict_path=STATE_DICT):
        self.state_dict_path = state_dict_path
        self.classes = {0: "NOTHING", 1: "SIGNED_BLOCK", 2: "UNSIGNED_BLOCK"}
        self.n_classes = len(self.classes)
        self.device = get_device()
        self.model = self._load_model()
        self.img = self.serialize(img)

        with torch.no_grad():
            self.model.eval()
            self.predictions = self._get_prediction()

    def _load_model(self):
        weights = models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        # change the head
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, self.n_classes
        )

        model.load_state_dict(
            torch.load(self.state_dict_path, map_location=self.device)
        )

        return model.to(self.device)

    def filter_overlap(self, predictions, iou_threshold=0.3):
        boxes = predictions[0]["boxes"]
        scores = predictions[0]["scores"]
        nms_filter = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)
        return nms_filter

    def filter_scores(self, predictions, score_thrs=0.94):
        nms_filter = self.filter_overlap(predictions)
        boxes = predictions[0]["boxes"]
        scores = predictions[0]["scores"]
        labels = predictions[0]["labels"]

        score_filter = scores[nms_filter] > score_thrs
        boxes = boxes[nms_filter][score_filter]
        scores = scores[nms_filter][score_filter]
        labels = labels[nms_filter][score_filter]
        return boxes, scores, labels

    def _get_prediction(self):
        transform = T.Compose([T.ToTensor()])
        img = transform(self.img)
        img = img.to(self.device)
        predictions = self.model([img])
        boxes, scores, labels = self.filter_scores(predictions)
        return [{"boxes": boxes, "scores": scores, "labels": labels}]

    def get_boxes(self):
        pred = self._get_prediction()
        boxes = pred[0]["boxes"].cpu().detach().numpy()
        int_boxes = []
        for box in boxes:
            box = [int(x) for x in box]
            int_boxes.append(box)
        return int_boxes

    def get_scores(self):
        pred = self._get_prediction()
        scores = pred[0]["scores"].cpu().detach().numpy()
        return scores

    def get_labels(self):
        pred = self._get_prediction()
        labels = pred[0]["labels"].cpu().detach().numpy()
        return labels

    def get_labels_names(self):
        pred = self._get_prediction()
        labels = pred[0]["labels"].cpu().detach().numpy()
        label_names = [self.classes[label] for label in labels]
        return label_names

    def _get_prediction_dict(self):
        boxes = self.get_boxes()
        scores = self.get_scores()
        labels = self.get_labels()
        return {"boxes": boxes, "scores": scores, "labels": labels}

    def _signature_crops(self, show=True):
        boxes = self.get_boxes()
        scores = self.get_scores()
        labels = self.get_labels()
        signature_crops = []
        for box, label, score in tuple(zip(boxes, labels, scores)):
            crop = self.extract_box(box)
            if show:
                crop = plt.imshow(crop)
            signature_crops.append(crop)
        return signature_crops

    def get_prediction(self):
        return self._get_prediction_dict()

    def get_image(self):
        return self.img

    def get_image_array(self):
        return np.array(self.img)

    def get_box_crops(self):
        boxes = self.get_boxes()
        box_crops = []
        for box in boxes:
            crop = self.img.crop(box)
            box_crops.append(crop)
        return box_crops

    def extract_box(self, box):
        xmin, ymin, xmax, ymax = box
        image = np.array(self.img)
        return image[ymin:ymax, xmin:xmax]

    def show_boxes(self):
        boxes = self.get_boxes()
        scores = self.get_scores()
        labels = self.get_labels()
        box_crops = []
        for box, label, score in tuple(zip(boxes, labels, scores)):
            print(f"Status: {self.classes[label]}")
            print(f"Score: {score}")
            crop = self.extract_box(box)
            plt.imshow(crop)
            plt.show()
            plt.close()
            box_crops.append(crop)
        return box_crops

    def draw_boxes(self):
        img = np.array(self.img)
        boxes = self.get_boxes()
        labels = self.get_labels()
        thickness = 2
        overlay = img.copy()
        for box, label in zip(boxes, labels):
            box = [int(x) for x in box]
            if label == 2:
                color = (0, 0, 255)  # red
            elif label == 1:
                color = (0, 255, 0)  # green
            cv2.rectangle(
                overlay, (box[0], box[1]), (box[2], box[3]), color, -1
            )  # Filled rectangle

        alpha = 0.4  # Transparency factor
        image_boxes = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Draw box outlines
        for box, label in zip(boxes, labels):
            box = [int(x) for x in box]
            if label == 2:
                color = (0, 0, 255)  # red
            elif label == 1:
                color = (0, 255, 0)  # green
            cv2.rectangle(
                image_boxes, (box[0], box[1]), (box[2], box[3]), color, thickness
            )

        return Image.fromarray(cv2.cvtColor(image_boxes, cv2.COLOR_BGR2RGB))
