from label_studio_ml.model import LabelStudioMLBase
import onnxruntime as ort
import numpy as np
import cv2
from ultralytics import YOLO
from uuid import uuid4

class MyYolov8(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(MyYolov8, self).__init__(**kwargs)
        self.model_path = 'C:/Users/Nathathai/Documents/chula_year4.1/Capstone/detection/runs/detect/yolov8s-epoch52/weights/best.pt'  
        self.img_local_path = 'C:/Users/Nathathai/Documents/chula_year4.1/Capstone/detection/datasets/football_player/valid/images'

        self.model = YOLO(self.model_path) 
        self.labels = ['Ball', 'Football-Player', 'traffic-light']
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4

    def predict(self, tasks, **kwargs):
        """ This is where inference happens: model returns 
            the list of predictions based on input list of tasks 
        """
        predictions = []
        for task in tasks:
            image_path = self.img_local_path +'/'+ task['data']['image'][25:]
            task_image_path = task['data']['image'][25:]
            print(f'task image: {task_image_path}')
            print(f'image path: {image_path}')
            output = self.model([image_path])[0]
            
            # Apply confidence threshold
            boxes = np.array(output.boxes.xyxy.tolist())
            scores = np.array(output.boxes.conf.tolist())
            classes = np.array(output.boxes.cls.tolist())
            
            mask = scores >= self.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]
            
            # Apply Non-Max Suppression (NMS)
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.confidence_threshold, self.iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = boxes[indices]
                scores = scores[indices]
                classes = classes[indices]

            img_shape = output.orig_shape
            width = img_shape[1]
            height = img_shape[0]
            results = []
            for box, class_id in zip(boxes, classes):
                x1, y1, x2, y2 = box
                result = {
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": x1 / width * 100,
                        "y": y1 / height * 100,
                        "width": (x2 - x1) / width * 100,
                        "height": (y2 - y1) / height * 100,
                        "rotation": 0,
                        "rectanglelabels": [
                            self.labels[int(class_id)]  # Map class_id to the label
                        ]
                    },
                    "id": str(uuid4())[:9],
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "origin": "myYoloV8"
                }
                results.append(result)

            prediction = {
                'created_ago': '0 hours', 
                'model_version': '1.0',
                'result': results
            }
            predictions.append(prediction)
        return predictions

    def fit(self, annotations, **kwargs):
        """ This is where training happens: train your model given list of annotations, 
            then returns dict with created links and resources
        """
        return {'path/to/created/model': 'my/model.bin'}
