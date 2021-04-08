import torch
import torchvision
import time
import numpy as np

from tqdm.autonotebook import tqdm

from pipeline.pipeline import Model


class Detection(Model):
    def __init__(self, confidence, return_numpy, device):
        self.device = device
        self.confidence = confidence
        self.return_numpy = return_numpy
        
        self.dataset_args = {}
        
    def _check_return_type(self, obj):
        if self.return_numpy:
            if type(obj) != np.ndarray:
                return obj.cpu().numpy()
            return obj
        else:
            torch.as_tensor(obj)
            
    def time_pure_inference(self, dataloader, num_iter):
        # EVAL loop
        predictions = []
        for imgs, image_ids, annotations in tqdm(dataloader):
            # log time
            start = time.time()
            self.forward(imgs, annotations)
            end = time.time()
            
            value = (end - start) * 1000
            predictions.extend([value/len(imgs)] * len(imgs))
            
            if len(predictions) >= num_iter:
                break
        
        return np.mean(predictions)


class YOLODetection(Detection):
    def __init__(self, confidence=0.5, return_numpy=True, device='cpu'):
        super().__init__(confidence, return_numpy, device)
        
        # download model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
        self.model.to(device)
    
        # arguments for input data
        self.dataset_args = {
            'is_int': True 
        }
    
    def forward(self, images, annos=None):
        """
        return: list of (bboxes(xyxy), scores([0,1]))
        """
        # model forward
        pred = self.model(list(images), size=640).pred
        
        result = []
        # get bboxes and scores
        for detection in pred:
            detection = self._check_return_type(detection.detach())
            
            # take only bounding boxes (x1, y1, x2, y2) with people (category = 0) with high confidence
            detection = detection[(detection[:, -1] == 0) & (detection[:, -2] > self.confidence)]
            bboxes, scores = detection[:, :4], detection[:, 4]
            
            result.append((bboxes, scores))
            
        return result
    
    def __str__(self):
        return f'YOLO-{self.confidence}-detection'
 
    
class FasterRCNNDetection(Detection):
    def __init__(self, confidence=0.7, return_numpy=True, device='cpu'):
        super().__init__(confidence, return_numpy, device)
        
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        self.model.eval().double().to(device)
    
        self.dataset_args = {'is_int': False}
    
    def forward(self, images, annos=None):
        """
        return: list of (bboxes(xyxy), scores([0,1]))
        """
        
        # model forward
        pred = self.model([torch.as_tensor(x).to(self.device).permute(2, 0, 1) for x in images])
        
        result = []
        # get bboxes and scores
        for detection in pred:
            # take only people with confidence > 0.7
            idx = (detection['labels'] == 1) & (detection['scores'] > self.confidence)

            bboxes = detection['boxes'][idx].detach()
            scores = detection['scores'][idx].detach()
            
            bboxes = self._check_return_type(bboxes)
            scores = self._check_return_type(scores)
            result.append((bboxes, scores))
            
        return result
    
    def __str__(self):
        return f'FRCNN-{self.confidence}-detection'
    
    
class GTDetection(Detection):
    def __init__(self, return_numpy=True):
        super().__init__(None, return_numpy, None)
        
        self.dataset_args = {'is_int': True}
        
    def forward(self, images, annos=None):
        """
        return: list of (bboxes(xyxy), scores([0,1]))
        """
        
        result = []
        # get GT bboxes
        for image_anno in annos:
            
            bboxes = np.array([anno['bbox'] for anno in image_anno if sum(anno['bbox']) != 0])
            scores = np.ones(len(bboxes))
            
            # if empty add empty set
            if len(bboxes) == 0:
                bboxes, scores = np.zeros((1, 4)), np.zeros(1)
                
            # xyhw -> xyxy
            bboxes[:, 2:4] += bboxes[:, :2]
        
            bboxes = self._check_return_type(bboxes)
            scores = self._check_return_type(scores)
            result.append((bboxes, scores))
            
        return result
    
    def __str__(self):
        return f'GT-detection'
