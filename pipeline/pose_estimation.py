import time
import numpy as np
import mediapipe as mp

from tqdm.autonotebook import tqdm

from pipeline.pipeline import Model
from pipeline.detection import GTDetection
from plot_helper import crop_bbox
from joint_helper import get_mhp_keypoints_in_coco, COCO_KEYPOINTS


class PoseEstimation(Model):
    def time_pure_inference(self, dataloader, num_iter):
        detection_model = GTDetection()
        
        # EVAL loop
        predictions = []
        for imgs, image_ids, annotations in tqdm(dataloader):
            # DETECTION
            object_model_pred = detection_model.forward(imgs, annotations)
            
            # for every image
            for image_id, image, (bboxes, scores), image_anno in zip(image_ids, imgs, object_model_pred, annotations):
                # POSE ESTIMATION
                
                # log time
                start = time.clock()
                self.forward(bboxes, image, image_anno)
                end = time.clock()
                
                value = (end - start) * 1000
                predictions.extend([value/len(bboxes)] * len(bboxes))
            
            if len(predictions) >= num_iter:
                break
                
        return np.mean(predictions)


class BLAZEPOSEstimation(PoseEstimation):
    def __init__(self, confidence=0.5):
        self.confidence = confidence
        self.model = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=self.confidence)
        
    def forward(self, bboxes, image, annos=None):
        result = []
        # for every image crop
        for bbox in bboxes:
            cropped_image = crop_bbox(image, bbox)
            
            if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                result.append(np.zeros((len(COCO_KEYPOINTS), 3)))
                continue
            
            # model inference
            pose = self.model.process(cropped_image)
            
            if pose.pose_landmarks is not None:
                # get keypoint position on crop
                keypoints = get_mhp_keypoints_in_coco(pose, cropped_image.shape)
                # translate to crop position
                keypoints[keypoints[:, 2] != 0] += [bbox[0], bbox[1], 0]
            else:
                keypoints = np.zeros((len(COCO_KEYPOINTS), 3))
                
            result.append(keypoints)
        
        return result
        
    def __str__(self):
        return f'BP-{self.confidence}-pose'
        

class GTPoseEstimation(PoseEstimation):
    def __init__(self):
        pass
        
    def forward(self, bboxes, image, annos=None):
        return np.array([anno['keypoints'] for anno in annos])
        
    def __str__(self):
        return f'GT-pose'
