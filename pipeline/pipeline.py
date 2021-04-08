import os
import json
import matplotlib.pyplot as plt
import numpy as np

from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pipeline.dataloader import MHPDataset, collate_fn
from plot_helper import plot_image, plot_coco

class Model(ABC):
    @abstractmethod
    def time_pure_inference(self, dataloader, num_iter):
        raise NotImplementedError()
    
    @abstractmethod
    def forward(self, images, annos=None):
        raise NotImplementedError()


class Pipeline:
    def __init__(self, coco_path='/home/ishvlad/datasets/LV-MHP-v2/val',
                 result_path='../data/', num_workers=10, device=None):
        self.coco_path = coco_path
        self.result_path = result_path
        self.num_workers = num_workers
        self.device = device
        
        # load dataset
        self.coco = COCO(os.path.join(coco_path, 'COCO-annotation.json'))
        self.dataset, self.dataloader = None, None
        
    def _get_COCO_output(self, image_id, bboxes, scores, keypoints):
        result = []
        # for every image's bbox
        for bbox, score, kps in zip(bboxes, scores, keypoints):
            # xyxy -> xyhw
            bbox[2:4] = bbox[2:4] - bbox[:2]
            
            result.append({
                'category_id': 1,  # person
                'iscrowd': 0,
                'image_id': int(image_id),
                'bbox': bbox.astype(float).tolist(),
                'score': float(score), 
                'keypoints': kps.flatten().tolist()
            })
        
        return result 
    
    @staticmethod
    def _save(predictions, output_name):
        with open(output_name, 'w+') as f:
            json.dump(predictions, f)
            
    def _init_dataloader(self, model, batch_size, override=True):
        if self.dataloader is not None and not override:
            return self.dataloader
        
        if 'dataset_args' in model.__dict__:
            self.dataset = MHPDataset(self.coco_path, self.coco, **model.dataset_args)
        else:
            self.dataset = MHPDataset(self.coco_path, self.coco)
            
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers,
                                     collate_fn=collate_fn)
        return self.dataloader

    def val_inference(self, detection_model, pose_model, batch_size=10):
        # load dataset and dataloader
        self._init_dataloader(detection_model, batch_size)

        # EVAL loop
        predictions = []
        for imgs, image_ids, annotations in tqdm(self.dataloader):
            # DETECTION
            object_model_pred = detection_model.forward(imgs, annotations)

            # for every image
            for image_id, image, (bboxes, scores), image_anno in zip(image_ids, imgs, object_model_pred, annotations):
                # POSE ESTIMATION
                keypoints = pose_model.forward(bboxes, image, image_anno)

                # save result
                pred = self._get_COCO_output(image_id, bboxes, scores, keypoints)
                predictions.extend(pred)
        
        output_name = os.path.join(self.result_path, f'outputs/val_{detection_model}_{pose_model}.json')
        self._save(predictions, output_name)
        print(f'Saved result to {output_name}')
        return output_name
        
    def calc_metric(self, result_file=None):
        if result_file is None:
            result_file = self.result_file
        
        # read
        coco_pred = self.coco.loadRes(result_file)
        metrics = {}
    
        for key in ['bbox', 'keypoints']:
            # calculate COCO metric
            coco_eval = COCOeval(self.coco, coco_pred, key)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metrics[key] = coco_eval.stats.astype(float).tolist()
        
        # save file in 'numbers' dir
        name_words = result_file.split('/')
        name_words[-2] = 'numbers'
        result_file = '/'.join(name_words)
        
        self._save(metrics, result_file)
        return result_file

    def plot_sample(self, result_file=None, num_in_a_row=4, save=False):
        if result_file is None:
            result_file = self.result_file
        
        # read
        coco_pred = self.coco.loadRes(result_file)
        font_size = 18
        
        plt.figure(figsize=(25,15))
        for i, image_id in enumerate(np.random.choice(coco_pred.getImgIds(), num_in_a_row)):

            plt.subplot(3, num_in_a_row, i + 1)
            plot_image(image_id)
            if i == 0:
                plt.ylabel('Origin images', fontsize=font_size)

            plt.subplot(3, num_in_a_row, num_in_a_row + i + 1)
            plot_coco(self.coco, image_id)
            if i == 0:
                plt.ylabel('GT bbox + GT pose', fontsize=font_size)

            plt.subplot(3, num_in_a_row, 2 * num_in_a_row + i + 1)
            plot_coco(coco_pred, image_id)
            if i == 0:
                plt.ylabel(' + '.join(result_file.split('/')[-1][:-len('.json')].split('_')[1:]), fontsize=font_size)
        
        # save file in 'pictures' dir
        name_words = result_file.split('/')
        name_words[-2] = 'pictures'
        result_file = '/'.join(name_words)
        result_file = result_file[:-len('.json')] + '.png'
        
        if save:
            plt.savefig(result_file, dpi=300)

    def measure_time(self, model, batch_size=10, num_iter=200):
        # load dataloader
        self._init_dataloader(model, batch_size)
        
        # calculate time for 1000 iteration and return value for 1 iteration
        time_in_ms = model.time_pure_inference(self.dataloader, num_iter)
        
        # load dictionary for times
        result_file = os.path.join(self.result_path, 'numbers/time.json')
        times_dict = {}
        if os.path.exists(result_file):
            with open(result_file, 'r+') as f:
                times_dict = json.load(f)
        
        # save time
        times_dict[str(model)] = time_in_ms
        self._save(times_dict, result_file)
        print(f'Saved in {result_file}. Time: {time_in_ms} ms per one sample')
        
        return time_in_ms
