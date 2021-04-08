import os
import cv2
import imageio
import numpy as np

from torch.utils import data
from pycocotools.coco import COCO


class MHPDataset(data.Dataset):
    def __init__(self, data_dir, coco=None, is_int=True):
        self.data_dir = data_dir
        self.is_int = is_int
        
        # load annotation for the whole dataset
        if coco is None:
            self.coco = COCO(os.path.join(data_dir, 'COCO-annotation.json'))
        else:
            self.coco = coco

    def __getitem__(self, index):
        # get image anno from COCO
        image_index = self.coco.getImgIds()[index]
        image_info = self.coco.loadImgs(image_index)[0]

        # load image
        image = imageio.imread(os.path.join(self.data_dir, 'images', image_info['file_name']))
        
        if not self.is_int:
            image = image / 256
        
        # if grayscaled -> move to color
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # transform: annotations -> target
        anno = self.coco.getAnnIds(imgIds=image_index)
        anno = self.coco.loadAnns(anno)
        return np.array(image), image_index, anno
    
    def __len__(self):
        return len(self.coco.imgs)


def collate_fn(batch):
    return tuple(zip(*batch))
