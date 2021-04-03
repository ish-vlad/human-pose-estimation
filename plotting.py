DATASET_DIR = '/home/ishvlad/datasets/LV-MHP-v2/'
INNER_DIR = 'val'

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

from scipy.io import loadmat


def plot_coco(coco, image_id=None, person_id=0):
    # select random if image_id is None
    if image_id is None:
        image_ids = coco.getImgIds(catIds=[1]);
        image_id = int(np.random.choice(image_ids))
    
    # Load image to show
    image_anno = coco.loadImgs(image_id)[0]
    image = imageio.imread(os.path.join(DATASET_DIR, INNER_DIR, 'images', image_anno['file_name']))

    # get image annotation
    anno_ids = coco.getAnnIds(imgIds=[image_id])
    annos = coco.loadAnns([anno_ids[person_id]])

    # plot image and annotation
    plt.axis('off')
    plt.imshow(image)
    coco.showAnns(annos, draw_bbox=True)
    
    return image_id, annos

def plot_mhp(image_id, person='person_0'):
    anno_path = os.path.join(DATASET_DIR, INNER_DIR, 'pose_annos')
    img_path = os.path.join(DATASET_DIR, INNER_DIR, 'images')

    name = str(image_id)

    anno = loadmat(os.path.join(anno_path, name + '.mat'))
    img = imageio.imread(os.path.join(img_path, name + '.jpg'))
    
    plt.axis('off')
    plt.imshow(img)
    plt.scatter(anno[person][:,0][:16], anno[person][:,1][:16], color='red')
    
    return image_id, anno