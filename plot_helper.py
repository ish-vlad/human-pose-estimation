import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

from scipy.io import loadmat
from matplotlib import patches

DATASET_DIR = '/home/ishvlad/datasets/LV-MHP-v2/val'
INNER_DIR = 'val'


def crop_bbox(image, bbox):
    idx = bbox.astype(int)
    return image[idx[1]:idx[3], idx[0]:idx[2]].astype(np.uint8)


def plot_image(image_id=None, data_path=DATASET_DIR):
    image = imageio.imread(os.path.join(data_path, f'images/{image_id}.jpg'))

    # plot image
    # plt.axis('off')
    plt.imshow(image)


def plot_coco(coco, image_id=None, data_path=DATASET_DIR):
    # select random if image_id is None
    if image_id is None:
        image_ids = coco.getImgIds(catIds=[1])
        image_id = int(np.random.choice(image_ids))
    
    # Load image to show
    image_anno = coco.loadImgs([image_id])[0]
    image = imageio.imread(os.path.join(data_path, 'images', image_anno['file_name']))

    # plot image and annotation
    # plt.axis('off')
    plt.imshow(image)
        
    # get image annotation
    annos = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
    coco.showAnns(annos, draw_bbox=True)
    
    return image_id, annos


def plot_mhp(image_id, person='person_0'):
    anno_path = os.path.join(DATASET_DIR, INNER_DIR, 'pose_annos')
    img_path = os.path.join(DATASET_DIR, INNER_DIR, 'images')

    name = str(image_id)

    anno = loadmat(os.path.join(anno_path, name + '.mat'))
    img = imageio.imread(os.path.join(img_path, name + '.jpg'))
    
    # plt.axis('off')
    plt.imshow(img)
    plt.scatter(anno[person][:,0][:16], anno[person][:,1][:16], color='red')
    
    return image_id, anno


def plot_bbox(bboxes):
    ax = plt.gca()
    
    for bb in bboxes:
        begin = bb[:2]
        end = bb[2:] - bb[:2] 
        
        rect = patches.Rectangle(begin, end[0], end[1], linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(rect)
