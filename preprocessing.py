import os
import glob
import tqdm
import json
import imageio
import numpy as np

from scipy.io import loadmat
from tqdm.autonotebook import tqdm

DATASET_DIR = '/home/ishvlad/datasets/LV-MHP-v2/'
INNER_DIR = 'val'

COCO_KEYPOINTS = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]
COCO_SKELETON = [
    [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
    [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
]


def joints_mapping(joints):
    # preprocess joints: adapt visibility flag
    mpii_joints = [[x, y, 2] if v == 0 and x > 0 and y > 0 else [0, 0, 0] for x, y, v in joints]

    coco_joints = np.zeros((len(COCO_KEYPOINTS), 3))

    # HEAD (5): nose, eyes, ears (ALL ZEROS)
    # BODY (7): thorax/upper-neck, shoulders, elbows, wrists
    coco_joints[5:11:2] = mpii_joints[13:16] # left
    coco_joints[6:12:2] = mpii_joints[10:13][::-1] # right
    
    # LEGS (6): hips, knees, ankles
    coco_joints[11:17:2] = mpii_joints[3:6] # left
    coco_joints[12:18:2] = mpii_joints[:3][::-1] # right
    
    # transfer instance bbox
    person_bbox = np.array(mpii_joints[18][:2] + mpii_joints[19][:2])
    person_bbox[2:] -= mpii_joints[18][:2]
    
    # transfer face bbox
    face_bbox = np.array(mpii_joints[16][:2] + mpii_joints[17][:2])
    face_bbox[2:] -= mpii_joints[16][:2]

    return coco_joints, person_bbox, face_bbox


def main():
    result_json = {
        "info": {
            "description": "Transfered MHP dataset to COCO notation",
            "version": "1.0",
            "year": 2021,
            "date_created": "2021/04/03"
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc/2.0/",
                "id": 2,
                "name": "Attribution-NonCommercial License"
            }
        ],
        "categories": [{
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": COCO_KEYPOINTS,
            "skeleton": COCO_SKELETON
        }],
        'images': [],
        "annotations": []
    }
    
    person_id = 0
    for path in tqdm(glob.glob(os.path.join(DATASET_DIR, INNER_DIR, 'images/*.jpg'))[:100]):
        name = path.split('/')[-1]
        image_id = int(name.split('.')[0])

        # TODO: do not need to read the whole image, size only
        try:
            img = imageio.imread(path)
        except (SyntaxError, ValueError):
            continue

        if len(img.shape) == 3:
            height, width, _ = imageio.imread(path).shape
        else:
            height, width = imageio.imread(path).shape

        # ADD to global image list
        result_json['images'].append({
            'dir': INNER_DIR,
            'file_name': name,
            'height': height,
            'width': width,
            'id': image_id
        })

        # change directory: images -> pose_annos
        buffer = path.split('/')
        buffer[-2] = 'pose_annos'
        anno_path = '/'.join(buffer)[:-4] + '.mat'

        # load annotation
        anno = loadmat(anno_path)

        # transfer all persons
        for p_id in anno:
            if not p_id.startswith('person_'):
                continue

            joints = anno[p_id]
            # transfer joints notation to COCO
            coco_joints, person_bbox, face_bbox = joints_mapping(joints)
            
            # skip pose if there are no joints inside  
            if coco_joints.sum() == 0:
                continue
            
            # ADD to global annotation list
            result_json['annotations'].append({
                "num_keypoints": len(COCO_KEYPOINTS),
                "keypoints": coco_joints.flatten().astype(int).tolist(),
                "image_id": image_id,
                "bbox": person_bbox.tolist(),
                "face_bbox": face_bbox.tolist(),
                "category_id": 1,
                'iscrowd': 0,
                'area': float(person_bbox[2] * person_bbox[3]),
                "id": person_id
            })
            person_id += 1
            
    with open(os.path.join(DATASET_DIR, INNER_DIR, 'COCO-annotation.json'), 'w+') as f:
        json.dump(result_json, f)
        

if __name__ == '__main__':
    main()