import os
import glob
import tqdm
import json
import imageio

from scipy.io import loadmat
from tqdm.autonotebook import tqdm

from joint_helper import COCO_KEYPOINTS, COCO_SKELETON, joints_mapping

DATASET_DIR = '/home/ishvlad/datasets/LV-MHP-v2/'
INNER_DIR = 'val'


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
    for path in tqdm(glob.glob(os.path.join(DATASET_DIR, INNER_DIR, 'images/*.jpg'))):
        name = path.split('/')[-1]
        image_id = int(name.split('.')[0])

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
