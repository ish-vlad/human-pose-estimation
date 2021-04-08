import argparse

from pipeline.pipeline import Pipeline
from pipeline.detection import YOLODetection, FasterRCNNDetection, GTDetection
from pipeline.pose_estimation import BLAZEPOSEstimation, GTPoseEstimation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/home/ishvlad/datasets/LV-MHP-v2/val',
                        help='dir with origin MHP v2')
    parser.add_argument('--result_dir', type=str, default='data/',
                        help='dir for saving files (with subdirectories: outputs, numbers, pictures)', )
    parser.add_argument('--batch_size', type=int, help='Batch size (default=10)', default=10)
    parser.add_argument('--num_workers', type=int, help='Number of processes', default=10)
    parser.add_argument('--device', type=str, help='Device for models and data', default='cpu')

    parser.add_argument('--detection', choices=["YOLO", "GT", "FRCNN"], default="YOLO",
                        help='choose object detection method: "YOLO" -- for YOLOv5, '
                        '"GT" -- for ground truth, "FRCNN" -- for faster R-CNN')
    parser.add_argument('--pose', choices=["BP", "GT"], default="BP",
                        help='choose pose estimation method: "BP" -- for BlazePose, "GT" -- for ground truth')
    
    args = parser.parse_args()
    return args


def main(args):
    # init models
    if args.detection == 'YOLO':
        detection_model = YOLODetection(device=args.device)
    elif args.detection == 'GT':
        detection_model = GTDetection()
    elif args.detection == 'FRCNN':
        detection_model = FasterRCNNDetection(device=args.device)
    else:
        raise ValueError('Please use one of declared detection models')
    
    if args.pose == 'BP':
        pose_model = BLAZEPOSEstimation()
    elif args.pose == 'GT':
        pose_model = GTPoseEstimation()
    else:
        raise ValueError('Please use one of declared pose estimation models')

    # init pipeline
    world = Pipeline(args.dataset_dir, args.result_dir, num_workers=args.num_workers)
    result_file = world.val_inference(detection_model, pose_model, batch_size=args.batch_size)

    # calc COCO metric, measure time and plot samples
    world.calc_metric(result_file)
    world.measure_time(detection_model, args.batch_size)
    world.measure_time(pose_model, args.batch_size)
    world.plot_sample(result_file, save=True)
    
    
if __name__ == '__main__':
    args_ = parse_args()
    main(args_)
