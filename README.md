# human-pose-estimation

Task: **2D/3D Human Single/Multi Pose Estimation.** 

Dataset: Multi-Human Parsing V2

Notation: COCO Keypoint Detection Task

Future work: 2D single -> 3D single; 2D single -> 2D multi-pose

    Estimation сеть. Нотация кейпоинтов и критерии качества должны
    соответствовать COCO Keypoint Detection Task. Упор должен быть на
    максимальную легковесность сети (с целью последующего запуска её на
    мобильных устройствах). Код оформить в виде репозитория на GitHub.
    Отчёт по работе - отдельным файлом.

    Предложить рекомендации, как на базе разработанной сети реализовать 3D
    Human Pose Estimation и 2D Human Multi-pose Estimation, либо объяснить,
    почему это нельзя сделать.

## Запуск сети

    usage: main.py [-h] [--dataset_dir DATASET_DIR] [--result_dir RESULT_DIR]
               [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
               [--device DEVICE] [--detection {YOLO,GT,FRCNN}]
               [--pose {BP,GT}]
               
    optional arguments:
          -h, --help            show this help message and exit
          --dataset_dir DATASET_DIR
                                dir with origin MHP v2
          --result_dir RESULT_DIR
                                dir for saving files (with subdirectories: outputs,
                                numbers, pictures)
          --batch_size BATCH_SIZE
                                Batch size (default=10)
          --num_workers NUM_WORKERS
                                Number of processes
          --device DEVICE       Device for models and data
          --detection {YOLO,GT,FRCNN}
                                choose object detection method: "YOLO" -- for YOLOv5,
                                "GT" -- for ground truth, "FRCNN" -- for faster R-CNN
          --pose {BP,GT}        choose pose estimation method: "BP" -- for BlazePose,
                                "GT" -- for ground truth 
