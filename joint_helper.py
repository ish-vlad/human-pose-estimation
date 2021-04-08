import numpy as np
import mediapipe as mp

MP_HOLISTIC = mp.solutions.holistic
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
]


def get_mhp_keypoints_in_coco(pose, image_shape):
    coco_joints = np.zeros((len(COCO_KEYPOINTS), 3))
    joints_enum = [
        MP_HOLISTIC.PoseLandmark.LEFT_SHOULDER, MP_HOLISTIC.PoseLandmark.RIGHT_SHOULDER,
        MP_HOLISTIC.PoseLandmark.LEFT_ELBOW, MP_HOLISTIC.PoseLandmark.RIGHT_ELBOW,
        MP_HOLISTIC.PoseLandmark.LEFT_WRIST, MP_HOLISTIC.PoseLandmark.RIGHT_WRIST,
        MP_HOLISTIC.PoseLandmark.LEFT_HIP, MP_HOLISTIC.PoseLandmark.RIGHT_HIP,
        MP_HOLISTIC.PoseLandmark.LEFT_KNEE, MP_HOLISTIC.PoseLandmark.RIGHT_KNEE,
        MP_HOLISTIC.PoseLandmark.LEFT_ANKLE, MP_HOLISTIC.PoseLandmark.RIGHT_ANKLE
    ]

    for idx, joint_name in zip(range(5, len(COCO_KEYPOINTS)), joints_enum):
        joint = pose.pose_landmarks.landmark[joint_name]

        if joint.visibility > 0.7:
            coco_joints[idx] = [joint.x * image_shape[1], joint.y * image_shape[0], 2]
        else:
            coco_joints[idx] = [0, 0, 0]

    return coco_joints


def joints_mapping(joints):
    # preprocess joints: adapt visibity flag
    mpii_joints = [[x, y, 2] if v == 0 and x > 0 and y > 0 else [0, 0, 0] for x, y, v in joints]

    coco_joints = np.zeros((len(COCO_KEYPOINTS), 3))

    # HEAD (5): nose, eyes, ears (ALL ZEROS)
    # BODY (7): thorax/upper-neck, shoulders, elbows, wrists
    coco_joints[5:11:2] = mpii_joints[13:16]  # left
    coco_joints[6:12:2] = mpii_joints[10:13][::-1]  # right

    # LEGS (6): hips, knees, ankles
    coco_joints[11:17:2] = mpii_joints[3:6]  # left
    coco_joints[12:18:2] = mpii_joints[:3][::-1]  # right

    # transfer instance bbox
    person_bbox = np.array(mpii_joints[18][:2] + mpii_joints[19][:2])
    person_bbox[2:] -= mpii_joints[18][:2]

    # transfer face bbox
    face_bbox = np.array(mpii_joints[16][:2] + mpii_joints[17][:2])
    face_bbox[2:] -= mpii_joints[16][:2]

    return coco_joints, person_bbox, face_bbox
