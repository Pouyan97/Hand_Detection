import os
import cv2
import numpy as np
import datajoint as dj
from pose_pipeline import Video
from hand_dj import handBbox

def mmpose_HPE(key, method='RTMPose'):

    from mmpose.apis import inference_topdown, init_model
    from mmpose.evaluation.functional import nms

    video =  Video.get_robust_reader(key, return_cap=False) # returning video allows deleting it
    bboxes = (handBbox & key).fetch1("bboxes")

    if method == 'freihand':
        pose_model_cfg = 'models/td-hm_res50_8xb64-100e_freihand2d-224x224.py'
        pose_model_ckpt = 'https://download.openmmlab.com/mmpose/hand/resnet/res50_freihand_224x224-ff0799bc_20200914.pth'
        num_keypoints = 21
    if method == 'RTMPose':
        pose_model_cfg = 'models/rtmpose-m_8xb256-210e_hand5-256x256.py'
        pose_model_ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
        num_keypoints = 21
        
    device = 'cuda'
    model = init_model(pose_model_cfg, pose_model_ckpt, device=device)
    cap = cv2.VideoCapture(video)
    # video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    results = []
    for bbox in bboxes:
        ret, frame = cap.read()
        assert ret and frame is not None
        # handle the case where hand is not detected
        if np.any(np.isnan(bbox)):
            results.append(np.zeros((num_keypoints, 2)))
            continue
        #run the frame through the model
        pose_results = inference_topdown(model, frame, bbox)
        #get prediction instances from mmpose results
        pred_instances = pose_results.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        #concat scores and keypoints
        results.append(np.concatenate((keypoints,keypoint_scores[:,None]),axis = 1))


    cap.release()
    os.remove(video)

    return np.asarray(results)