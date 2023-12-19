import os
import cv2
import numpy as np
import datajoint as dj
from pose_pipeline import Video


def mmpose_hand_det(key, method='RTMDet'):

    from mmpose.apis import init_model
    try:
        from mmdet.apis import inference_detector, init_detector
        has_mmdet =True
    except (ImportError, ModuleNotFoundError):
        has_mmdet = False 
    from mmpose.utils import adapt_mmdet_pipeline
    from mmpose.evaluation.functional import nms

    video =  Video.get_robust_reader(key, return_cap=False) # returning video allows deleting it

    if method == 'RTMDet':
        detection_cfg = 'wrappers/models/rtmdet_nano_320-8xb32_hand.py'
        detection_ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth'
        device = 'cpu'

    # build detector
    detector = init_detector(detection_cfg, detection_ckpt, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    
    #capture video
    cap = cv2.VideoCapture(video)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    boxes_list = []
    num_boxes = 0
    # iterate trough frames
    for frame_id in range(video_length):
        ret, frame = cap.read()
        assert ret and frame is not None
        #get detection results
        det_result = inference_detector(detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()
        #calculate bboxes
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)

        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                    pred_instance.scores > .3)]
        bboxes = bboxes[nms(bboxes, .3), :4]
        #expand bboxes by 100 pixels
        bboxes[:,:2] -= 100
        bboxes[:,-2:] += 100
        if(bboxes.shape[0] > num_boxes):
            num_boxes = bboxes.shape[0]
        boxes_list.append(bboxes)

    cap.release()
    os.remove(video)

    return num_boxes, boxes_list