import os
import cv2
import numpy as np
import datajoint as dj
from pose_pipeline import Video
from hand_detection.hand_dj import HandBbox
from tqdm import tqdm

def mmpose_HPE(key, method='RTMPoseHand5'):

    from mmpose.apis import inference_topdown, init_model
    from mmpose.evaluation.functional import nms
    path = os.path.dirname(os.path.abspath(__file__))
    if method == 'freihand':
        pose_model_cfg = os.path.join(path, 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/freihand2d/td-hm_res50_8xb64-100e_freihand2d-224x224.py')
        pose_model_ckpt = 'https://download.openmmlab.com/mmpose/hand/resnet/res50_freihand_224x224-ff0799bc_20200914.pth'
        num_keypoints = 21
    elif method == 'RTMPoseHand5':
        pose_model_cfg = os.path.join(path, 'mmpose/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py')
        pose_model_ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
        num_keypoints = 21
    elif method == 'RTMPoseCOCO':
        pose_model_cfg = os.path.join(path, 'mmpose/configs/hand_2d_keypoint/rtmpose/coco_wholebody_hand/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py')
        pose_model_ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_210e-256x256-99477206_20230228.pth'
        num_keypoints = 21
    elif method == 'HRNet_dark':
        pose_model_cfg = os.path.join(path, "mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py")
        pose_model_ckpt = "https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth"
        num_keypoints = 21
    elif method == 'HRNet_udp':
        pose_model_cfg = os.path.join(path, "mmpose/configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_udp-8xb64-210e_onehand10k-256x256.py")
        pose_model_ckpt = "https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_onehand10k_256x256_udp-0d1b515d_20210330.pth"
        num_keypoints = 21



    device = 'cuda'
    model = init_model(pose_model_cfg, pose_model_ckpt, device=device)
    
    video =  Video.get_robust_reader(key, return_cap=False) # returning video allows deleting it
    bboxes = (HandBbox & key).fetch1("bboxes")
    
    
    cap = cv2.VideoCapture(video)
    # video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    results = []
    for bbox in tqdm(bboxes):
        ret, frame = cap.read()
        assert ret and frame is not None
        # handle the case where hand is not detected
        # if np.any(np.isnan(bbox)):
        #     results.append(np.zeros((num_keypoints, 3)))
        #     continue
        #run the frame through the model
        pose_results = inference_topdown(model, frame, bbox)
        ##
        #Pose_results includes the number of detections
        #pred_instances includes the scores as well as keypoint
        #
        #get prediction instances from mmpose results
        num_hands = len(pose_results)
        keypoints_2d =[]
        for i in range(num_hands):
            pred_instances = pose_results[i].pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores
            keypoints_2d.append(np.concatenate((keypoints[0,:,:],keypoint_scores.T), axis = -1))
        # print(keypoints.shape,keypoint_scores.T.shape,len(pose_results), len(bbox))
        #concat scores and keypoints(flatten)
        results.append(np.concatenate(keypoints_2d, axis=0))


    cap.release()
    os.remove(video)

    return np.array(results)



def overlay_hand_keypoints(video, output_file, keypoints, bboxes):
        """Process a video and create overlay of keypoints

        Args:
        video (str): filename for source (from key)
        output_file (str): output filename
        keypoints (list): list of list of keypoints
        """
        from pose_pipeline.utils.visualization import draw_keypoints
        
        #Get video details
        cap = cv2.VideoCapture(video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_size = (int(w),int(h))
        # bboxes = bboxes.astype(int)

        # bbox = np.min(bboxes,axis=0)-100
        # bbox[-2:]= np.max(bboxes,axis=0)[-2:]+100
        #set writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_file,fourcc, fps,output_size)
        #process every frame
        for frame_idx in tqdm(range(total_frames)):
            success, frame = cap.read()
            if not success:
                break
            keypoints_2d = keypoints[frame_idx][:,:]
            frame = draw_keypoints(frame,keypoints_2d, threshold=0.2)
            for bbox in bboxes[frame_idx]:
                frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)  # Green color, 2 pixel thickness

            out.write(frame)
        #remove
        out.release()
        cap.release()











def plot_triangulated_keypoints(key, kp3d, only_osim = False):
        from hand_detection.hand_dj import HandPoseEstimation, HandBbox, HandPoseReconstruction
        from multi_camera.datajoint.sessions import Recording
        from multi_camera.datajoint.multi_camera_dj import Calibration, MultiCameraRecording, SingleCameraVideo
        from multi_camera.analysis.camera import project_distortion, get_intrinsic, get_extrinsic, distort_3d

        from pose_pipeline.pipeline import Video, VideoInfo, BlurredVideo
        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints
        import cv2

        videos = (HandPoseEstimation * MultiCameraRecording  * SingleCameraVideo & Recording & key).proj()
        camera_params= (Recording * Calibration & key).fetch1("camera_calibration")
        camera_names = (Recording * Calibration & key).fetch1("camera_names")
        video_keys = (videos).fetch("KEY")
        fps = np.unique((VideoInfo & video_keys[0]).fetch1("fps"))
        width = np.unique((VideoInfo & video_keys).fetch("width"))[0]
        height = np.unique((VideoInfo & video_keys).fetch("height"))[0]

        keypoints_2d_MJX = np.array([project_distortion(camera_params, i, kp3d) for i in range(camera_params["mtx"].shape[0])])
        # keypoints_2d_triangulated = np.array([project_distortion(camera_params, i, kp3d) for i in range(camera_params["mtx"].shape[0])])

        num_cameras = len(camera_names)
        results = []
        for ci in range(num_cameras):
            cam_idx = ci
            
            i = cam_idx
            # get camera parameters
            K = np.array(get_intrinsic(camera_params, i))

            # don't use real extrinsic since we apply distortion which does this
            R = np.eye(3)
            T = np.zeros((3,))
            cameras = {"K": [K], "R": [R], "T": [T]}


            background = np.ones((height, width, 3), dtype=np.uint8) * 127
            vid_file = (BlurredVideo & video_keys[i]).fetch1("output_video")
            # vid_file = (Video & video_keys[i]).fetch1("video")
            vid = cv2.VideoCapture(vid_file)

            kp2d_camera = np.asarray((HandPoseEstimation & video_keys[ci]).fetch1("keypoints_2d"))
            kp2d_camera = kp2d_camera.reshape(kp2d_camera.shape[0], -1, kp2d_camera.shape[-1])    
            bboxes = np.asarray((HandBbox & video_keys[ci]).fetch1('bboxes'))
            bboxes = bboxes[:,0,:]

            bbox = np.min(bboxes,axis=0)
            bbox[-2:]= np.max(bboxes,axis=0)[-2:]
            # bbox[:2] -= bbox[:2]/2
            # bbox[-2:] += bbox[-2:]/2
            
            def render_overlay(frame, idx, frame_idx):
                #Blue
                color = (200, 30, 30)
                raw_frame = draw_keypoints(frame, np.array(keypoints_2d_MJX[idx, frame_idx]), radius=5, threshold=0.10, border_color=color, color=color)
                    
                if not only_osim:
                    # #Red
                    # color = (30, 30, 200)
                    # raw_frame = draw_keypoints(raw_frame, np.array(keypoints_2d_triangulated[idx, frame_idx]), radius=5, threshold=0.10, border_color=color, color=color)

                    #Green
                    color = (30, 200, 30)
                    raw_frame = draw_keypoints(raw_frame, np.array(kp2d_camera[frame_idx,:21]), radius=5, threshold=0.10, border_color=color, color=color)
                    
                raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

                # raw_frame = crop_image_bbox(
                #     raw_frame, bboxes[frame_idx], target_size=(288, int(288 * 1920 / 1080)), dilate=1.0
                # )[0]
                target_size= (280, int(280 * 1920 / 1080))
                dilate= 1.4 
                image = raw_frame
                # bbox = bboxes[frame_idx]            
                # bbox = fix_bb_aspect_ratio(bbox, ratio=target_size[0] / target_size[1], dilate=dilate)
                
                # three points on corner of bounding box
                src = np.asarray([[bbox[0], bbox[1]], [bbox[2] ,bbox[3]], [bbox[0], bbox[3]]])
                dst = np.array([[0, 0], [target_size[0], target_size[1]], [0, target_size[1]]])  # .astype(np.float32)
                trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
                image = cv2.warpAffine(image, trans, target_size, flags=cv2.INTER_LINEAR)

                return image
            
            def make_frames():
                list_frames = []
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for frame_num in tqdm(range(keypoints_2d_MJX.shape[1])):#range(260,290):
                    # print(frame_num, '|', kp3d.shape[0])
                    # vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    _, frame = vid.read()
                    frame = render_overlay(frame, cam_idx, frame_num)
                    list_frames.append(frame)
                return list_frames
            
            results.append(make_frames())
            os.remove(vid_file)
            vid.release()
        return results
        

def plot_3d_reprojected_keypoints(key, kp3d, only_osim = False):
    results = plot_triangulated_keypoints(key, kp3d, only_osim = only_osim)
    def images_to_grid(images, n_cols=4):
        n_rows = int(np.ceil(len(images) / n_cols))
        grid = np.zeros((n_rows * images[0].shape[0], n_cols * images[0].shape[1], 3), dtype=np.uint8)
        for i, img in enumerate(images):
            row = i // n_cols
            col = i % n_cols
            grid[row * img.shape[0] : (row + 1) * img.shape[0], col * img.shape[1] : (col + 1) * img.shape[1], :] = img
        return grid

    # collate the results into a grid
    grid = [images_to_grid(r) for r in zip(*results)]
    return results, grid
