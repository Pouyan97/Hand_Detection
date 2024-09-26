 # from body_models.biomechanics_mjx.implicit_fitting import fetch_keypoints
from pose_pipeline import  VideoInfo
from multi_camera.datajoint.sessions import Recording
from multi_camera.analysis import fit_quality
import cv2
import numpy as np
from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo, CalibratedRecording
from multi_camera.datajoint.calibrate_cameras import Calibration
from multi_camera.analysis.camera import project_distortion, get_intrinsic, get_extrinsic, distort_3d
from hand_detection.hand_dj import HandPoseEstimation, HandPoseEstimationMethodLookup, HandPoseReconstructionMethodLookup
        
def GC_analysis(key, kp3d):
        # kp3d = (OpenSimReconstruction & key).fetch1('keypoints')
        # tables = Recording *  HandPoseEstimation * HandPoseEstimationMethodLookup * HandPoseReconstructionMethodLookup
        camera_name = (SingleCameraVideo * MultiCameraRecording * HandPoseEstimation & key).fetch('camera_name')

        #First 21 keypoints is the right hand points
        # kp3d = kp3d[:, -21:, :]
        calibration_key = (CalibratedRecording & key).fetch1("KEY")
        camera_params, camera_names = (Calibration & calibration_key).fetch1("camera_calibration", "camera_names")

        keypointsHPE = (HandPoseEstimation & key).fetch('keypoints_2d')
        #pad zeros for all cameras
        N = max([len(k) for k in keypointsHPE])
        keypointsHPE = np.stack(
            [np.concatenate([k, np.zeros([N - k.shape[0], *k.shape[1:]])], axis=0) for k in keypointsHPE], axis=0
        )

        #pad zeros for 3d keypoints
        kp3d = np.concatenate([kp3d, np.zeros([N - kp3d.shape[0], *kp3d.shape[1:]])], axis=0)
        
        # work out the order that matches the calibration (should normally match)
        order = [list(camera_name).index(c) for c in camera_names]
        points2d = np.stack([keypointsHPE[o][:, :21, :] for o in order], axis=0)

        #Get the right hand compared to 3d keypoints
        metrics2, thresh, confidence = fit_quality.reprojection_quality( kp3d[:, :, :3], camera_params, points2d[:,:,:21,:])
        pck10 = metrics2[np.argmin(np.abs(thresh - 10)), np.argmin(np.abs(confidence - 0.5))]
        pck5 = metrics2[np.argmin(np.abs(thresh - 5)), np.argmin(np.abs(confidence - 0.5))]
        pcks = np.array([metrics2[np.argmin(np.abs(thresh - i)), np.argmin(np.abs(confidence - 0.5))].item() for i in range(16)])
        
        return pck5.item(), pck10.item(), pcks

def MPJPError(key, kp3d):
        calibration_key = (CalibratedRecording & key).fetch1("KEY")
        camera_params, camera_names = (Calibration & calibration_key).fetch1("camera_calibration", "camera_names")
        keypointsHPE = (HandPoseEstimation & key).fetch('keypoints_2d')
        #pad zeros for all cameras
        N = max([len(k) for k in keypointsHPE])
        keypointsHPE = np.stack(
            [np.concatenate([k, np.zeros([N - k.shape[0], *k.shape[1:]])], axis=0) for k in keypointsHPE], axis=0
        )

        #pad zeros for 3d keypoints
        kp3d = np.concatenate([kp3d, np.zeros([N - kp3d.shape[0], *kp3d.shape[1:]])], axis=0)
        
        camera_name = (SingleCameraVideo * MultiCameraRecording * HandPoseEstimation & key).fetch('camera_name')
        # work out the order that matches the calibration (should normally match)
        order = [list(camera_name).index(c) for c in camera_names]
        keypoints2d = np.stack([keypointsHPE[o][:, :21, :] for o in order], axis=0)

        #SETTING UP CALCULATION FOR SPATIAL ERROR
        videos = (HandPoseEstimation * MultiCameraRecording  * SingleCameraVideo & Recording & key).proj()
        rvec = camera_params["rvec"]
        rmats = [cv2.Rodrigues(np.array(r[None, :]))[0].T for r in rvec]

        tvec = camera_params["tvec"]
        rvec = camera_params["rvec"]
        # Removed display function calls to prevent error
        # Assuming camera_params['mtx'] and camera_params['dist'] are well-defined
        video_keys = (videos).fetch("KEY")
        fps = np.unique((VideoInfo & video_keys[0]).fetch1("fps"))
        width = np.unique((VideoInfo & video_keys).fetch("width"))[0]
        height = np.unique((VideoInfo & video_keys).fetch("height"))[0]

        # Convert rotation vectors to rotation matrices
        rmats = [cv2.Rodrigues(np.array(r[None, :]))[0].T for r in rvec]

        pos = np.array([-R.dot(t) for R, t in zip(rmats, tvec)])

        #CALCULATING THE SPATIAL ERROR
        keypoints_2d_triangulated = np.array([project_distortion(camera_params, i, kp3d) for i in range(camera_params["mtx"].shape[0])])
        SC= []
        for ci in range(keypoints2d.shape[0]):
            
            # Extract the translation vector from the camera parameters
            # Calculate the distance to the point
            distance = np.linalg.norm(pos[ci,:] - kp3d[...,:3], axis=-1)
            
            projection_error = keypoints_2d_triangulated[ci,:,:21,:2] - keypoints2d[ci,:,:, :2]
            # keypoint_conf = keypoints2d[ci,..., 2]
            intrinsics = get_intrinsic(camera_params,ci)

            # Extract the focal lengths from the camera parameters
            focal_length_x = intrinsics[0, 0]  # assuming the focal length in x direction is at this location
            focal_length_y = intrinsics[1, 1]  # assuming the focal length in y direction is at this location

            # Calculate the FOVs
            fov_x = 2 * np.arctan(width / (2 * focal_length_x))
            fov_y = 2 * np.arctan(height / (2 * focal_length_y))
            # Calculate the degree of view for each pixel
            degree_per_pixel_x = fov_x / width
            degree_per_pixel_y = fov_y / height

            angular_error = np.array([degree_per_pixel_x, degree_per_pixel_y]) * projection_error
            # Calculate the spatial errors
            spatial_error_x = distance * np.tan(angular_error[...,0])
            spatial_error_y = distance * np.tan(angular_error[...,1])
            spatial_error = np.linalg.norm(np.array([spatial_error_x, spatial_error_y]),axis=0)
            # Calculate the widths of the view at the distance of the object
            # print('camera', ci, 'median error: ', np.median(spatial_error), 'mm')
            # # Calculate the spatial errors
            SC.append(np.mean(spatial_error))

        #CALCULATING THE NOISE
        point_difference = np.diff(kp3d/1000, axis=0)**2
        diff_sum = np.sum(point_difference, axis=0) 
        Noise_mjx = np.mean(diff_sum, axis=0)

        
        MPJPE = np.median(SC).item()
        Noise = np.mean(Noise_mjx).item()
        return MPJPE, Noise
