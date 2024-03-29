import datajoint as dj
import numpy as np

from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo, CalibratedRecording
from pose_pipeline.pipeline import TopDownPerson
from multi_camera.datajoint.calibrate_cameras import Calibration

schema = dj.schema("hand_detection")


@schema
class HandBboxMethodLookup(dj.Lookup):
    definition = """
    detection_method      : int
    ---
    detection_method_name : varchar(50)
    """
    contents = [
        {"detection_method": 0, "detection_method_name": "RTMDet"},
        {"detection_method": 1, "detection_method_name": "TopDown"},
        
    ]

@schema
class HandBboxMethod(dj.Manual):
    definition = """
    -> SingleCameraVideo
    -> HandBboxMethodLookup
    ---
    """


@schema
class HandBbox(dj.Computed):
    definition = """
    -> HandBboxMethod
    ---
    num_boxes   :   int
    bboxes      :   longblob
    """   
    def make(self,key):
        if (HandBboxMethodLookup & key).fetch1("detection_method_name") == "RTMDet":
            from hand_detection.wrappers.hand_bbox import mmpose_hand_det
            num_boxes, bboxes = mmpose_hand_det(key=key, method="RTMDet")
            key["bboxes"] = bboxes
            key["num_boxes"] = num_boxes
        if (HandBboxMethodLookup & key).fetch1("detection_method_name") == "TopDown":
           
            from hand_detection.wrappers.hand_bbox import make_bbox_from_keypoints
            keypoints = (TopDownPerson & key & "top_down_method=2").fetch1("keypoints")
            num_boxes, bboxes = make_bbox_from_keypoints(keypoints)
            key["bboxes"] = bboxes
            key["num_boxes"] = num_boxes
        self.insert1(key)

@schema
class HandPoseEstimationMethodLookup(dj.Lookup):
    definition = """
    estimation_method      : int
    ---
    estimation_method_name : varchar(50)
    """
    contents = [
        {"estimation_method": 0, "estimation_method_name": "RTMPoseHand5"},
        {"estimation_method": 1, "estimation_method_name": "RTMPoseCOCO"},
        {"estimation_method": 2, "estimation_method_name": "freihand"},
        {"estimation_method": 3, "estimation_method_name": "HRNet_dark"},
        {"estimation_method": 4, "estimation_method_name": "HRNet_udp"},
        
    ]

@schema
class HandPoseEstimationMethod(dj.Manual):
    definition = """
    -> HandBbox
    -> HandPoseEstimationMethodLookup
    ---
    """


@schema
class HandPoseEstimation(dj.Computed):
    definition = """
    -> HandPoseEstimationMethod
    ---
    keypoints_2d       : longblob
    """   
    def make(self,key):
        if (HandPoseEstimationMethodLookup & key).fetch1("estimation_method_name") == "RTMPoseHand5":
            from hand_detection.wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'RTMPoseHand5')
        elif (HandPoseEstimationMethodLookup & key).fetch1("estimation_method_name") == "RTMPoseCOCO":
            from hand_detection.wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'RTMPoseCOCO')
        elif (HandPoseEstimationMethodLookup & key).fetch1("estimation_method_name") == "freihand":
            from hand_detection.wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'freihand')
        elif (HandPoseEstimationMethodLookup & key).fetch1("estimation_method_name") == "HRNet_dark":
            from hand_detection.wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'HRNet_dark')
        elif (HandPoseEstimationMethodLookup & key).fetch1("estimation_method_name") == "HRNet_udp":
            from hand_detection.wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'HRNet_udp')
        
        
        self.insert1(key)

    @staticmethod
    def joint_names(method="RTMPoseHand5"):
        if method == "RTMPoseHand5" or method == "RTMPoseCOCO" or method == "freihand" or method=="HRNet_udp":
            return ['Wrist','CMC1','MCP1','IP1','TIP1','MCP2','PIP2',
        'DIP2', 'TIP2', 'MCP3', 'PIP3', 'DIP3','TIP3', 'MCP4',
        'PIP4', 'DIP4', 'TIP4', 'MCP5', 'PIP5','DIP5', 'TIP5'
        ]
        elif method == "HRNet_dark":
            return ['Wrist','TIP1','IP1','MCP1','CMC1','TIP2','DIP2','PIP2','MCP2',
                      'TIP3','DIP3','PIP3','MCP3','TIP4', 'DIP4',  'PIP4', 'MCP4',
                      'TIP5','DIP5', 'PIP5','MCP5',
            ]


@schema
class HandPoseEstimationVideo(dj.Computed):
    definition = """
    -> HandPoseEstimation
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """   
    def make(self,key):
        from hand_detection.wrappers.hand_estimation import overlay_hand_keypoints
        import os
        import tempfile
        from pose_pipeline.pipeline import Video

        keypoints = (HandPoseEstimation & key).fetch1("keypoints_2d")
        vid_file = (Video & key).fetch1('video')      
        bboxes = np.asarray((HandBbox & key).fetch1('bboxes'))

        fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        
        overlay_hand_keypoints(vid_file, out_file_name, keypoints.copy(),bboxes)
        
        key["output_video"] = out_file_name
        self.insert1(key)
        #remove videos fetched
        os.remove(out_file_name)
        os.remove(vid_file)


@schema
class HandPoseReconstructionMethodLookup(dj.Lookup):
    definition = """
    reconstruction_method      : int
    ---
    reconstruction_method_name : varchar(50)
    """
    contents = [
        {"reconstruction_method": 0, "reconstruction_method_name": "Robust Triangulation"},
        {"reconstruction_method": 1, "reconstruction_method_name": r"Robust Triangulation $\\sigma=100$"},
        {"reconstruction_method": 2, "reconstruction_method_name": r"Robust Triangulation $\\sigma=50$"},
        {"reconstruction_method": 3, "reconstruction_method_name": r"Robust Triangulation $\\gamma=0.3$"},
        {"reconstruction_method": 4, "reconstruction_method_name": r"Robust Triangulation $\\sigma=10$"},
        # {"reconstruction_method": 1, "reconstruction_method_name": "Explicit Optimization KP Conf, MaxHuber=10"},
        # {"reconstruction_method": 2, "reconstruction_method_name": "Implicit Optimization KP Conf, MaxHuber=10"},
        # {"reconstruction_method": 3, "reconstruction_method_name": "Implicit Optimization"},
        # {"reconstruction_method": 4, "reconstruction_method_name": "Triangulation"},
        # {"reconstruction_method": 7, "reconstruction_method_name": "Explicit Optimization"},
        # {"reconstruction_method": 9, "reconstruction_method_name": "Implicit Optimization KP Conf"},
        # {"reconstruction_method": 10, "reconstruction_method_name": r"Implicit Optimization $\\gamma=0.3$"},
        # {"reconstruction_method": 11, "reconstruction_method_name": "Implicit Optimization, MaxHuber=10"},
        # {"reconstruction_method": 12, "reconstruction_method_name": r"Implicit Optimization $\\sigma=50$"},
    ]


@schema
class HandPoseReconstructionMethod(dj.Manual):
    definition = """
    -> CalibratedRecording
    -> HandPoseReconstructionMethodLookup
    estimation_method       : int
    detection_method        : int
    ---
    """

@schema
class HandPoseReconstruction(dj.Computed):
    definition = """
    -> HandPoseReconstructionMethod
    ---
    keypoints3d         : longblob
    camera_weights      : longblob  
    reprojection_loss   : float
    """   
    def make(self,key):
        import numpy as np
        from multi_camera.analysis.camera import robust_triangulate_points, triangulate_point
        from multi_camera.analysis.optimize_reconstruction import  reprojection_loss, smoothness_loss        
  
        calibration_key = (Calibration & key).fetch1("KEY")
        recording_key = (MultiCameraRecording & key).fetch1("KEY")
        reconstruction_method = key["reconstruction_method"]
        estimation_method = key["estimation_method"]
        detection_method = key["detection_method"]
        camera_calibration, camera_names = (Calibration & calibration_key).fetch1("camera_calibration", "camera_names")
        keypoints, camera_name = (
            HandPoseEstimation * SingleCameraVideo * MultiCameraRecording
            & {
                "estimation_method": estimation_method,
                "detection_method" : detection_method,
                "reconstruction_method": reconstruction_method,
            }
            & recording_key
        ).fetch("keypoints_2d", "camera_name")
        
        #concatenate all keypoionts for left and right hand
        new_keypoints=[]
        for camera_k in keypoints:
            new_keypoints.append(np.asarray([np.concatenate(frame_kp,axis=0) for frame_kp in camera_k]))
        keypoints = np.stack(new_keypoints)


        #pad zeros for all cameras
        N = max([len(k) for k in keypoints])
        keypoints = np.stack(
            [np.concatenate([k, np.zeros([N - k.shape[0], *k.shape[1:]])], axis=0) for k in keypoints], axis=0
        )

        print('CAMERAS', len(camera_names))
        # work out the order that matches the calibration (should normally match)
        order = [list(camera_name).index(c) for c in camera_names]
        points2d = np.stack([keypoints[o][:, :, :] for o in order], axis=0)


        # select method for reconstruction
        reconstruction_method_name = (HandPoseReconstructionMethodLookup & key).fetch1(
            "reconstruction_method_name"
        )

        if reconstruction_method_name == "Robust Triangulation":
            points3d, camera_weights = robust_triangulate_points(camera_calibration, points2d, return_weights=True)
        elif reconstruction_method_name == r"Robust Triangulation $\\sigma=100$":
            points3d, camera_weights = robust_triangulate_points(
                camera_calibration, points2d, return_weights=True, sigma=100
            )

        elif reconstruction_method_name == r"Robust Triangulation $\\sigma=50$":
            points3d, camera_weights = robust_triangulate_points(
                camera_calibration, points2d, return_weights=True, sigma=50
            )

        elif reconstruction_method_name == r"Robust Triangulation $\\gamma=0.3$":
            points3d, camera_weights = robust_triangulate_points(
                camera_calibration, points2d, return_weights=True, threshold=0.3
            )
        
        elif reconstruction_method_name == r"Robust Triangulation $\\sigma=10$":
            points3d, camera_weights = robust_triangulate_points(
                camera_calibration, points2d, return_weights=True, sigma=10, threshold = 0.3
            )        
        print(reconstruction_method_name)
        key["keypoints3d"] = np.array(points3d)
        key["camera_weights"] = np.array(camera_weights)
        key["reprojection_loss"] = reprojection_loss(camera_calibration, points2d, points3d[:, :, :3], huber_max=100)

        self.insert1(key, allow_direct_insert=True)



    def plot_joint(self, joint_idx=range(21), relative=False):
        from pose_pipeline import VideoInfo
        from matplotlib import pyplot as plt
        
        method_name = (HandPoseEstimationMethodLookup & self).fetch1("estimation_method_name")
        joint_names = HandPoseEstimation.joint_names(method_name)

        kp3d = self.fetch1("keypoints3d")
        timestamps = (VideoInfo * SingleCameraVideo & self).fetch("timestamps", limit=1)[0]
        # present = np.stack((HandBbox * SingleCameraVideo & self).fetch("present"))
        # present = np.sum(present, axis=0) / present.shape[0]
        kp2d = (HandPoseEstimation * SingleCameraVideo & self).fetch("keypoints_2d")
        #concatenate all keypoionts for left and right hand
        new_keypoints=[]
        for camera_k in kp2d:
            new_keypoints.append(np.asarray([np.concatenate(frame_kp,axis=0) for frame_kp in camera_k]))
        kp2d = np.stack(new_keypoints)
        
        kp3d = kp3d[:, :len(joint_names)]

        N = min([k.shape[0] for k in kp2d])
        keypoints2d = np.stack([k[:N] for k in kp2d], axis=0)

        keypoints2d = keypoints2d[:, :, : kp3d.shape[1]]
        dt = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        dt = dt[: kp3d.shape[0]]
        # print(kp3d[:, joint_idx,3])
        fig, ax = plt.subplots(3, 1, figsize=(5, 4))
        ax[0].plot(dt, kp3d[:, joint_idx, :3])
        ax[0].set_title('3d keypoints')
        ax[1].plot(dt, kp3d[:, joint_idx, 3])
        # ax[1].plot(dt, present)
        ax[1].set_ylim(0, 1)
        ax[1].set_title('3d keypoint confidence')
        ax[2].plot(dt, keypoints2d[:, :, joint_idx, 2].T)
        ax[2].set_title('2d keypoints confidence over cameras')
        plt.tight_layout()

    def points3d_to_trc(self, points3d, filename, marker_names, fps=30, rotMatrix=np.eye(3)):
        import pandas as pd
        '''
        Exports a set of points into an OpenSim TRC file

        Modified from Pose2Sim.make_trc

        Parameters:
            points3d (np.array) : time X joints X 3 array
            filename (string) : file to export to
            marker_names (list of strings) : names of markers to annotate
            fps : frame rate of points
        '''

        assert len(marker_names) == points3d.shape[1], "Number of marker names must match number of points"
        f_range = [0, points3d.shape[0]]

        # flatten keypoints after reordering axes
        points3d = np.take(points3d, [1, 2, 0], axis=-1)
        points3d = points3d.reshape([points3d.shape[0], -1])
        for m in range(len(marker_names)):
            points3d[:,(m*3):(m*3+3)] = points3d[:,(m*3):(m*3+3)]@rotMatrix
        #Header
        DataRate = CameraRate = OrigDataRate = fps
        NumFrames = points3d.shape[0]
        NumMarkers = len(marker_names)
        header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + filename,
                'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames',
                '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, f_range[0], f_range[1]])),
                'Frame#\tTime\t' + '\t\t\t'.join(marker_names) + '\t\t',
                '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(marker_names))])]

        #Add Frame# and Time columns
        Q = pd.DataFrame(points3d)
        Q = Q.interpolate()
        Q.insert(0, 't', Q.index / fps)

        #Write file
        with open(filename, 'w') as trc_o:
            [trc_o.write(line+'\n') for line in header_trc]
            Q.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')

        return Q
        
    def export_trc(self, filename, z_offset=0, start=None, end=None, return_points=False, smooth=False, addMovi=True):
        """Export an OpenSim file of marker trajectories

        Params:
            filename (string) : filename to export to
            z_offset (float, optional) : optional vertical offset
            start    (float, optional) : if set, time to start at
            end      (float, optional) : if set, time to end at
            return_points (bool, opt)  : if true, return points
        """

        from pose_pipeline import  VideoInfo
        from multi_camera.datajoint.multi_camera_dj import PersonKeypointReconstruction
        from multi_camera.analysis.biomechanics.opensim import normalize_marker_names

        method_name = (HandPoseEstimationMethodLookup & self).fetch1("estimation_method_name")
        joint_names = HandPoseEstimation.joint_names(method_name)

        joints3d = self.fetch1("keypoints3d").copy()
        joints3d = joints3d[:, : len(joint_names)]  # discard "unnamed" joints
        joints3d = joints3d / 1000.0  # convert to m
        fps = np.unique((VideoInfo * SingleCameraVideo * MultiCameraRecording & self).fetch("fps"))

        if joints3d.shape[-1] == 4:
            joints3d = joints3d[..., :-1]

        if end is not None:
            joints3d = joints3d[: int(end * fps)]
        if start is not None:
            joints3d = joints3d[int(start * fps) :]
        if smooth:
            import scipy

            for i in range(joints3d.shape[1]):
                for j in range(joints3d.shape[2]):
                    joints3d[:, i, j] = scipy.signal.medfilt(joints3d[:, i, j], 5)

        joints3d[...,:] = np.where(joints3d[...,:]==0, np.nan, joints3d[...,:])

        if addMovi:
            movi_joints= [  
                "R.clavicle",#2
                "R.Shoulder.M",#39
                "R.Elbow.Lateral",#41
                # "R.Forearm",#42
                "R.Wrist.Lateral.Thumb",#43
                "R.Wrist.Medial.pinky",#44
                "R.Elbow.Medial.Inner",#57
                    ]
            moviinds = np.array((2,39,41,43,44,57))
            joint_names = movi_joints + joint_names
            movikeys = self.fetch1('KEY')
            movikeys['reconstruction_method'] = 0
            movikeys['top_down_method']=12
            # movikeys = ((PersonKeypointReconstruction & self)).fetch('KEY')
            joints3dMovi = (PersonKeypointReconstruction & movikeys).fetch1('keypoints3d')[:,moviinds,:3]

            joints3d = np.concatenate((joints3dMovi/1000.0,joints3d),axis=1)

        #ROTATE ALONG THE Y AXIS
        theta = np.pi
        # transformX = np.array(([1, 0, 0],[0, np.cos(theta), -np.sin(theta)],[0,np.sin(theta), np.cos(theta)]))
        transformY = np.array(([np.cos(theta),0, np.sin(theta)],[0, 1, 0],[-np.sin(theta),0, np.cos(theta)]))
        # transformZ= np.array(([np.cos(theta),-np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0, 0, 1]))
        self.points3d_to_trc(
            joints3d + np.array([[[0, z_offset, 0]]]), filename, 
            normalize_marker_names(joint_names), fps=fps, rotMatrix=transformY
            )

        if return_points:
            return joints3d



@schema
class MJXReconstruction(dj.Computed): 
    definition = """
    -> HandPoseReconstruction
    # -> BiomechanicalReconstructionSettingsLookup
    ---
    body_scale             : longblob
    site_offsets           : longblob
    mean_reprojection_loss : float
    site_offset_loss       : float
    timestamps             : longblob   
    qpos                   : longblob
    qvel                   : longblob
    joints                 : longblob
    sites                  : longblob
    """   
    def make(self,key):
        from body_models.biomechanics_mjx.implicit_fitting import fit_keys, fit_key
        from body_models.biomechanics_mjx import ForwardKinematics
        import jax
        from body_models.biomechanics_mjx.implicit_fitting import fetch_keypoints

        from pose_pipeline import VideoInfo
        # biomechanical_reconstruction_settings = (BiomechanicalReconstructionSettingsLookup & key).fetch1("biomechanical_reconstruction_settings")

        model, metrics = fit_key(key, max_iters=40000)
        key["body_scale"] = np.array(model.body_scale)
        key["site_offsets"] = np.array(model.site_offsets)
        key["mean_reprojection_loss"] = float(metrics["keypoints_loss"])
        key["site_offset_loss"] = float(metrics["site_offset"])
        # key["delta_camera_loss"] = float(metrics["delta_camera"]) if "delta_camera" in metrics else 0.0

        timestamps, keypoints2d = fetch_keypoints(key)

        trial_result = model(timestamps, return_full=True, fast_inference=True)

        # key.update(base_key)

        key["timestamps"] = timestamps
        key["qpos"] = np.array(trial_result.qpos)
        key["qvel"] = np.array(trial_result.qvel)
        key["joints"] = np.array(trial_result.xpos)
        key["sites"] = np.array(trial_result.site_xpos)
        # key["reprojection_loss"] = float(metrics[f"keypoint_loss_{i}"])
        self.insert1(key)