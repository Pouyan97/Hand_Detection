import datajoint as dj
import numpy as np

from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo, CalibratedRecording
from pose_pipeline.pipeline import TopDownPerson, BlurredVideo
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
        {"detection_method": 2, "detection_method_name": "3Dto2D"},
        
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
        elif (HandBboxMethodLookup & key).fetch1("detection_method_name") == "TopDown":
           
            from hand_detection.wrappers.hand_bbox import make_bbox_from_keypoints
            keypoints = (TopDownPerson & key & "top_down_method=2").fetch1("keypoints")
            num_boxes, bboxes = make_bbox_from_keypoints(keypoints)
            key["bboxes"] = bboxes
            key["num_boxes"] = num_boxes
        elif (HandBboxMethodLookup & key).fetch1("detection_method_name") == "3Dto2D":
            from multi_camera.analysis.camera import robust_triangulate_points,project_distortion
            from hand_detection.wrappers.hand_bbox import make_bbox_from_keypoints
            from multi_camera.datajoint.sessions import Recording
            keypoints, camera_name = (SingleCameraVideo * MultiCameraRecording * TopDownPerson & (Recording & key) & "top_down_method=2").fetch("keypoints", "camera_name")
            calibration_key = (CalibratedRecording & key).fetch1("KEY")
            camera_calibration, camera_names = (Calibration & calibration_key).fetch1("camera_calibration", "camera_names")
            
            
            order = [list(camera_name).index(c) for c in camera_names]
            points2d = np.stack([keypoints[o][:, -42:, :] for o in order], axis=0)
 
            camera_num = list(camera_name).index((SingleCameraVideo & key).fetch1("camera_name"))
            points3d, _ = robust_triangulate_points(camera_calibration, points2d, return_weights=True)
            kp2d_proj = np.array([project_distortion(camera_calibration, i, points3d) for i in range(camera_calibration["mtx"].shape[0])])
            num_boxes, bboxes = make_bbox_from_keypoints(kp2d_proj[camera_num])
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
        {"estimation_method": -1, "estimation_method_name": "Halpe"},
        {"estimation_method": 0,  "estimation_method_name": "RTMPoseHand5"},
        {"estimation_method": 1,  "estimation_method_name": "RTMPoseCOCO"},
        {"estimation_method": 2,  "estimation_method_name": "freihand"},
        {"estimation_method": 3,  "estimation_method_name": "HRNet_dark"},
        {"estimation_method": 4,  "estimation_method_name": "HRNet_udp"},
        
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
        elif (HandPoseEstimationMethodLookup & key).fetch1("estimation_method_name") == "Halpe":
            from hand_detection.wrappers.hand_estimation import mmpose_HPE
            kp2d = (TopDownPerson & key & "top_down_method=2").fetch1("keypoints")
            key["keypoints_2d"] = np.concatenate((kp2d[:,-21:,:],kp2d[:,-42:-21,:]),axis=1)
        
        
        self.insert1(key)

    @staticmethod
    def joint_names(method="RTMPoseHand5"):
        if method == "RTMPoseHand5" or method == "RTMPoseCOCO" or method == "freihand" or method=="HRNet_udp" or method=="Halpe":
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
        vid_file = (BlurredVideo & key).fetch1('output_video')      
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
        # if estimation_method != -1:
        #     new_keypoints=[]
        #     for camera_k in keypoints:
        #         new_keypoints.append(np.asarray([np.concatenate(frame_kp,axis=0) for frame_kp in camera_k]))
        #     keypoints = np.stack(new_keypoints)


        #pad zeros for all cameras
        N = max([len(k) for k in keypoints])
        keypoints = np.stack(
            [np.concatenate([k, np.zeros([N - k.shape[0], *k.shape[1:]])], axis=0) for k in keypoints], axis=0
        )
        print('Triangulating over ', len(camera_names))
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
        # new_keypoints=[]
        # for camera_k in kp2d:
        #     new_keypoints.append(np.asarray([np.concatenate(frame_kp,axis=0) for frame_kp in camera_k]))
        # kp2d = np.stack(new_keypoints)
        
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
        fps = np.unique((VideoInfo * SingleCameraVideo * MultiCameraRecording & self).fetch("fps"))[0]

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
            movikeys.pop('reconstruction_method')
            # movikeys['reconstruction_method'] = 0
            movikeys['top_down_method']=12

            robust_movi = (PersonKeypointReconstruction & movikeys & 'reconstruction_method=0').fetch('reprojection_loss')
            implicit_movi = (PersonKeypointReconstruction & movikeys & 'reconstruction_method=2').fetch('reprojection_loss')
            if len(robust_movi) == 0:
                print('No MOVI data found for robust triangulation')
            if len(implicit_movi) == 0:
                print('No MOVI data found for implicit optimization')
            if robust_movi < implicit_movi:
                movikeys['reconstruction_method'] = 0
                print('Using robust triangulation for MOVI')
            else:
                movikeys['reconstruction_method'] = 2
                print('Using implicit optimization for MOVI')
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
class MJXReconstructionMethod(dj.Manual):
    definition = """
    -> CalibratedRecording
    estimation_method       : int
    detection_method        : int
    ---
    """

@schema
class MJXReconstruction(dj.Computed): 
    definition = """
    -> MJXReconstructionMethod
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
        # print(key['estimation_method'])
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
        
        from body_models.biomechanics_mjx.implicit_fitting import fetch_keypoints,get_joint_names_index
        from pose_pipeline import  VideoInfo
        from multi_camera.datajoint.multi_camera_dj import PersonKeypointReconstruction
        from multi_camera.analysis.biomechanics.opensim import normalize_marker_names

        method_name = (HandPoseEstimationMethodLookup & self).fetch1("estimation_method_name")
        joint_names = HandPoseEstimation.joint_names(method_name)
        movi_joints= [  
                "R.clavicle",#2
                "R.Shoulder.M",#39
                "R.Elbow.Lateral",#41
                # "R.Forearm",#42
                "R.Wrist.Lateral.Thumb",#43
                "R.Wrist.Medial.pinky",#44
                "R.Elbow.Medial.Inner",#57
                ]
        joint_names = movi_joints + joint_names

        pred = self.fetch1('sites')
        joints3d = pred.copy()


        # joints3d = self.fetch1("keypoints3d").copy()
        # joints3d = joints3d[:, : len(joint_names)]  # discard "unnamed" joints
        # joints3d = joints3d / 1000.0  # convert to m
        fps = np.unique((VideoInfo * SingleCameraVideo * MultiCameraRecording & (CalibratedRecording & self)).fetch("fps"))

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
class MJXReconstructionVideo(dj.Computed):
    definition = """
    -> MJXReconstruction
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """
    def make(self,key):
        import tempfile
        import os
        import cv2
        from tqdm import tqdm
        from hand_detection.wrappers.hand_estimation import plot_3d_reprojected_keypoints
        # pose = (MJXReconstruction & key).fetch1('qpos')
        pred = (MJXReconstruction & key).fetch1('sites')
        kp3d = pred*1000
        results, grid = plot_3d_reprojected_keypoints(key, kp3d, only_osim = False)
        # write the collated frames into a video matching the original frame rate using opencv VideoWriter
        fd, filename = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        method = (HandPoseEstimationMethodLookup & key).fetch1('estimation_method_name')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, 29, (grid[0].shape[1], grid[0].shape[0]))
        frames = []
        for frame in tqdm(grid, desc="Writing"):
            cv2.putText(frame,  
                        method+'  |  Green: 2D detections  |  Blue: 3D reprojection',  
                        (50, 50),  
                        cv2.FONT_HERSHEY_SIMPLEX , 1,  
                        (0, 255, 255),  
                        2,  
                        cv2.LINE_4) 
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        key["output_video"] = filename
        self.insert1(key)
        os.remove(filename)


@schema
class MJXReconstructionAnalysis(dj.Computed):
    definition = """
    -> MJXReconstruction
    ---
    pk_5              : float
    pk_10             : float
    spatial_loss      : float
    pose_noise        : float
    """
    def make(self,key):
        # from body_models.biomechanics_mjx.implicit_fitting import fetch_keypoints
        from pose_pipeline import  VideoInfo
        from multi_camera.datajoint.sessions import Recording
        from multi_camera.analysis import fit_quality
        import cv2
        from multi_camera.analysis.camera import project_distortion, get_intrinsic, get_extrinsic, distort_3d
        
        # timestamps, keypoints2d = fetch_keypoints(key)

        # pose = (MJXReconstruction & key).fetch1('qpos')
        pred = (MJXReconstruction & key).fetch1('sites')
        # tables = Recording *  HandPoseEstimation * HandPoseEstimationMethodLookup * HandPoseReconstructionMethodLookup
        camera_name = (SingleCameraVideo * MultiCameraRecording * HandPoseEstimation & key).fetch('camera_name')


        #Last 21 keypoints is the hand points
        kp3dMuj = pred*1000
        # kp3dMuj = trajectory(timestamps)*1000
        #RIGHT HAND without the MOVI stuff
        kp3dMuj = kp3dMuj[:, -21:, :]
        calibration_key = (Calibration & key).fetch1("KEY")
        camera_params, camera_names = (Calibration & calibration_key).fetch1("camera_calibration", "camera_names")

        keypointsHPE = (HandPoseEstimation & key).fetch('keypoints_2d')


        #pad zeros for all cameras
        N = max([len(k) for k in keypointsHPE])
        keypointsHPE = np.stack(
            [np.concatenate([k, np.zeros([N - k.shape[0], *k.shape[1:]])], axis=0) for k in keypointsHPE], axis=0
        )

        #pad zeros for 3d keypoints
        kp3dMuj = np.concatenate([kp3dMuj, np.zeros([N - kp3dMuj.shape[0], *kp3dMuj.shape[1:]])], axis=0)
        
        # work out the order that matches the calibration (should normally match)
        order = [list(camera_name).index(c) for c in camera_names]
        points2d = np.stack([keypointsHPE[o][:, :21, :] for o in order], axis=0)

        #Get the right hand compared to Osim
        metrics2, thresh, confidence = fit_quality.reprojection_quality( kp3dMuj[:, :, :3], camera_params, points2d[:,:,:21,:])
        pck10 = metrics2[np.argmin(np.abs(thresh - 10)), np.argmin(np.abs(confidence - 0.5))]
        pck5 = metrics2[np.argmin(np.abs(thresh - 5)), np.argmin(np.abs(confidence - 0.5))]

        #SETTING UP CALCULATION FOR SPATIAL ERROR
        videos = (HandPoseEstimation * MultiCameraRecording  * SingleCameraVideo & Recording & key).proj()
        keypoints2d = points2d
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

        pos = np.array([-R.dot(t) for R, t in zip(rmats, tvec)])*1000

        #CALCULATING THE SPATIAL ERROR
        keypoints_2d_triangulated = np.array([project_distortion(camera_params, i, kp3dMuj) for i in range(camera_params["mtx"].shape[0])])
        SC= []
        for ci in range(keypoints2d.shape[0]):
            
            # Extract the translation vector from the camera parameters
            # Calculate the distance to the point
            distance = np.linalg.norm(pos[ci,:] - kp3dMuj[...,:3], axis=-1)
            
            projection_error = keypoints_2d_triangulated[ci,:,:21,:2] - keypoints2d[ci,:,:, :2]
            # keypoint_conf = keypoints2d[ci,..., 2]
            intrinsics = get_intrinsic(camera_params,0)

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
            SC.append(np.median(spatial_error))

        #CALCULATING THE NOISE
        point_difference = np.diff(kp3dMuj/1000, axis=0)**2
        diff_sum = np.sum(point_difference, axis=0) 
        Noise_mjx = np.mean(diff_sum, axis=0)
        
        key['pk_5']= pck5.item()
        key['pk_10']= pck10.item()
        
        key['spatial_loss'] = np.median(SC).item()
        key['pose_noise'] = np.mean(Noise_mjx).item()
        print(key)
        self.insert1(key)