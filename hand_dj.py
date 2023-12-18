import datajoint as dj
import numpy as np

from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo #, PersonReconstruction


schema = dj.schema("Hand_Detection")


@schema
class handBboxMethodLookUp(dj.LookUp):
    definition = """
    detection_method      : int
    ---
    detection_method_name : varchar(50)
    """
    contents = [
        {"detection_method": 0, "detection_method_name": "RTMDet"},
    ]

@schema
class handBboxMethod(dj.Manual):
    definition = """
    -> SingleCameraVideo
    -> CalibratedRecording
    -> handBboxMethodLookUp
    detection_method   : int
    ---
    """


@schema
class handBbox(dj.Computed):
    definition = """
    -> handBboxMethod
    detection_method   : int
    ---
    num_boxes   :   int
    bboxes      :   longblob
    """   
    def make(self,key):
        if (handBboxMethodLookUp & key).fetch1("lifting_method_name") == "RTMDet":
            from wrappers.hand_bbox import mmpose_hand_det
            bboxes = mmpose_hand_det(key=key, method="RTMDet")
            key["bboxes"] = bboxes
        self.insert1(key)

@schema
class handPoseEstimationMethodLookUp(dj.LookUp):
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
    ]

@schema
class handPoseEstimationMethod(dj.Manual):
    definition = """
    -> handBbox
    -> handPoseEstimationMethodLookUp
    ---
    """


@schema
class handPoseEstimation(dj.Computed):
    definition = """
    -> handPoseEstimationMethod
    ---
    keypoints_2d       : longblob
    """   
    def make(self,key):
        if (handPoseEstimationMethodLookUp & key).fetch1("estimation_method_name") == "RTMPoseHand5":
            from wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'RTMPoseHand5')
        elif (handPoseEstimationMethodLookUp & key).fetch1("estimation_method_name") == "RTMPoseCOCO":
            from wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'RTMPoseCOCO')
        elif (handPoseEstimationMethodLookUp & key).fetch1("estimation_method_name") == "freihand":
            from wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'freihand')
        elif (handPoseEstimationMethodLookUp & key).fetch1("estimation_method_name") == "HRNet_dark":
            from wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'HRNet_dark')
        
        
        self.insert1(key)