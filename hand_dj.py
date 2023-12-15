import datajoint as dj
import numpy as np

from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo #, PersonReconstruction


schema = dj.schema("Hand_Detection")


@schema
def handBboxMethodLookUp(dj.LookUp):
    definition = """
    detection_method      : int
    ---
    detection_method_name : varchar(50)
    """
    contents = [
        {"detection_method": 0, "detection_method_name": "RTMDet"},
    ]

@schema
def handBboxMethd(dj.Manual):
    definition = """
    -> SingleCameraVideo
    -> CalibratedRecording
    -> handBboxMethodLookUp
    detection_method   : int
    ---
    """


@schema
def handBbox(dj.Computed):
    definition = """
    -> SingleCameraVideo
    -> CalibratedRecording
    -> handBboxMethodLookUp
    detection_method   : int
    ---
    num_boxes   :   int
    Bboxes      :   longblob
    """   
    def make(self,key):
        video = SingleCameraVideo.get_robust_reader(key, return_cap=False)
        # bboxes = my_method process the video()

@schema
def handPoseEstimationMethodLookUp(dj.LookUp):
    definition = """
    estimation_method      : int
    ---
    estimation_method_name : varchar(50)
    """
    contents = [
        {"estimation_method": 0, "estimation_method_name": "RTMPose"},
    ]

@schema
def handPoseEstimationMethd(dj.Manual):
    definition = """
    -> handBbox
    -> handPoseEstimationMethodLookUp
    ---
    """


@schema
def handPoseEstimation(dj.Computed):
    definition = """
    -> handPoseEstimationMethodLookUp
    ---
    keypoints_2d       : longblob
    """   
    def make(self,key):
        video = SingleCameraVideo.get_robust_reader(key, return_cap=False)
        # keypoints = my_method process the video()