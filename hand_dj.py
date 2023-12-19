import datajoint as dj
import numpy as np

from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo #, PersonReconstruction


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
            from wrappers.hand_bbox import mmpose_hand_det
            num_boxes, bboxes = mmpose_hand_det(key=key, method="RTMDet")
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
            from wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'RTMPoseHand5')
        elif (HandPoseEstimationMethodLookup & key).fetch1("estimation_method_name") == "RTMPoseCOCO":
            from wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'RTMPoseCOCO')
        elif (HandPoseEstimationMethodLookup & key).fetch1("estimation_method_name") == "freihand":
            from wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'freihand')
        elif (HandPoseEstimationMethodLookup & key).fetch1("estimation_method_name") == "HRNet_dark":
            from wrappers.hand_estimation import mmpose_HPE
            key["keypoints_2d"] = mmpose_HPE(key, 'HRNet_dark')
        
        
        self.insert1(key)


        
# @schema
# class HandPoseEstimationVideo(dj.Computed):
#     definition = """
#     -> HandPoseEstimation
#     ---
#     output_video      : attach@localattach    # datajoint managed video file
#     """   
#     def make(self,key):
        # keypoints = (HandPoseEstimation & key).fetch1("keypoints2d")
        # vid_file = (Video & key).fetch('video')[0]        
        # keypoints_2d = (HandPoseEstimation & vid_file).fetch1("keypoints_2d")
#         fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
#         os.close(fd)
        # def render_video(video, output_file, keypoints):
        #     cap = cv2.VideoCapture(video)
        #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     fps = cap.get(cv2.CAP_PROP_FPS)
        #     output_size = (int(w),int(h))

        #     fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        #     out = cv2.VideoWriter(output_file,fourcc, fps,output_size)

        #     for frame_idx in tqdm(range(total_frames)):
        #         success, frame = cap.read()

        #         if not success:
        #             break
        #         keypoints = keypoints[frame_idx,:,:].copy()
        #         frame = draw_keypoints(frame,keypoints)
        #         out.write(frame)

        #     out.release()
        #     cap.release()
        # os.remove(vid_file)

#         key["output_video"] = out_file_name
#         self.insert1(key)
