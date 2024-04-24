import os
from pose_pipeline.utils.jupyter import play,play_grid
from pose_pipeline.pipeline import BlurredVideo,LiftingPerson,LiftingMethod,TopDownPerson,Video
from multi_camera.datajoint.multi_camera_dj import PersonKeypointReconstruction,SingleCameraVideo, CalibratedRecording, MultiCameraRecording,PersonKeypointReconstructionMethod
from multi_camera.datajoint.sessions import Recording
from hand_detection.hand_dj import HandPoseReconstructionMethodLookup,HandPoseReconstructionMethod, HandPoseReconstruction,HandPoseEstimationMethodLookup
from hand_detection.hand_dj import HandBbox,HandBboxMethod, HandPoseEstimation,HandPoseEstimationMethod,HandPoseEstimationVideo,MJXReconstruction

def populate_MJX_reconstruction(keys):
    MJXReconstruction.populate(keys,reserve_jobs=True)

detection_methods = [1,2]
estimation_methods = range(-1,5)
# reconstruction_methods = [3]
keys_list = ['filename LIKE "m002%"','filename LIKE "p40_rom_right_20221018_161656%"']
for name in keys_list:
    recording_keys = (Recording & (SingleCameraVideo & name )).fetch('KEY')
    key = (HandPoseReconstruction & (HandBbox & name)).fetch('KEY')
    for detection_method in detection_methods:
        for estimation_method in estimation_methods:
            for k in key:
                k['estimation_method'] = estimation_method
                k['detection_method'] = detection_method
            populate_MJX_reconstruction(key)
