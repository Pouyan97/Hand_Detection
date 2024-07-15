import os
from pose_pipeline.utils.jupyter import play,play_grid
from pose_pipeline.pipeline import BlurredVideo,LiftingPerson,LiftingMethod,TopDownPerson,Video
from multi_camera.datajoint.multi_camera_dj import PersonKeypointReconstruction,SingleCameraVideo, CalibratedRecording, MultiCameraRecording,PersonKeypointReconstructionMethod
from multi_camera.datajoint.sessions import Recording
from hand_detection.hand_dj import HandPoseReconstructionMethodLookup,HandPoseReconstructionMethod, HandPoseReconstruction,HandPoseEstimationMethodLookup
from hand_detection.hand_dj import MJXReconstructionMethod, HandBbox,HandBboxMethod, HandPoseEstimation,HandPoseEstimationMethod,HandPoseEstimationVideo,MJXReconstruction

participants = ["yj843","rko5c","8wj64","lgtfc"]
filenames = ['\_A\_%']
for participant_id in participants:
    for filename in filenames:
        detection_methods = [1]
        estimation_methods = [-1,0,1,2,3,4]
        participant_videos = (Recording & (SingleCameraVideo & f'filename LIKE "{participant_id}{filename}"')).fetch('KEY')
        for rk in participant_videos:
            keys = (CalibratedRecording &(SingleCameraVideo & rk) ).fetch('KEY')
            for estimation_method in estimation_methods:
                for k in keys:
                    k['detection_method'] = 1 
                    k['estimation_method'] = estimation_method     
                print(keys)
                MJXReconstructionMethod.insert(keys,skip_duplicates=True)
                MJXReconstruction.populate(keys)        