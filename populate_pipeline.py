import os
from pose_pipeline.utils.jupyter import play,play_grid
from pose_pipeline.pipeline import BlurredVideo,LiftingPerson,LiftingMethod,TopDownPerson,Video
from multi_camera.datajoint.multi_camera_dj import PersonKeypointReconstruction,SingleCameraVideo, CalibratedRecording, MultiCameraRecording,PersonKeypointReconstructionMethod
from multi_camera.datajoint.sessions import Recording
from hand_detection.hand_dj import HandPoseReconstructionMethodLookup,HandPoseReconstructionMethod, HandPoseReconstruction,HandPoseEstimationMethodLookup
from hand_detection.hand_dj import HandBbox,HandBboxMethod, HandPoseEstimation,HandPoseEstimationMethod,HandPoseEstimationVideo,MJXReconstruction


def populate_hand_bbox(keys, detection_methods = [1]):
    for detection_method in detection_methods:
        for k in keys:
            k['detection_method'] = detection_method 
        HandBboxMethod.insert(keys,skip_duplicates=True)
        HandBbox.populate(keys,reserve_jobs=True)

def populate_hand_estimation(keys, detection_methods= [1], estimation_methods = [-1]): 
    for detection_method in detection_methods:
        for estimation_method in estimation_methods:
            for k in keys:
                k['detection_method'] = detection_method 
                k['estimation_method'] = estimation_method 
            HandPoseEstimationMethod.insert(keys,skip_duplicates=True)
            HandPoseEstimation.populate(keys,reserve_jobs=True)

def populate_hand_reconstruction(keys, detection_methods=[1], estimation_methods = [-1], reconstruction_methods =[3]):
    for detection_method in detection_methods:
        for estimation_method in estimation_methods:
            for reconstruction_method in reconstruction_methods:
                for k in keys:
                    k['detection_method'] = detection_method
                    k['estimation_method'] = estimation_method
                    k['reconstruction_method'] = reconstruction_method 
                HandPoseReconstructionMethod.insert(keys,skip_duplicates=True)
                HandPoseReconstruction.populate(keys,reserve_jobs=True)


keys_list = ['filename LIKE "m002%"']
detection_methods = [1,2]
estimation_methods = range(-1,5)
reconstruction_methods = [3]

for name in keys_list:
    recording_keys = (Recording & (SingleCameraVideo & name )).fetch('KEY')
    for rk in recording_keys:
        keys = (SingleCameraVideo & rk ).fetch('KEY')
        print("Processing", len(keys), "Keys")
        populate_hand_bbox(keys, detection_methods = detection_methods)
        populate_hand_estimation(keys,  detection_methods= detection_methods, estimation_methods = estimation_methods)
        # # populate pose reconstruction as well       
        key = (CalibratedRecording & (HandPoseEstimation & keys)).fetch('KEY')
        populate_hand_reconstruction(key, detection_methods = detection_methods, 
                                     estimation_methods = estimation_methods, 
                                     reconstruction_methods = reconstruction_methods)
