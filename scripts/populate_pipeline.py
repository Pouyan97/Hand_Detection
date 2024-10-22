import argparse
import os
from pose_pipeline.utils.jupyter import play,play_grid
from pose_pipeline.pipeline import BlurredVideo,LiftingPerson,LiftingMethod,TopDownPerson,Video
from multi_camera.datajoint.multi_camera_dj import PersonKeypointReconstruction,SingleCameraVideo, CalibratedRecording, MultiCameraRecording,PersonKeypointReconstructionMethod
from multi_camera.datajoint.sessions import Recording
from hand_detection.hand_dj import HandPoseReconstructionMethodLookup,HandPoseReconstructionMethod, HandPoseReconstruction,HandPoseEstimationMethodLookup
from hand_detection.hand_dj import HandBbox,HandBboxMethod, HandPoseEstimation,HandPoseEstimationMethod,HandPoseEstimationVideo,MJXReconstruction

# from hand_detection.hand_dj import schema
# schema.jobs.delete()

def populate_hand_bbox(keys, detection_methods = [1]):
    for detection_method in detection_methods:
        for k in keys:
            k['detection_method'] = detection_method 
        HandBboxMethod.insert(keys,skip_duplicates=True)
        print("HandBbox Populate ", 
              ' '.join((SingleCameraVideo & k).fetch('filename')[0].split('_')[:2]),
              detection_method)
        HandBbox.populate(keys)

def populate_hand_estimation(keys, detection_methods= [1], estimation_methods = [-1]): 
    for detection_method in detection_methods:
        for estimation_method in estimation_methods:
            for k in keys:
                k['detection_method'] = detection_method 
                k['estimation_method'] = estimation_method     
            HandPoseEstimationMethod.insert(keys,skip_duplicates=True)
            print("Populating Hand Estimation ", 
                  ' '.join((SingleCameraVideo & k).fetch('filename')[0].split('_')[:2]),
                  detection_method, estimation_method)
            HandPoseEstimation.populate(keys, reserve_jobs=True)

def populate_hand_reconstruction(keys, detection_methods=[1], estimation_methods = [-1], reconstruction_methods =[3]):
    for detection_method in detection_methods:
        for estimation_method in estimation_methods:
            for reconstruction_method in reconstruction_methods:
                for k in keys:
                    k['detection_method'] = detection_method
                    k['estimation_method'] = estimation_method
                    k['reconstruction_method'] = reconstruction_method 
                HandPoseReconstructionMethod.insert(keys,skip_duplicates=True)
                print("Hand Reconstruction Populate ", 
                      ' '.join((SingleCameraVideo & k).fetch('filename')[0].split('_')[:2]),
                      detection_method, estimation_method)
                HandPoseReconstruction.populate(keys,reserve_jobs=True)


def main():
    parser = argparse.ArgumentParser(description='Populate hand bbox, estimation, and reconstruction.')
    parser.add_argument('--participants', type=str, nargs='+', required=True, help='List of participants')
    parser.add_argument('--filenames', type=str, nargs='+', required=True, help='List of filenames')
    parser.add_argument('--detection_methods', type=int, nargs='+', default=[1,2], help='Detection methods')
    parser.add_argument('--estimation_methods', type=int, nargs='+', default=[-1,0,1], help='Estimation methods')
    parser.add_argument('--reconstruction_methods', type=int, nargs='+', default=[3], help='Reconstruction methods')
    parser.add_argument('--populate', choices=['bbox', 'estimation', 'reconstruction', 'all'], required=True, help='Choose what to populate')

    args = parser.parse_args()
    
    # participants = ["yj843","rko5c","8wj64","lgtfc","oxbcl"]
    # participants = ["oxbcl"]
    # filenames = ['_%']
    # detection_methods = [2]
    # estimation_methods = [-1,0,1,2,3,4]
    # reconstruction_methods=[3]
    participants = args.participants
    filenames = args.filenames
    detection_methods = args.detection_methods
    estimation_methods = args.estimation_methods
    reconstruction_methods = args.reconstruction_methods
    populate = args.populate

    for participant_id in participants:
        for filename in filenames:
            # participant_videos = (Recording & f'participant_id="{participant_id}"').fetch('KEY')
            participant_videos = (Recording & (SingleCameraVideo & f'filename LIKE "{participant_id}%{filename}%"')).fetch('KEY')
            for rk in participant_videos:
                keys = (SingleCameraVideo & rk ).fetch('KEY')
                print("Processing", len(keys), "Keys")
                if populate == 'bbox' or populate == 'all':
                    populate_hand_bbox(keys, detection_methods = detection_methods)
                if populate == 'estimation' or populate == 'all':
                    populate_hand_estimation(keys,  detection_methods= detection_methods, estimation_methods = estimation_methods)
                # populate pose reconstruction as well       
                if populate == 'reconstruction' and populate == 'all':
                    key = (CalibratedRecording & (HandPoseEstimation & keys)).fetch('KEY')
                    populate_hand_reconstruction(key, detection_methods = detection_methods, 
                                                estimation_methods = estimation_methods, 
                                                reconstruction_methods = reconstruction_methods)

if __name__ == '__main__':
    main()