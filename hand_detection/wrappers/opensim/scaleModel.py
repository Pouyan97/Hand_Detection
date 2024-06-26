
import os 
import glob
import numpy as np
import yaml
import traceback

from hand_detection.wrappers.opensim.utilsOpenSim import runScaleTool, getScaleTimeRange, runIKTool, generateVisualizerJson
import argparse


def scale_model(root_dir, trace_file_path, model_path, setup_file_name, locked_joints = []):

    openSimPipelineDir = root_dir       
    
    openSimFolderName = 'Data'
    
    openSimDir = os.path.join(openSimPipelineDir, openSimFolderName)        
    outputScaledModelDir = os.path.join(openSimDir, 'Scaled_Model')


    os.makedirs(openSimDir, exist_ok=True)
    os.makedirs(outputScaledModelDir, exist_ok=True)
    # Path setup file.
    #CHANGEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    genericSetupFile4ScalingName = (setup_file_name)
    pathGenericSetupFile4Scaling = os.path.join(
        openSimPipelineDir, genericSetupFile4ScalingName)
    # Path model file.
    pathGenericModel4Scaling = os.path.join(
        openSimPipelineDir, 
        model_path)            
    # Path TRC file.
    ##CHANGEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    pathTRCFile4Scaling =  trace_file_path
    # Get time range.
    try:
        thresholdPosition = 0.006
        maxThreshold = 0.060
        increment = 0.001
        success = False
        timeRange4Scaling = None
        while thresholdPosition <= maxThreshold and not success:
            try:
                timeRange4Scaling = getScaleTimeRange(
                    pathTRCFile4Scaling,
                    thresholdPosition=thresholdPosition,
                    thresholdTime=0.1, removeRoot=True)
                success = True
            except Exception as e:
                print(f"Attempt with thresholdPosition {thresholdPosition} failed: {e}")
                thresholdPosition += increment  # Increase the threshold for the next iteration
        if timeRange4Scaling is None:
            timeRange4Scaling = [0.8, 1.0]
        # Run scale tool.
        print('Running Scaling')
        pathScaledModel = runScaleTool(
            pathGenericSetupFile4Scaling, 
            pathGenericModel4Scaling,
            5,
            # sessionMetadata['mass_kg'], 
            pathTRCFile4Scaling, 
            timeRange4Scaling, 
            outputScaledModelDir,
            locked_joints = locked_joints
            # subjectHeight=sessionMetadata['height_m'],
            )
    except Exception as e:
        if len(e.args) == 2: # specific exception
            raise Exception(e.args[0], e.args[1])
        elif len(e.args) == 1: # generic exception
            exception = "Musculoskeletal model scaling failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed neutral pose."
            raise Exception(exception, traceback.format_exc())
        else:
            raise Exception(e, traceback.format_exc())
    return pathScaledModel


def IK_model( root_dir, trace_file_path, scaled_model_path, IK_file, locked_joints = []):
    trialName = trace_file_path.split('/')[-1]
    openSimPipelineDir = root_dir
    openSimFolderName = 'Data'
    
    openSimDir = os.path.join(openSimPipelineDir, openSimFolderName)   
    pathScaledModel = scaled_model_path
    pathOutputIK = pathScaledModel[:-5]+'.mot'     
    
    # Inverse kinematics.
    outputIKDir = os.path.join(openSimDir, 'Kinematics')
    os.makedirs(outputIKDir, exist_ok=True)
    # Check if there is a scaled model.
    # pathScaledModel = os.path.join(outputScaledModelDir, 
    #                                 sessionMetadata['openSimModel'] + 
    #                                 "_scaled.osim")
    if os.path.exists(pathScaledModel):
        # Path setup file.
        genericSetupFile4IKName = IK_file
        pathGenericSetupFile4IK = os.path.join(
            openSimPipelineDir, genericSetupFile4IKName)
        # Path TRC file.
        #CHANGE
        pathTRCFile4IK = trace_file_path
        # Run IK tool. 
        print('Running Inverse Kinematics')
        try:
            pathOutputIK = runIKTool(
                pathGenericSetupFile4IK, pathScaledModel, 
                pathTRCFile4IK, outputIKDir,locked_joints = locked_joints)
        except Exception as e:
            if len(e.args) == 2: # specific exception
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1: # generic exception
                exception = "Inverse kinematics failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."
                raise Exception(exception, traceback.format_exc())
    else:
        raise ValueError("No scaled model available.")

    # # Write body transforms to json for visualization.
    # outputJsonVisDir = os.path.join(outputIKDir,'VisualizerJsons',
    #                                 trialName)
    # os.makedirs(outputJsonVisDir,exist_ok=True)
    # outputJsonVisPath = os.path.join(outputJsonVisDir,
    #                                     trialName + '.json')
    # generateVisualizerJson(pathScaledModel, pathOutputIK,
    #                         outputJsonVisPath, 
    #                         vertical_offset=0)  
    return pathOutputIK


def main():

    parser = argparse.ArgumentParser(description='Scale and run inverse kinematics on a model.')
    parser.add_argument('-r', '--root_dir', type=str, help='root Directory where the files are found')
    parser.add_argument('-trc', '--trace_file_path', type=str, help='Absolute path to the TRC file (include .trc)')
    parser.add_argument('-m', '--model_path', type=str, help='Name of the model file (include .osim)')
    parser.add_argument('-sc', '--scale_file', type=str, help='Name of the setup file (include .xml)')
    parser.add_argument('-ik', '--IK_file', type=str, help='Name of the IK file (include .xml)')
    parser.add_argument('-j','--locked_joints', nargs='+', default=['r_x', 'r_y', 'r_z'], help='List of joints that need to be locked')
    args = parser.parse_args()

    pathScaledModel = scale_model(args.root_dir, args.trace_file_path, args.model_path, args.scale_file, args.locked_joints)
    pathOutputIK = IK_model(args.root_dir, args.trace_file_path, pathScaledModel, args.IK_file)

if __name__ == '__main__':
    main()
'''
EXAMPLE RUN: python scaleModel.py -r 'Models' -trc '/home/pfirouzabadi/projects/Hand_Detection/trace_files/m002/m002_trial0_RTMPoseHand5_TopDown_smoothed.trc' \
    -m ARM_Hand_Wrist_Model.osim -sc 27scales.xml -ik 27IK.xml -j r_x r_y r_z
'''
# trace_file_path = '/home/pfirouzabadi/projects/Hand_Detection/trace_files/m002/m002_trial0_RTMPoseHand5_TopDown_smoothed.trc'
# model_path = 'ARM_Hand_Wrist_Model.osim'
# scale_file = '27scales.xml'
# IK_file = '27IK.xml'
# root_dir = 'Models'
# locked_joints = ['r_x','r_y','r_z']

# pathScaledModel = scale_model(root_dir, trace_file_path, model_path, scale_file)

# IK_model( root_dir, trace_file_path, pathScaledModel, IK_file)
