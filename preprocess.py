
import os
#import sys
import SimpleITK as sitk

def write_as_nnunet(IMG, GT, p_dir, ID):
    """
    Writes training image and label (IMG and GT)
    in a nnunet file structure inside p_dir directory
    """
    if IMG is not None:
        p_img_new = os.path.join(p_dir, 'imagesTr')
        if not os.path.exists(p_img_new):
            os.makedirs(p_img_new)
        sitk.WriteImage(IMG ,os.path.join(p_img_new, ID+'_0000.nii.gz'))

    if GT is not None:
        p_gt_new = os.path.join(p_dir , 'labelsTr')
        if not os.path.exists(p_gt_new):
            os.makedirs(p_gt_new)
        sitk.WriteImage(GT, os.path.join(p_gt_new, ID+'.nii.gz'))

def nnunet_directory_structure(base_dir):
    """
    Create the directory structure for nnU-Net.

    Parameters:
    base_dir (str): Base directory where the nnU-Net structure will be set up.

    The structure will be as follows:
    nnUNet/
        nnUNet_raw/
            nnUNet_raw_data/
            nnUNet_cropped_data/
        nnUNet_preprocessed/
        nnUNet_trained_models/
    """
    paths = [
        "nnUNet_raw/nnUNet_raw_data/TaskXXX",
        "nnUNet_raw/nnUNet_cropped_data",
        "nnUNet_preprocessed",
        "nnUNet_trained_models"
    ]

    for path in paths:
        p = os.path.join(base_dir, path)
        if not os.path.exists(p):
            os.makedirs(p)

def script_command(file,
                   file_args,  # arguments of the file used
                   ):
    command = 'python -m ' + file

    for key, value in file_args.items():

        if key == 'sav' or key == 'cbe':
            # command+= ' --{}={}'.format(key,value)
            command += ' --{}'.format(key)
            continue
        elif value == '':
            continue
        command += ' --{}={}'.format(key, value)
    print('Script command defined:\n', command)
    return command



