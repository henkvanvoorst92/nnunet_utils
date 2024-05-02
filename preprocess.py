
import os
import numpy as np
import SimpleITK as sitk
import nnunetv2
import shutil

from .utils import set_env_nnunet, write_envlines_nnunet

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


def nnunet_directory_structure(base_dir,
                               taskname=None,
                               version: [str or int or float] = None):
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
        "nnUNet_raw/nnUNet_cropped_data",
        "nnUNet_preprocessed",
        "nnUNet_trained_models"
    ]

    if version is None:
        version = 2

    if version == 2:
        paths.append(os.path.join("nnUNet_raw", taskname))
    elif version != 2:
        # add old version file order
        if taskname is None:
            taskname = "TaskXXX"
        paths.append(os.path.join("nnUNet_raw/nnUNet_raw_data", taskname))

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

def img_lbl_paircount(root_images: str, root_gt: str):
    img_IDs = [f.split('_')[0] for f in os.listdir(root_images)]
    gt_IDs = [f.split('.')[0] for f in os.listdir(root_gt)]
    if np.all(np.isin(gt_IDs, img_IDs)):
        print('all images have a gt')
    return int(np.isin(gt_IDs, img_IDs).sum())

def preprocess_data(root: str,
                    datano: [int or str or float],  # 501
                    datasetID: str,  # Dataset501_XXX
                    dataset_name: str,
                    modalities: list[str],
                    configs: list[str] = None, # ['2d', '3d_fullres', '3d_lowres'],
                    verify_integrity=True,
                    plan_and_preprocess='nnUNetv2_plan_and_preprocess',
                    version: [int or str or float] = 2):
    # prepares data for training
    if version >= 2:
        p_data = os.path.join(root, 'nnUNet_raw', datasetID)
    else:
        p_data = os.path.join(root, 'nnUNet_raw_data_base', 'nnUNet_raw_data', datasetID)

    f_out = os.path.join(p_data, 'dataset.json')
    d_tr = os.path.join(p_data, 'imagesTr')
    d_lbl = os.path.join(p_data, 'labelsTr')
    num_tr = img_lbl_paircount(root_images=d_tr, root_gt=d_lbl)

    set_env_nnunet(root, version=version)

    if version >= 2:
        from nnunetv2.dataset_conversion import generate_dataset_json
        generate_dataset_json.generate_dataset_json(
                                    output_folder=p_data,
                                    num_training_cases=num_tr,
                                    channel_names={m: str(c) for c, m in enumerate(modalities)},  # 'synMRA'
                                    labels={'background': 0, 'foreground': 1},
                                    file_ending=".nii.gz",
                                    dataset_name=dataset_name,
                                    imagesTr_dir=d_tr,
                                    imagesTs_dir=None,
                                    # modalities=modalities,
                                    license='hands off!',
                                    dataset_description="dataset nnUnet",
                                    overwrite_image_reader_writer="SimpleITKIO"
                                )

        cmd_pp = f'{plan_and_preprocess} -d {datano}'
        if configs is not None:
            if isinstance(configs, list):
                configs = ' '.join(configs)
            cmd_pp += f' -c {configs}'

        if verify_integrity:
            cmd_pp += ' --verify_dataset_integrity'

    else:
        from nnunet.dataset_conversion.utils import generate_dataset_json
        generate_dataset_json(output_file=f_out,
                              imagesTr_dir=d_tr,
                              imagesTs_dir=None,
                              modalities=(modalities),  # 'synMRA'
                              labels={0: ' background', 1: 'foreground'},
                              dataset_name=dataset_name,
                              license='hands off!',
                              dataset_description="dataset nnUnet"
                              )
        cmd_pp = 'nnUNet_plan_and_preprocess -t {}'.format(datano)  # --verify_dataset_integrity

    os.system(cmd_pp)
    print('finished preprocessing')

def create_inference_data_from_folder(path_folder_in, path_folder_out, bounds=None):
    for file in os.listdir(path_folder_in):
        if not '.nii' in file:
            continue
        # creat in and out files and skip if out file exists
        f = os.path.join(path_folder_in, file)
        ID = file.split('.')[0]
        file_out = ID + '_0000.nii.gz'
        path_out = os.path.join(path_folder_out, file_out)
        if os.path.exists(path_out):
            continue
        # write file out, create folder if it does not exist
        if not os.path.exists(path_folder_out):
            os.makedirs(path_folder_out)
        create_inference_image(f, path_out, bounds=None)


def create_inference_image(path_image_in, path_image_out, bounds=None):
    # read image
    img = sitk.ReadImage(path_image_in)
    # preprocess image
    img = sitk.Cast(img, sitk.sitkInt16)
    if bounds is not None:
        img = sitk.Clamp(img, lowerBound=bounds[0], upperBound=bounds[1])

    sitk.WriteImage(img, path_image_out)


def plan_additional_experiment(root: str,
                               datano: [int or str or float],  # 501
                               datasetID: str,  # Dataset501_XXX
                               configs: list[str] = None,  # ['2d', '3d_fullres', '3d_lowres'],
                               verify_integrity=True,
                               plan_and_preprocess='nnUNetv2_plan_experiment',
                               model='nnUNetPlannerResEnc(M/L/XL)'
                               ):
    """
    Plans additional experiment for non-original nnUNet models
    """

    p_data = os.path.join(root, 'nnUNet_raw', datasetID)

    set_env_nnunet(root, version=2)

    cmd_pp = f'{plan_and_preprocess} -d {datano}'
    if model is not None:
        cmd_pp += f' -pl {model}'

    if configs is not None:
        if isinstance(configs, list):
            configs = ' '.join(configs)
        cmd_pp += f' -c {configs}'

    if verify_integrity:
        cmd_pp += ' --verify_dataset_integrity'

    os.system(cmd_pp)
    print(cmd_pp)
    print('finished preprocessing')


