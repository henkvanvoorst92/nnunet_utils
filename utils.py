
import os

def set_env_nnunet(root: str, version=2):
    # set nnUnet environment path
    if version >= 2:
        os.environ['nnUNet_raw'] = os.path.join(root, 'nnUNet_raw')
        os.environ['nnUNet_results'] = os.path.join(root, 'nnUNet_trained_models')
    else:
        os.environ['nnUNet_raw_data_base'] = os.path.join(root, 'nnUNet_raw_data_base')
        os.environ['RESULTS_FOLDER'] = os.path.join(root, 'nnUNet_trained_models')
    os.environ['nnUNet_preprocessed'] = os.path.join(root, 'nnUNet_preprocessed')

    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    print('------------ environment set ------------')

def write_envlines_nnunet(file, root: str, version=2):
    # set nnUnet environment path
    if version >= 2:
        file.writelines('export nnUNet_raw={}\n'.format(os.path.join(root, 'nnUNet_raw')))
        file.writelines('export nnUNet_results={}\n'.format(os.path.join(root, 'nnUNet_trained_models')))
    else:
        file.writelines('export nnUNet_raw_data_base={}\n'.format(os.path.join(root, 'nnUNet_raw_data_base')))
        file.writelines('export RESULTS_FOLDER={}\n'.format(os.path.join(root, 'nnUNet_trained_models')))

    file.writelines('export nnUNet_preprocessed={}\n'.format(os.path.join(root, 'nnUNet_preprocessed')))
    file.writelines('export MKL_THREADING_LAYER={}\n'.format(os.path.join(root, 'GNU')))

def assign_to_gpus(num_gpus,
                   num_folds,
                   resolutions):
    """
    Assigns each fold of different nnU-Net model configurations to a GPU in a round-robin fashion.

    :param resolutions: A list of nnU-Net model configurations (e.g., ['2d','3d_fullres','3d_lowres', '3d_cascade_fullres']).
    :param num_folds: The total number of folds for each configuration.
    :param num_gpus: Either the number of available GPUs or a list of GPU IDs.
    :return: A dictionary where keys are GPU IDs and values are lists of tuples (configuration, fold ID) assigned to that GPU.
    """

    if len(resolutions) == 0:
        resolutions = ['no_resolutions']

    # If num_gpus is an integer, convert it to a list of GPU IDs
    if isinstance(num_gpus, int):
        num_gpus = list(range(num_gpus))

    job_counter = 0
    gpu_assignments = {gpu_id: [] for gpu_id in num_gpus}
    total_gpus = len(num_gpus)
    for resolution in resolutions:
        for fold_id in range(num_folds):
            gpu_id = num_gpus[job_counter % total_gpus]
            if resolution == 'no_resolutions':
                gpu_assignments[gpu_id].append(fold_id)
            else:
                gpu_assignments[gpu_id].append((resolution, fold_id))
            job_counter += 1

    return gpu_assignments

def copy_inference_image(path_image_in, folder_image_out):
    ID = path_image_in.split(os.sep)[-2]

    if not os.path.exists(folder_image_out):
        os.makedirs(folder_image_out)

    p_out = os.path.join(folder_image_out, ID + '_0000.nii.gz')
    shutil.copy2(path_image_in, p_out)

def copy_seg_to_original_folder(nnunet_seg_folder,original_folder,addname='-CTA_vesselseg'):
    #copy nnunet segmentations back to original folder
    for f in os.listdir(nnunet_seg_folder):
        ID = f.split(".")[0]
        pid = os.path.join(original_folder,ID)
        if not os.path.exists(pid):
            os.makedirs(pid)
        f_nnunet = os.path.join(nnunet_seg_folder, f)
        f_original = os.path.join(pid,'{}{}.nii.gz'.format(ID, addname))
        shutil.copy2(f_nnunet,f_original)
    return

#
# def assign_folds_to_gpus(num_folds, num_gpus):
#     """
#     Assigns each fold to a GPU in a round-robin fashion.
#
#     :param num_folds: The total number of folds to train.
#     :param num_gpus: The number of available GPUs.
#     :return: A dictionary where keys are GPU IDs and values are lists of fold IDs assigned to that GPU.
#     """
#     gpu_assignments = {gpu_id: [] for gpu_id in range(num_gpus)}
#
#     for fold_id in range(num_folds):
#         gpu_id = fold_id % num_gpus
#         gpu_assignments[gpu_id].append(fold_id)
#
#     return gpu_assignments


