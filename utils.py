
import os
import shutil
import numpy as np
import SimpleITK as sitk

def np2sitk(arr: np.ndarray, original_img: sitk.SimpleITK.Image):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(original_img.GetSpacing())
    img.SetOrigin(original_img.GetOrigin())
    img.SetDirection(original_img.GetDirection())
    # this does not allow cropping (such as removing thorax, neck)
    #img.CopyInformation(original_img)
    return img

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

#used for assigning training jobs to gpus
def assign_trainjobs_to_gpus(num_gpus,
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
    else:
        num_gpus = num_gpus #a list with the numbers for gpus to use

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

#used for inference file batches
def split_files_into_batches(file_paths, num_batches):
    """
    Splits a list of file paths into equal-sized batches.

    :param file_paths: A list of file paths to be split.
    :param num_batches: The number of batches to split the file paths into.
    :return: A list of batches, where each batch is a list of file paths.
    """
    # Calculate the size of each batch
    batch_size = len(file_paths) // num_batches
    remainder = len(file_paths) % num_batches

    # Create the batches
    batches = []
    for i in range(num_batches):
        start_index = i * batch_size + min(i, remainder)
        end_index = start_index + batch_size + (1 if i < remainder else 0)
        batches.append(file_paths[start_index:end_index])
    return batches

def gpu_distributed_inference(images,
                              num_gpus,
                              resolutions,
                              separate_folders=False,
                              seg_dir=None):
    """
    Assigns images to GPUs based on criteria defined by resolution arguments in a round-robin fashion.

    :param images: list of paths to images or directory with nnUnet styled inference images
    :param num_gpus: The number of available GPUs.
    :param resolutions: A list of strings representing arguments for a function that checks if an image meets certain criteria.
    :param separate_folders: if true creates separate folders and copies image batches
    :param seg_dir: Can be provided as path to existing segmentations, skips image if segmentation exists
    :return: A dictionary where keys are GPU IDs and values are lists of image paths assigned to that GPU.
    """

    if len(resolutions) == 0:
        resolutions = ['no_resolutions']

    # If num_gpus is an integer, convert it to a list of GPU IDs
    if isinstance(num_gpus, int):
        num_gpus = list(range(num_gpus))
    else:
        num_gpus = num_gpus  # a list with the numbers for gpus to use

    if os.path.isdir(images):
        # identify images with path in a single nnUnet style folder
        img_paths = []
        for image_name in os.listdir(images):
            image_path = os.path.join(images, image_name)
            # skip images that are already segmented
            if seg_dir is not None:
                ID = image_name.split('_')[0]
                seg_path = os.path.join(image_dir, ID + '.nii.gz')
                if os.path.exists(seg_path):
                    continue
            img_paths.append(image_path)

    elif isinstance(images, list) and os.path.isfile(images[0]):
        # images can also be a list of files
        img_paths = images
        # pm include skipping of files with segmentation

    total_gpus = len(num_gpus)
    total_resolutions = len(resolutions)

    img_batches = split_files_into_batches(img_paths, total_gpus * total_resolutions)

    # for nnunetv1 you could create separate folders
    # that can be assigned to gpus to run
    if separate_folders and os.path.isdir(images):
        root = os.path.join(images.split(os.sep)[:-1])
        for i, batch in enumerate(img_batches):
            p_batch = os.path.join(root, 'batch_{}'.format(i))
            if not os.path.exists(p_batch):
                os.makedirs(p_batch)
            for p_img in batch:
                shutil.copy2(p_img, p_batch)

    # assign jobs to gpu numbers
    job_counter = 0
    gpu_assignments = {gpu_id: [] for gpu_id in num_gpus}
    for batch in img_batches:
        for resolution in resolutions:
            gpu_id = num_gpus[job_counter % total_gpus]
            if resolution == 'no_resolutions':
                gpu_assignments[gpu_id].append(batch)
            else:
                gpu_assignments[gpu_id].append((resolution, batch))
            job_counter += 1

    return gpu_assignments

def copy_inference_image(path_image_in, folder_image_out):
    ID = path_image_in.split(os.sep)[-2]

    if not os.path.exists(folder_image_out):
        os.makedirs(folder_image_out)

    p_out = os.path.join(folder_image_out, ID + '_0000.nii.gz')
    shutil.copy2(path_image_in, p_out)


def copy_seg_to_original_folder(nnunet_seg_folder, original_folder, addname='-CTA_vesselseg'):
    # copy nnunet segmentations back to original folder
    for f in os.listdir(nnunet_seg_folder):
        ID = f.split(".")[0]
        if '_0000' in ID:
            ID = ID.split('_0000')[0]
        pid = os.path.join(original_folder, ID)
        if not os.path.exists(pid):
            os.makedirs(pid)

        f_nnunet = os.path.join(nnunet_seg_folder, f)

        if '.nii.gz' in f:
            f_original = os.path.join(pid, '{}{}.nii.gz'.format(ID, addname))
        elif '.pkl' in f:
            f_original = os.path.join(pid, '{}{}.pkl'.format(ID, addname))
        elif '.npz' in f:
            f_original = os.path.join(pid, '{}{}.npz'.format(ID, addname))

        shutil.copy2(f_nnunet, f_original)
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


