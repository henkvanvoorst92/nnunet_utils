
import os
from .utils import write_envlines_nnunet


#code for single training single fold
def train_single_model(gpu,datasetID,resolution,fold):
    cmd = f'CUDA_VISIBLE_DEVICES={gpu} nnUNetv2_train {datasetID} {resolution} {fold}'
    print('command to run:',cmd)
    os.system(cmd)

def nnunet_train_shell(datasetID,
                       root: str,
                       conda_env: str,
                       gpu_res_fold_dct: dict = None, #is created with utils function assign_to_gpu
                       version=2):
    """
    create shell scripts to train each fold on another device

    dev_res_fold_dct: a dictionary gpu:[resolution,fold number]
    where resolution can be ['2d','3d_fullres','3d_lowres', '3d_cascade_fullres']
    the dictionary is used to divide the jobs over different gpus
    """
    job_file = os.path.join(root, 'train_jobs.sh')

    with open(job_file, 'w') as f:
        # create a shell script to run on a server
        # alternatively run a single python command
        f.writelines('#!/bin/bash\n')
        # for job scheduler only
        #         for key,value in job_dct.items():
        #             f.writelines('#SBATCH --{}={}\n'.format(key,value))

        # conda env activation
        f.writelines('module purge\n')
        f.writelines('eval "$(conda shell.bash hook)"\n')
        f.writelines('conda activate {}\n'.format(conda_env))
        f.writelines('\n')

        write_envlines_nnunet(file=f,
                              root=root,
                              version=version)
        f.writelines('\n')
        # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md
        for gpu, data in gpu_res_fold_dct.items():
            for [resolution, fold] in data:
                f.writelines(
                    f'CUDA_VISIBLE_DEVICES={gpu} nnUNetv2_train {datasetID} {resolution} {fold} --npz &\n')

        f.writelines("wait")
    return job_file


def nnunetv2_inference_shell(root: str,
                             conda_env: str,
                             gpu_dct: dict,  # is created with utils function assign_to_gpu
                             path_model: str,
                             dir_output_seg: str,
                             return_probabilities: bool = True,
                             path_nnunet_utils='',
                             version=2):
    """
    create shell scripts for across gpu parallel inference for a folder of images (root)

    root: root directory of nnUnet model
    conda_env: anaconda environment with the right requirements installed (nnunetv2)
    gpu_dct: output of gpu_distributed_inference function in .utils
             key=gpu_number, value=data with resolution,input_images
    path_model: directory with trained nnUnet folds
    dir_output_seg: directory to store the segmentation results
    return_probabilities: if True returns .npy probability files
    version: nnUnet version (does not work yet for old version only for 2)
    """
    job_file = os.path.join(root, 'inference_jobs.sh')

    with open(job_file, 'w') as f:
        # create a shell script to run on a server
        # alternatively run a single python command
        f.writelines('#!/bin/bash\n')

        # conda env activation
        # f.writelines('module purge\n') #does not seem to work
        f.writelines('eval "$(conda shell.bash hook)"\n')
        f.writelines('conda activate {}\n'.format(conda_env))
        f.writelines('\n')

        write_envlines_nnunet(file=f,
                              root=root,
                              version=version)

        f.writelines(f"export PYTHONPATH={path_nnunet_utils}:$PYTHONPATH\n")
        f.writelines('\n')

        # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md
        for gpu, data in gpu_dct.items():
            for [resolution, input_images] in data:
                line = f"CUDA_VISIBLE_DEVICES={gpu}"
                if isinstance(input_images, list) and os.path.isfile(input_images[0]):
                    images = ''.join([img if ix==0 else " " + img for ix,img in enumerate(input_images)])
                    line += f" python nnunet_utils/infv2.py"
                    line += f" --path_model {path_model}"
                    line += f" --images \"{images}\""
                    line += f" --seg_dir {dir_output_seg}"
                    if return_probabilities:
                        line += f" --return_probabilities"
                elif os.path.isdir(input_images):
                    line += f" nnUNetv2_predict -i {input_images}"
                    line += f" -o {dir_output_seg}"
                    line += f" -c {resolution}"
                    line += f" -t {path_model}"
                    if return_probabilities:
                        line += f" --save_probabilities"

                f.writelines(line)
                f.writelines('\n')
                f.writelines("wait")

    return job_file


def nnunet_inference_on_dir(model_path,  # path to trained models
                            dir_input_images,  # directory with input images
                            dir_output_seg,  # output directory
                            resolution,  # 3d_fullres
                            save_probs=True,
                            version=2,
                            run=False,
                            task=None
                            ):
    """
    Runs an nnUNetV2 inference command using os.system on a directory of test images

    :param model_path: The path to the trained model directory.
    :param input_images: The path to the directory containing input images.
    :param output_folder: The path to the directory where output will be saved.

    see also:
    https://github.com/DIAGNijmegen/nnUNet_v2/blob/master/documentation/how_to_use_nnunet.md
    """

    if version == 2:
        # Construct the nnUNetV2 inference command based on your specific requirements
        command = f"nnUNetv2_predict -i {dir_input_images}"
        command += f"-o {dir_output_seg}"
        command += f"-t {model_path}"
        command += f" -c {resolution}"
        #command += f" -d {task}"

        if save_probs:
            command += f" --save_probabilities"

        if run:
            print('Start running command')
            print(command)
            status = os.system(command)

            if status != 0:
                print("Error executing the command")
            else:
                print("Command executed successfully")

    else:
        # run the old version of nnunet
        command = []
        # nnUNet_plan_and_preprocess -t TaskXX_MYTASK --verify_dataset_integrity
        cmd = f"nnUNet_plan_and_preprocess-t {task} --verify_dataset_integrity"
        command.append(cmd)
        if run:
            status = os.system(cmd)
            if status != 0:
                print("Error executing the command")
            else:
                print("Command executed successfully")
            status = os.system(cmd)

        # nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TaskXX_MYTASK -m 3d_fullres
        cmd = f"nnUNet_predict -i {dir_input_images}"
        cmd += f"-o {dir_output_seg}"
        cmd += f"-t {task}"
        cmd += f" -m {resolution}"
        command.append(cmd)
        if run:
            status = os.system(cmd)
            if status != 0:
                print("Error executing the command")
            else:
                print("Command executed successfully")

    if not run:
        return command