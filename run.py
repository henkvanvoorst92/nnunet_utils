
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

