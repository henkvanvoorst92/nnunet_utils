

#from nnunetv2.paths import nnUNet_results, nnUNet_raw
#from batchgenerators.utilities.file_and_folder_operations import join
#from nnunetv2.inference import predict_from_raw_data
import os
import sys
import SimpleITK as sitk
import argparse
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunet_utils.utils import np2sitk

def init_predictor_ensemble(path_model, device=torch.device('cuda', 0)):
    #from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        path_model,
        use_folds=(0 ,1 ,2 ,3 ,4),
        checkpoint_name='checkpoint_best.pth',
    )
    return predictor

def init_single_predictor(path_model, fold, modelname, device=torch.device('cuda', 0)):
    #from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        #perform_everything_on_gpu=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        path_model,
        use_folds=([fold]),
        checkpoint_name=modelname,
    )
    return predictor



def nnunetv2_get_props(IMG):
    # can be used for inline props extraction
    # required as input for nnunetv2 predictor on npy
    props = {
        'sitk_stuff': {
            # this saves the sitk geometry information. This part is NOT used by nnU-Net!
            'spacing': IMG.GetSpacing(),
            'origin': IMG.GetOrigin(),
            'direction': IMG.GetDirection()
        },
        # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
        # are returned x,y,z but spacing is returned z,y,x. Duh.
        'spacing': list(IMG.GetSpacing())[::-1]
    }
    return props


def nnunetv2_predict(img: sitk.Image,
                     predictor,
                     return_probabilities=False,
                     use_iterator=False
                     ):
    # predictor is a nnuentv2 object where inference is defined
    props = nnunetv2_get_props(img)
    img = np.expand_dims(sitk.GetArrayFromImage(img), 0).astype(np.float16)

    if use_iterator:
        iterator = predictor.get_data_iterator_from_raw_npy_data([img], None, [props], None, 1)
        seg = predictor.predict_from_data_iterator(iterator, return_probabilities, 1)

    else:
        seg = predictor.predict_single_npy_array(input_image=img, image_properties=props,
                                                 # segmentation_previous_stage= None,
                                                 # output_file_truncated= None,
                                                 save_or_return_probabilities=return_probabilities)

    return seg

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


def init_args():
    # Setup argparse
    parser = argparse.ArgumentParser(description="Default settings for nnUnetv2 based ")

    parser.add_argument('--path_model', type=str, default='',
                        help='directory with trained nnUnet model folds in subdirectories')
    parser.add_argument('--images',
                        default='',
                        help='path to dir with image files (old nnUnet style) or list of image paths')
    parser.add_argument('--seg_dir', type=str,
                        default='', help='directory to store output seggmentations')
    parser.add_argument('--return_probabilities', action='store_true',
                        help='flags if true exports probability npz files')

    return parser



if __name__ == "__main__":
    """
    Script runs inference using nnUnet model given input images
    
    Input arguments should be:
    path_model: path to trained nnUnet models
    images: path to dir with image files (old nnUnet style) or list of image paths
    seg_dir: dir to store the output segmentations
    return_probabilities: True/False for storing the segmentation's probabilities
    """

    parser = init_args()
    args = parser.parse_args()
    print('Inference args:', args)

    #load the nnUnet model predictor class
    predictor = init_predictor_ensemble(args.path_model)


    if os.path.isdir(args.images):
        image_files = os.listdir(args.images)
    # elif isinstance(args.images,list) and os.path.isfile(args.images[0]):
    #     image_files = args.images
    elif isinstance(args.images,str):
        image_files = args.images.split(' ')

    if not os.path.exists(args.seg_dir):
        os.makedirs(args.seg_dir)

    for f in image_files:
        file = os.path.join(args.images,f)
        if '_000' in f:
            ID = f.split('_')[0]
        else:
            ID = f.split('-')[0]

        p_vseg_out = os.path.join(args.seg_dir, '{}_vesselseg.nii.gz'.format(ID))
        p_npy_vseg = os.path.join(args.seg_dir, '{}_vesselseg'.format(ID))
        if os.path.exists(p_vseg_out):
            if args.return_probabilities and os.path.exists(p_npy_vseg + '.npy'):
                continue
            elif not args.return_probabilities:
                continue
        try:
            #read image
            img = sitk.ReadImage(file)
            #get properties of image used by nnUnet
            # props = nnunetv2_get_props(img)
            # #generate batch dimension (current inference goes one-by-one)
            # img_inp = np.expand_dims(sitk.GetArrayFromImage(img), 0).astype(np.float32)
            #predict segmentation channels= mask,probabilities
            seg = nnunetv2_predict(img, predictor, return_probabilities=args.return_probabilities)
            #write the binary prediction map
            sitk.WriteImage(np2sitk(seg[0], img), p_vseg_out)
            #write the probabilities
            np.save(p_npy_vseg, seg[1])
        except:
            print('Error for:',file)
            continue
