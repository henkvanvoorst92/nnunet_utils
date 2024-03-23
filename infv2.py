

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
from .utils import np2sitk

def init_predictor(path_model):
    #from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
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


def nnunetv2_predict(img ,props ,predictor, return_probabilities=False):
    # predictor is a nnuentv2 object where inference is defined

    if isinstance(img ,list):
        if props is None and isinstance(img ,list):
            # img should then be a list of file locations
            print('do')
        else:
            seg = predictor.predict_from_list_of_npy_arrays(img,
                                                            None,
                                                            props,
                                                            None, 2, save_probabilities=False,
                                                            num_processes_segmentation_export=2)

    elif isinstance(img ,np.ndarray):
        seg = predictor.predict_single_npy_array(input_image=img, image_properties=props,
                                                 segmentation_previous_stage= None,
                                                 output_file_truncated= None,
                                                 save_or_return_probabilities= return_probabilities)
    else:
        raise ValueError('input type not list or np.ndarray:' ,type(img))

    return seg


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
    parser.add_argument('--return_probabilities', action='store_true'
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
    predictor = init_predictor(args.path_model)


    if os.path.isdir(args.images):
        image_files = os.listdir(args.images)
    elif isinstance(args.images) and os.path.isfile(args.images[0]):
        image_files = args.images

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
            props = nnunetv2_get_props(img)
            #generate batch dimension (current inference goes one-by-one)
            img_inp = np.expand_dims(sitk.GetArrayFromImage(img), 0)
            #predict segmentation channels= mask,probabilities
            seg = nnunetv2_predict(img_inp, props, predictor, return_probabilities=args.return_probabilities)
            #write the binary prediction map
            sitk.WriteImage(np2sitk(seg[0], img), p_vseg_out)
            #write the probabilities
            np.save(p_npy_vseg, seg[1])
        except:
            print('Error for:',file)
            continue
