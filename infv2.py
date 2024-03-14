

#from nnunetv2.paths import nnUNet_results, nnUNet_raw
#from batchgenerators.utilities.file_and_folder_operations import join
#from nnunetv2.inference import predict_from_raw_data
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def init_predictor(path_model):

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

#
# predictor = nnUNetPredictor(
#     tile_step_size=0.5,
#     use_gaussian=True,
#     use_mirroring=True,
#     # perform_everything_on_device=True,
#     device=torch.device('cuda', 0),
#     verbose=False,
#     verbose_preprocessing=False,
#     allow_tqdm=True
# )
#
# predictor.initialize_from_trained_model_folder(
#     join(nnUNet_results, 'Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_lowres'),
#     use_folds=(0,),
#     checkpoint_name='checkpoint_final.pth',
# )
#
# # predict a single numpy array
# img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTr/liver_63_0000.nii.gz')])
# ret = predictor.predict_single_npy_array(img, props, None, None, False)

