

#from nnunetv2.paths import nnUNet_results, nnUNet_raw
#from batchgenerators.utilities.file_and_folder_operations import join
#from nnunetv2.inference import predict_from_raw_data
import os
import sys
import SimpleITK as sitk
import argparse
import numpy as np
import torch
sys.path.append('..')
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunet_utils.utils import np2sitk

def init_predictor_ensemble(path_model, device=torch.device('cuda', 0)):
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
        use_folds=(0 ,1 ,2 ,3 ,4),
        checkpoint_name='checkpoint_best.pth',
    )
    return predictor

def init_single_predictor(path_model,
                          fold,
                          checkpoint_name,
                          device=torch.device('cuda', 0),
                          #torch_type=None
                          ):
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
        checkpoint_name=checkpoint_name,
    )
    #
    # if torch_type is not None:
    #     predictor.network = predictor.network.type(torch_type)

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


def nnunetv2_predict(img: sitk.Image or list,
                     predictor,
                     return_probabilities=False,
                     use_iterator=False
                     ):
    # predictor is a nnuentv2 object where inference is defined

    if isinstance(img, sitk.Image):
        props = nnunetv2_get_props(img)
        img = np.expand_dims(sitk.GetArrayFromImage(img), 0).astype(np.float16)
    else: #for multichannel
        props = nnunetv2_get_props(img[0])
        img = np.vstack([np.expand_dims(sitk.GetArrayFromImage(im), 0).astype(np.float16) for im in img])

    if use_iterator:
        img = img.astype(np.float32)
        #this is not right you need to give a data iterators
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

def get_ID_image_dict(inp):
    """
    inp : either a list of files or a directory
          files or files in directory should have .nii.gz as extension

    returns: dictionary with ID: list of image files
    """

    if isinstance(inp,list):
        inp = inp
    elif os.path.isdir(inp):
        inp = os.listdir(inp)

    #Identify unique IDs
    channels = list(set([os.path.basename(f).replace('.nii.gz', '').split('_')[-1] for f in inp]))
    sorted_channels = sorted(channels, key=int)
    print('channels:', sorted_channels)

    for ch in sorted_channels:
        IDs = [(os.path.dirname(f),os.path.basename(f).replace(f'_{ch}.nii.gz','')) for f in inp if f'_{ch}' in os.path.basename(f)]
        if len(IDs)>0:
            break
    print(len(IDs), 'IDs available')
    #identify available channels


    dct_out = {}
    for root,ID in IDs:
        missing_chan = False
        image_files = []
        for channel in sorted_channels:
            file = os.path.join(root,f'{ID}_{channel}.nii.gz')
            if not os.path.exists(file):
                missing_chan = True
            image_files.append(file)
        if missing_chan:
            print(f'----- Channel missing skipping ID {ID} -------')
            continue
        dct_out[ID] = image_files

    return dct_out



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
    #required for processing resnet models --> something is wrong with this
    parser.add_argument('--use_iterator', action='store_true',
                        help='for ResNet the iterator is required somehow')
    #required for inference on multichannel inputs --> does not work for 4D
    parser.add_argument('--predict_from_files', action='store_true',
                        help='automates loading and analyzing multiple 3D images using files as inputs')
    parser.add_argument('--single_fold', default=None,
                        help='if single fold is a number or all, only one model is used for inference else an ensemble of all models is used')
    parser.add_argument('--addname', type=str, default='',
                        help='additional naming for output segmentation and probability files')

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
    if args.single_fold is None:
        #use the ensemble of all 5 folds for predictions
        predictor = init_predictor_ensemble(args.path_model)
    else:
        #use a single model
        predictor = init_single_predictor(args.path_model,
                                          fold=args.single_fold,
                                          checkpoint_name='checkpoint_final.pth'
                                          )

    #either a directory with images or a list of images can be used as input
    #should both be transformed to list of image paths
    if os.path.isdir(args.images):
        image_files = [os.path.join(args.images, f) for f in os.listdir(args.images)]
    elif isinstance(args.images,str):
        image_files = args.images.split(' ')
    # #you can also pass a directory with images to segment
    # if len(image_files)==1 and os.path.isdir(image_files):
    #     image_files = os.listdir(image_files)

    if not os.path.exists(args.seg_dir):
        os.makedirs(args.seg_dir)

    ID_images_dct = get_ID_image_dict(image_files)

    if args.predict_from_files:
        file_inputs = []
        seg_files = []
        for ID,files in ID_images_dct.items():
            file_inputs.append(files)
            seg_files.append(os.path.join(args.seg_dir,f'{ID}{args.addname}.nii.gz'))

        predictor.predict_from_files(file_inputs,
                                     args.seg_dir,
                                     save_probabilities=args.return_probabilities,
                                     overwrite=False,
                                     num_processes_preprocessing=2,
                                     num_processes_segmentation_export=2,
                                     folder_with_segs_from_prev_stage=None,
                                     num_parts=1, part_id=0)

    else:
        #think of a way for multi channel input --> now only single image inference
        for ID,files in ID_images_dct.items():

            p_vseg_out = os.path.join(args.seg_dir, f'{ID}{args.addname}.nii.gz')
            p_npy_vseg = os.path.join(args.seg_dir, f'{ID}{args.addname}.')
            #skip if segmentatons already exist
            if os.path.exists(p_vseg_out):
                if args.return_probabilities and os.path.exists(p_npy_vseg + '.npy'):
                    continue
                elif not args.return_probabilities:
                    continue

            img = [sitk.ReadImage(f) for f in files]
            #for standard 3D volumes
            if len(img[0].GetSize())==3:
                print('Running 3D', ID, img[0].GetSize())
                seg = nnunetv2_predict(img, predictor,
                                       return_probabilities=args.return_probabilities,
                                       use_iterator=args.use_iterator)

                print(f'Saving {ID}', p_vseg_out)
                #write the probabilities
                if args.return_probabilities:
                    # write the binary prediction map
                    if args.use_iterator:
                        probs = seg[0][1][1]
                        segmentation = seg[0][0]
                    else:
                        probs = seg[1][1]
                        segmentation = seg[0]
                    sitk.WriteImage(np2sitk(segmentation, img), p_vseg_out)
                    np.save(p_npy_vseg, probs)

                else:
                    if args.use_iterator:
                        sitk.WriteImage(np2sitk(seg[0], img), p_vseg_out)
                    else:
                        sitk.WriteImage(np2sitk(seg, img), p_vseg_out)

            #for CTP series
            elif len(img[0].GetSize())==4:
                print('Running 4D', ID, img[0].GetSize())
                seg_out = []
                prob_out = []
                for i in range(img.GetSize()[-1]):
                    seg = nnunetv2_predict(img[:,:,:,i], predictor,
                                           return_probabilities=args.return_probabilities,
                                           use_iterator=args.use_iterator)

                    if args.return_probabilities:
                        if args.use_iterator:
                            probs = seg[0][1][1]
                            segmentation = seg[0][0]
                        else:
                            probs = seg[1][1]
                            segmentation = seg[0]
                        seg_out.append(np2sitk(segmentation, img[:, :, :, i]))
                        prob_out.append(probs)
                    else:
                        seg_out.append(np2sitk(seg, img[:, :, :, i]))

                seg_out = sitk.JoinSeries(seg_out)
                print(f'Saving {ID}', p_vseg_out)
                sitk.WriteImage(seg_out, p_vseg_out)

                if args.return_probabilities:
                    prob_out = np.stack(prob_out)
                    np.save(p_npy_vseg,prob_out)

        #except:
            # print('Error for:',file)
            # continue
