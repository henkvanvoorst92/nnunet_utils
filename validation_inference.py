import os
import sys
import SimpleITK as sitk
import argparse
import numpy as np
import torch
from tqdm import tqdm
import ast
import pandas as pd
import multiprocessing as mp
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
os.system('cd ..')
from nnunet_utils.utils import np2sitk, load_json, save_json, write_list2txt, read_list_from_txt, is_notebook, np_dice
from nnunet_utils.infv2 import nnunetv2_predict, init_single_predictor, nnunetv2_get_props


def inference_on_validation_splits(root_imgs: str,
                                   root_labels: str,
                                   root_model: str,
                                   splits: dict,
                                   name: str = 'tmp',
                                   sav_dir: str = None,
                                   add_info: dict = {},
                                   checkpoint: str = 'checkpoint_best.pth',
                                   dev_number: int = 0,
                                   **kwargs
                                   ):
    # kwargs enables multiprocessing
    if 'root_imgs' in kwargs:
        root_imgs = kwargs['root_imgs']
    if 'root_labels' in kwargs:
        root_labels = kwargs['root_labels']
    if 'root_model' in kwargs:
        root_model = kwargs['root_model']
    if 'splits' in kwargs:
        splits = kwargs['splits']
    if 'name' in kwargs:
        name = kwargs['name']
    if 'sav_dir' in kwargs:
        sav_dir = kwargs['sav_dir']
    if 'add_info' in kwargs:
        add_info = kwargs['add_info']
    if 'checkpoint' in kwargs:
        checkpoint = kwargs['checkpoint']
    if 'dev_number' in kwargs:
        dev_number = kwargs['dev_number']

    dev = torch.device('cuda', dev_number)

    if sav_dir is not None:
        os.makedirs(sav_dir, exist_ok=True)
        p_df = os.path.join(sav_dir, f'{name}.xlsx')
        dir_pred = os.path.join(sav_dir, 'pred')
        os.makedirs(dir_pred, exist_ok=True)
    else:
        dir_pred = None

    out = []
    for fold, split in splits.items():
        try:
            valIDs = split['val']  # validation IDs of set fold
            # load the split model
            model = init_single_predictor(root_model, fold, checkpoint, device=dev)
        except:
            print('Model load error', root_model, fold, split)
            continue

        for ID in tqdm(valIDs):
            try:
                p_img = os.path.join(root_imgs, f'{ID}_0000.nii.gz')
                p_lbl = os.path.join(root_labels, f'{ID}.nii.gz')
                img = sitk.ReadImage(p_img)
                lbl = sitk.GetArrayFromImage(sitk.ReadImage(p_lbl))
                seg = nnunetv2_predict(img,
                                       model,
                                       return_probabilities=False)
                dsc = np_dice(lbl, seg)

                row = list(add_info.values())
                row.extend([name, fold, ID, dsc, p_img, p_lbl,
                            root_model, 'checkpoint_best.pth'])
                out.append(row)
                if dir_pred is not None:
                    p_seg = os.path.join(dir_pred, f'{ID}-{fold}-{name}.nii.gz')
                    sitk.WriteImage(np2sitk(seg, img), p_seg)
            except:
                print('ID processing error:', ID)
                continue

    cols = [*list(add_info.keys()), 'Name', 'fold', 'ID', 'DSC',
            'p_img', 'p_lbl', 'root_model', 'checkpoint']
    df = pd.DataFrame(out, columns=cols)
    if sav_dir is not None:
        df.to_excel(p_df)

    return df

def init_args(args=None):
    # Setup argparse
    parser = argparse.ArgumentParser(description="Default settings for nnUnetv2 based ")

    parser.add_argument('--root', type=str, default='',
                        help='nnunet directory root')
    parser.add_argument('--exp', type=str, default='',
                        help='txt file with experiments defined including gpu allocation')
    parser.add_argument('--p', action='store_true',
                        help='If true runs processes in parallel using multiprocessing')
    if is_notebook():
        print("Detected notebook environment, using default argument values.")
        return parser.parse_args([])
    else:
        return parser.parse_args(args)

if __name__ == "__main__":
    """
    Script runs inference using nnUnet model given input images from validation set
    ascertains only validation set per fold is used
    """

    args = init_args()
    print('Inference args:', args)
    p_exp = os.path.join(args.root,args.exp)
    exps = read_list_from_txt(p_exp)

    root_nnunet =args.root
    sav_dir = os.path.join(root_nnunet, 'validation_inference')

    tasks_kwargs = []
    for datano, datasetID, mod, plan, name, dev_no in exps:
        p_split = os.path.join(root_nnunet, 'nnUNet_preprocessed', datasetID, 'splits_final.json')
        splits = load_json(p_split)

        root_model = os.path.join(root_nnunet, 'nnUNet_trained_models', datasetID, plan)
        root_imgs = os.path.join(root_nnunet, 'nnUNet_raw', datasetID, 'imagesTr')
        root_lbls = os.path.join(root_nnunet, 'nnUNet_raw', datasetID, 'labelsTr')

        add_info = {'datano': datano,
                    'datasetID': datasetID,
                    'model': mod,
                    'plan': plan}
        #define input for the (worker) function inference_on_validation_splits
        inp = {'root_model': root_model,
               'root_imgs': root_imgs,
               'root_labels': root_lbls,
               'splits': {i: split for i, split in enumerate(splits)},
               'name': name,
               'sav_dir': sav_dir,
               'add_info': add_info,
               'checkpoint': 'checkpoint_best.pth',
               'dev_number': dev_no
               }
        if not args.p:
            #run the file here
            res = inference_on_validation_splits(**inp)

        else:
            tasks_kwargs.append(inp)

    if args.p:
        processes = []
        for i,kwargs in enumerate(tasks_kwargs):
            print('---- start process:',i)
            p = mp.Process(target=inference_on_validation_splits, kwargs=kwargs)
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()