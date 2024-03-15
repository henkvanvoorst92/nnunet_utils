
import os

def nnunet_inference_on_dir(model_path,
                            dir_input_images,
                            dir_output_seg,
                            resolution,
                            save_probs=True
                            ):
    """
    Runs an nnUNetV2 inference command using os.system on a directory of test images

    :param model_path: The path to the trained model directory.
    :param input_images: The path to the directory containing input images.
    :param output_folder: The path to the directory where output will be saved.

    see also:
    https://github.com/DIAGNijmegen/nnUNet_v2/blob/master/documentation/how_to_use_nnunet.md
    """
    # Construct the nnUNetV2 inference command based on your specific requirements
    command = f"nnUNetv2_predict -i {dir_input_images}"
    command += f"-o {dir_output_seg}"
    command += f"-t {model_path}"
    command += f" -c {resolution}"

    if save_probs:
        command += f" --save_probabilities" 

    #print to check if the command is right
    print('Start running command')
    print(command)
    #use os.system to run the command
    status = os.system(command)

    if status != 0:
        print("Error executing the command")
    else:
        print("Command executed successfully")
