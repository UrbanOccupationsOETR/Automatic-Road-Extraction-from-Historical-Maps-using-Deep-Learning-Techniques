
import numpy as np
import os

def get_params():

    params = {}
    
    params['main_dir'] = r"C:\Users\burak\Desktop"

    params['experiment_log'] = r"upp_r50" # experiment ID. 
    params['data_dir'] = os.path.join(params['main_dir'], 'map_data', 'Dataset') # dataset path. 
    params['log_path'] = os.path.join(params['main_dir'], 'map_exp', '5cls_2') # path to save logs.
    params['inference_ims_input'] = os.path.join(params['main_dir'], 'map_data', 'all_dataset_ims') # images for geo-injection.
    params['inference_save_dir'] = os.path.join(params['main_dir'], 'map_data', 'save_geo') # path to save the geo-inject outputs 

    params['ENCODER'] = 'resnext50_32x4d' # backbone architecture, encoder. 
    params['ENCODER_WEIGHTS'] = 'imagenet' # pre-trained weights. 
    params['CLASSES'] = np.arange(0,6,1) # used for encoding-decoding the mask. 
    params['ACTIVATION'] = 'softmax2d' # activation function.
    params['DEVICE'] = 'cuda' # GPU or CPU.
    params['batch_size'] = 18 # batch size. 
    params['mul_factor'] = 5 # sampling parameter.
    params['lr'] = 0.0001 # learning rate.
    params['n_epoch'] = 50 # number of epochs.
    params['ig_ch'] = [0] # channel-s to ignore during evaluation.
    params['n_workers'] = 0 #number of workers for data loader, multi-process scheme.

    return params