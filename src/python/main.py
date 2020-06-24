'''
Training and testing script for CompenNet.

This script trains/tests CompenNet on different dataset specified in 'data_list' below.
The detailed training options are given in 'train_option' below.

1. We start by setting the training environment to GPU (if any), then put K=24 setups in 'data_lsit', the 24 setups are used in our paper.
2. We set number of training images to 500 and loss functions to l1+ssim, you can add other num_train and loss to 'num_train_list' and 'loss_list' for
comparsion. The training options are specified in 'train_option'.
3. The model is set to 'CompenNet', you can define your own models in CompenNetModel.py and add their names to 'model_list' for comparison.
4. The training data 'train_data' and validation data 'valid_data', are loaded in RAM by 'readImgsMT', and we train the model with 'trainModel'. The
training and validation results are shown in Visdom window as pictures as well as printed to the console.
5. Once the training is finished, we can evaluate the model by comparing predicted projector input image 'cam'cmp' with the ground truth
projector input image 'prj_valid'. The compensation images 'prj_cmp' are useful when you apply CompenNet to your own setup, you can project 'prj_cmp'
to the surface and see the compensation results as shown in our paper and supplementary material.

Example:
    python main.py

See CompenNetModel.py for CompenNet structure.
See CompenNetDataset.py for training and validation data loading.
See trainNetwork.py for detailed training process.
See utils.py for helper functions.
'''

from trainNetwork import *
import CompenNetModel

# %% Set environment
# set which GPUs to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2, 3'
device_ids = [0, 1, 2]

# set PyTorch device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Train with', torch.cuda.device_count(), 'GPUs!') if torch.cuda.device_count() >= 1 else print('Train with CPUs!')

# repeatability
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% K=24 setups
dataset_root = fullfile(os.getcwd(), '../../data')

data_list = [
    'light2/pos1/curves',
    'light2/pos1/squares',
    'light1/pos1/stripes',
    'light2/pos2/lavender',
    'light2/pos2/squares',
    'light4/pos2/curves',
    'light4/pos3/stripes',
    'light2/pos4/curves',
    'light2/pos5/cloud',
    'light1/pos3/cubes',
    'light1/pos4/curves',
    'light2/pos5/lavender',
    'light2/pos5/squares',
    'light2/pos5/stripes',
    'light1/pos5/cubes',
    'light3/pos2/cubes',
    'light3/pos1/curves',
    'light3/pos1/lavender',
    'light3/pos4/cubes',
    'light3/pos3/curves',
    'light3/pos3/lavender',
    'light3/pos3/squares',
    'light1/pos2/stripes',
    'light4/pos1/stripes'
]

# Training configurations of CompenNet reported in the paper
num_train_list = [500]
loss_list = ['l1+ssim']

# You can also compare different configurations, such as different number of training images and loss functions as shown below
# num_train_list = [125, 250, 500]
# loss_list = ['l1', 'l2', 'ssim', 'l1+l2', 'l1+ssim', 'l2+ssim', 'l1+l2+ssim']

model_list = ['CompenNet']  # you can create your own models in CompenNetModel.py and put their names in this list for comparisons.
# model_list = ['CompenNet_pretrain']  # if you have a pretrained model to compare, add 'CompenNet_pretrain' to model_list, and modify the pth path.

train_option = {'data_name': '',  # will be set later
                'model_name': '',
                'num_train': '',
                'max_iters': 1000,
                'batch_size': 64,
                'lr': 1e-3,  # learning rate
                'lr_drop_ratio': 0.2,
                'lr_drop_rate': 800,
                'loss': '',  # loss will be set to one of the loss functions in loss_list later
                'l2_reg': 1e-4,  # l2 regularization
                'device': device,
                'plot_on': True,  # plot training progress using visdom
                'train_plot_rate': 100,  # training and visdom plot rate
                'valid_rate': 100}  # validation and visdom plot rate

# a flag that decides whether to compute and save the compensated images to the drive
save_compensation = True

# log file
from time import localtime, strftime
log_dir = '../../log'
if not os.path.exists(log_dir): os.makedirs(log_dir)
log_file_name = strftime('%Y-%m-%d_%H_%M_%S', localtime()) + '.txt'
log_file = open(fullfile(log_dir, log_file_name), 'w')
title_str = '{:30s}{:<20}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
log_file.write(title_str.format('data_name', 'model_name', 'loss_function',
                                'num_train', 'batch_size', 'max_iters',
                                'uncmp_psnr', 'uncmp_rmse', 'uncmp_ssim',
                                'valid_psnr', 'valid_rmse', 'valid_ssim'))
log_file.close()

## evaluate all K=24 setups
for data_name in data_list:
    # data paths
    data_root = fullfile(dataset_root, data_name)
    cam_ref_path = fullfile(data_root, 'cam/warp/ref')
    cam_train_path = fullfile(data_root, 'cam/warp/train')
    prj_train_path = fullfile(dataset_root, 'train')
    cam_valid_path = fullfile(data_root, 'cam/warp/test')
    prj_valid_path = fullfile(dataset_root, 'test')
    print("Loading data from '{}'".format(data_root))

    # training data
    cam_surf = readImgsMT(cam_ref_path, index=[125])
    cam_train = readImgsMT(cam_train_path)
    prj_train = readImgsMT(prj_train_path)

    # validation data
    cam_valid = readImgsMT(cam_valid_path)
    prj_valid = readImgsMT(prj_valid_path)

    # convert valid data to CUDA tensor
    cam_valid = cam_valid.to(device)
    prj_valid = prj_valid.to(device)

    # surface image for training and validation
    cam_surf_train = cam_surf.expand_as(cam_train)
    cam_surf_valid = cam_surf.expand_as(cam_valid)

    valid_data = dict(cam_surf=cam_surf_valid, cam_valid=cam_valid, prj_valid=prj_valid)

    for num_train in num_train_list:
        train_option['num_train'] = num_train

        # select a subset to train
        train_data = dict(cam_surf=cam_surf_train[:num_train, :, :, :], cam_train=cam_train[:num_train, :, :, :],
                          prj_train=prj_train[:num_train, :, :, :])

        for model_name in model_list:
            train_option['model_name'] = model_name

            for loss in loss_list:
                log_file = open(fullfile(log_dir, log_file_name), 'a')

                # set seed of rng for repeatability
                resetRNGseed(0)

                # create a CompenNet model
                if model_name == 'CompenNet_pretrain':
                    # start from a pretrained model
                    compen_net = CompenNetModel.CompenNet()
                    if torch.cuda.device_count() > 1:
                        compen_net = nn.DataParallel(compen_net, device_ids=device_ids)
                    compen_net.to(device)

                    # this path should points to a pre-trained model,
                    compen_net.load_state_dict(
                        torch.load('../../checkpoint/light1_pos1_stripes_CompenNet_l1+ssim_500_64_10_0.001_0.2_800_0.0001.pth'))
                    compen_net.device_ids = device_ids

                    if torch.cuda.device_count() > 1:
                        compen_net.module.name = model_name
                    else:
                        compen_net.name = model_name
                else:
                    # start from scratch
                    compen_net = getattr(CompenNetModel, model_name)()
                    if torch.cuda.device_count() > 1:
                        compen_net = nn.DataParallel(compen_net, device_ids=device_ids)

                    compen_net.to(device)

                # % train option for current configuration, i.e., data name and loss function
                train_option['data_name'] = data_name.replace('/', '_')
                train_option['loss'] = loss

                print('-------------------------------------- Training Options -----------------------------------')
                print("\n".join("{}: {}".format(k, v) for k, v in train_option.items()))
                print('-------------------------------------- Start training CompenNet ---------------------------')
                compen_net, valid_psnr, valid_rmse, valid_ssim = trainModel(compen_net, train_data, valid_data, train_option)

                # save results to log file
                ret_str = '{:30s}{:<20}{:<20}{:<15}{:<15}{:<15}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}\n'
                log_file.write(ret_str.format(data_name, model_name, loss, num_train, train_option['batch_size'], train_option['max_iters'],
                                              psnr(cam_valid, prj_valid), rmse(cam_valid, prj_valid), ssim(cam_valid, prj_valid),
                                              valid_psnr, valid_rmse, valid_ssim))
                log_file.close()

                # create compensated testing images
                if save_compensation:
                    print('Saving compensation images\n')
                    torch.cuda.empty_cache()
                    cam_surf_valid = cam_surf_valid.to(device)

                    with torch.no_grad():
                        cam_cmp = compen_net(cam_valid, cam_surf_valid).detach()  # compensated cam captured image \hat{x}^{*}
                        prj_cmp = compen_net(prj_valid, cam_surf_valid).detach()  # compensated prj input image x^{*}

                    # create image save path
                    cmp_folder_name = '{}_{}_{}_{}_{}'.format(model_name, loss, num_train, train_option['batch_size'], train_option['max_iters'])
                    cam_cmp_path = fullfile(data_root, 'cam/cmp', cmp_folder_name)
                    prj_cmp_path = fullfile(data_root, 'prj/cmp', cmp_folder_name)
                    if not os.path.exists(cam_cmp_path): os.makedirs(cam_cmp_path)
                    if not os.path.exists(prj_cmp_path): os.makedirs(prj_cmp_path)

                    # save images
                    saveImgs(cam_cmp, cam_cmp_path)  # compensated validation images
                    saveImgs(prj_cmp, prj_cmp_path)  # compensated testing images, i.e., to be projected to the surface

                # clear cache
                del compen_net
                torch.cuda.empty_cache()
                print('-------------------------------------- Done! ---------------------------\n')
        del train_data
    del cam_valid, prj_valid

print('All dataset done!')
