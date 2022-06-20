import torch
import argparse
from data_loading import *
from utils_tools import *
from train import *
from models import *

def main():
    parser = argparse.ArgumentParser(description='SfSNet - Shading Residual')
    parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to pre_train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--wt_decay', type=float, default=0.0005, metavar='W',
                        help='SGD momentum (default: 0.0005)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--read_first', type=int, default=-1,
                        help='read first n rows (default: -1)')
    parser.add_argument('--details', type=str, default=None,
                        help='Explaination of the run')
    parser.add_argument('--load_pretrained_model', type=str, default='../pretrained/net_epoch_r5_5.pth',
                        help='Pretrained model path')

    parser.add_argument('--csv_FF', type=str, default='/media/hdr/oo/Datasets/DATA_pose_15/Syn_data/',
                    help='Csv data folder of training and testing')
    parser.add_argument('--log_dir', type=str, default='/media/hdr/oo/result/Public/',
                    help='Log Path')
    parser.add_argument('--stage', type=str, default='stage_1',
                    help='Folder of training stage')
    parser.add_argument('--gpuID', type=int, default=0,
                    help='which gpu is used to train')


    parser.add_argument('--load_model', type=str, default=None,
                        help='load model from')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    seed_num = 1000
    torch.manual_seed(seed_num)

    GPU_ID = args.gpuID  
    device = torch.device("cuda:" + str(args.gpuID) if torch.cuda.is_available() else "cpu")

    print('\n\nGPU %d is working for training....\n\n'%(GPU_ID))

    expNum = 'HD_Net32_white3'

    # initialization
    celeba_data = ''
    batch_size = args.batch_size
    lr         = args.lr
    wt_decay   = args.wt_decay
    log_dir    = args.log_dir
    epochs     = args.epochs
    read_first = args.read_first

    HD_Model      = HD_Net()
    # HD_Model      = HD_Half_SharedEncoder()
    if use_cuda:
        HD_Model = HD_Model.to(device)
 
    step_flag = args.stage
    model_dir = None   # 1

    # model_dir = '/media/hdr/oo/result/Public/stage_1/HD_Net32_DeL/checkpoints/20__ASNL.pkl'

    if model_dir is not None:
        # HD_Model.load_state_dict(torch.load(model_dir, map_location={'cuda:1':'cuda:0'}))
        HD_Model.load_state_dict(torch.load(model_dir))
        HD_Model.fix_weights()
        print('************************************************')
        print('************************************************')
        print('********          loading model            *****')
        print('************************************************')
        print('************************************************\n\n\n')
    else:
        HD_Model.apply(weights_init)
        print('************************************************')
        print('************************************************')
        print('********           init model              *****')
        print('************************************************')
        print('************************************************\n\n\n')


    train(expNum, HD_Model,  args.csv_FF, step_flag, device, celeba_data=celeba_data, read_first=read_first,\
           batch_size=batch_size, num_epochs=epochs, log_path=log_dir, \
           lr=lr, wt_decay=wt_decay)

if __name__ == '__main__':
    main()
