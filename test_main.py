import torch
import argparse
from data_loading import *
from utils_tools import *
from test import *
from models import *

def main():
    parser = argparse.ArgumentParser(description='HD-Net')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to pre_train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--wt_decay', type=float, default=0.0005, metavar='W',
                        help='SGD momentum (default: 0.0005)')
    parser.add_argument('--seed', type=int, default=1000, metavar='S',
                        help='random seed (default: 1000)')
    parser.add_argument('--dataset_path', type=str, default='./data/sample/',
                    help='csv data folder of training and testing')
    parser.add_argument('--log_dir', type=str, default='../results/HD-Net/',
                    help='path of saving training and testing results and logs')
    parser.add_argument('--stage', type=str, default='stage_1',
                    help='FolderName of training stage')
    parser.add_argument('--gpuID', type=int, default=0,
                    help='which gpu is used to train')
    parser.add_argument('--load_model', type=str, default='./checkpoint/ASNL.pkl',
                        help='load model from')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    GPU_ID = args.gpuID  
    device = torch.device("cuda:" + str(args.gpuID) if torch.cuda.is_available() else "cpu")
    print('\n\nGPU %d is working for training....\n\n'%(GPU_ID))

    # initialization
    batchsize = args.batch_size
    lr         = args.lr
    wt_decay   = args.wt_decay
    log_dir    = args.log_dir
    epochs     = args.epochs
    data_path  = args.dataset_path

    HD_Model      = HD_Net()
    HD_Model = HD_Model.to(device)
 
    stage_flag = args.stage
    model_dir = args.load_model    
    # model_dir = '/media/hdr/oo/result/Public/stage_1/HD_Net32_DeL/checkpoints/20__ASNL.pkl'

    if model_dir is not None:
        # HD_Model.load_state_dict(torch.load(model_dir, map_location={'cuda:1':'cuda:0'}))
        HD_Model.load_state_dict(torch.load(model_dir))
        # HD_Model.fix_weights()
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


    TestImg(HD_Model, data_path, stage_flag, device, batch_size=batchsize, num_epochs=epochs, log_path=log_dir, lr=lr, wt_decay=wt_decay)

if __name__ == '__main__':
    main()
