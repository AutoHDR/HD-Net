import argparse
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from skimage import color, io
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

# from ColorEncoder import ColorEncoder
from models import ColorEncoder, ColorUNet1, Discriminator,GenCycle
from vgg_model import vgg19
from data.data_loader import get_Pre_trainData, get_ImgPair_pretrain, get_PhotoDB
from torchvision.utils import save_image
from tools import get_Normal_Std_Mean, get_normal_N, get_normal_255, get_normal_P
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def N_SFS2CM(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = normal[:, 1, :, :]
    tt[:, 1, :, :] = - normal[:, 0, :, :]
    tt[:, 2, :, :] = normal[:, 2, :, :]
    return tt

def mkdirss(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def Lab2RGB_out(img_lab):
    img_lab = img_lab.detach().cpu()
    img_l = img_lab[:,:1,:,:]
    img_ab = img_lab[:,1:,:,:]
    # print(torch.max(img_l), torch.min(img_l))
    # print(torch.max(img_ab), torch.min(img_ab))
    img_l = img_l + 50
    pred_lab = torch.cat((img_l, img_ab), 1)[0,...].numpy()
    # grid_lab = utils.make_grid(pred_lab, nrow=1).numpy().astype("float64")
    # print(grid_lab.shape)
    out = (np.clip(color.lab2rgb(pred_lab.transpose(1, 2, 0)), 0, 1)* 255).astype("uint8")
    return out

def RGB2Lab(inputs):
    # input [0, 255] uint8
    # out l: [0, 100], ab: [-110, 110], float32
    return color.rgb2lab(inputs)

def Normalize(inputs):
    l = inputs[:, :, 0:1]
    ab = inputs[:, :, 1:3]
    l = l - 50
    lab = np.concatenate((l, ab), 2)

    return lab.astype('float32')

def numpy2tensor(inputs):
    out = torch.from_numpy(inputs.transpose(2,0,1))
    return out

def tensor2numpy(inputs):
    out = inputs[0,...].detach().cpu().numpy().transpose(1,2,0)
    return out

def preprocessing(inputs):
    # input: rgb, [0, 255], uint8
    img_lab = Normalize(RGB2Lab(inputs))
    img = np.array(inputs, 'float32') # [0, 255]
    img = numpy2tensor(img)
    img_lab = numpy2tensor(img_lab)
    return img.unsqueeze(0), img_lab.unsqueeze(0)

def uncenter_l(inputs):
    l = inputs[:,:1,:,:] + 50
    ab = inputs[:,1:,:,:]
    return torch.cat((l, ab), 1)

def train(
    args,
    train_dl,
    ph_val_dl,
    ph_train_dl,
    GenCycle,
    g_optim,
    device,
):
    train_loader = sample_data(train_dl)
    valtrain_loader = sample_data(valtrain_dl)
    pbar = range(args.iter + 1)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    g_loss_val = 0
    loss_dict = {}
    W_H = 256
    input_shape = (3, W_H, W_H)
    DN = Discriminator(input_shape).to(device)
    optimizer_D = torch.optim.Adam(DN.parameters(), lr=args.lr, weight_decay=0.005)
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    # model_dir = '/home/xteam/wang/result/SinDecom/Img1024/EnDen_V1_01_29/checkpoints/30__DN.pkl'
    # DN.load_state_dict(torch.load(model_dir))
    GenCycle_module = GenCycle

    lossflag = args.lossflag
    savepath = '/media/hdr/oo/result/E_normal/PT_for_P_'
    logPath = savepath + str(args.refD) + '/' + 'log' + '/'
    mkdirss(logPath)
    writer = SummaryWriter(logPath)
    imgsPath = savepath + str(args.refD) + '/' + 'imgs' + '/'
    mkdirss(imgsPath)
    expPath = savepath + str(args.refD) + '/' + 'exp' + '/'
    mkdirss(expPath)
    CosLoss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    L1Loss = nn.L1Loss()
    iterflag = 0
    for epoch in range(10000):
        for bix, batch in enumerate(ph_train_dl):
            ph_face, gt_n, pre_norm, randomNN, mask, index = batch
            b, c, w, h = ph_face.shape
            ph_face = ph_face.to(device)
            ph_face_grey = (ph_face[:,0,:,:]+ph_face[:,1,:,:]+ph_face[:,2,:,:])/3
            ph_face_grey = ph_face_grey.unsqueeze(1)
            pre_norm =pre_norm.to(device)
            mask = mask.to(device)
            gt_n = gt_n.to(device)

            pNormal = GenCycle(ph_face_grey)
            pNormal = F.normalize(pNormal)

            
            valid = Variable(Tensor(np.ones((pNormal.size(0), *DN.output_shape))), requires_grad=False).to(device)
            fake = Variable(Tensor(np.zeros((pNormal.size(0), *DN.output_shape))), requires_grad=False).to(device)
            optimizer_D.zero_grad()
            loss_real_1 = criterion_GAN(DN(gt_n.detach()), valid)
            loss_fake_3 = criterion_GAN(DN(pNormal.detach()), fake)
            D_loss = loss_real_1 + loss_fake_3
            D_loss.backward()
            optimizer_D.step()

            loss_real = args.refD * (criterion_GAN(DN(pNormal), valid))

            recon_N = (1 - CosLoss(pNormal, gt_n).mean())
            LossTotal = recon_N + loss_real
            print('epoch:{}-bix/len:{}/{} -recon_N: {:.5f}, loss_real:{:.5}'.format(epoch, bix, len(ph_train_dl), recon_N.item(), loss_real.item()))
            g_optim.zero_grad()
            LossTotal.backward()
            g_optim.step()
            save_image(get_normal_255(pNormal), imgsPath + str(epoch) + '.png', nrow=b, normalize=True)

        torch.save(
            {
                "GenCycle": GenCycle_module.state_dict(),
                "g_optim": g_optim.state_dict(),
                "args": args,
            },
            f"%s/{str(iterflag).zfill(6)}.pt"%(expPath),
        )
        torch.save(DN.state_dict(), f"%s/{str(iterflag).zfill(6)}_DF.pt"%(expPath))

        with torch.no_grad():
            testflag = 0
            sum_std = 0
            sum_mean = 0
            sum_n20 = 0
            sum_n25 = 0
            sum_n30 = 0
            GenCycle.eval()
            for bix, batch in enumerate(ph_val_dl):
                ph_face, gt_n, pre_norm, randomNN, mask, index = batch
                b, c, w, h = ph_face.shape
                ph_face = ph_face.to(device)
                ph_face_grey = (ph_face[:,0,:,:]+ph_face[:,1,:,:]+ph_face[:,2,:,:])/3
                ph_face_grey = ph_face_grey.unsqueeze(1)
                pre_norm =pre_norm.to(device)
                mask = mask.to(device)
                gt_n = gt_n.to(device)
                pre_norm = F.normalize(pre_norm)

                ph_fine_normal = GenCycle((ph_face_grey)) #[-1, 1]
                ph_fine_normal = F.normalize(ph_fine_normal)

                std, mean, n20, n25, n30 =  get_Normal_Std_Mean(ph_fine_normal, gt_n, mask)
                sum_std = sum_std + std.item()
                sum_mean = sum_mean + mean.item()
                sum_n20 += n20.item()
                sum_n25 += n25.item()
                sum_n30 += n30.item()
                testflag += 1  

                lossflag += 1
                LOn = F.smooth_l1_loss(ph_fine_normal, gt_n)
                writer.add_scalar('LOn', LOn.item(), lossflag)
                sampleImgs = torch.cat([ph_face, get_normal_255(gt_n), get_normal_255(ph_fine_normal)], 0)
                save_image(sampleImgs, imgsPath + 'ph_%d'%(iterflag) + '_.png', nrow=b, normalize=True)
                torch.cuda.empty_cache()
                print(bix)
            print("===> Avg. mean: {:.4f}, std:{:.4f}, n20:{:.4f}, n25:{:.4f}, n30:{:.4f}".format(sum_mean / testflag, sum_std / testflag, sum_n20 / testflag, sum_n25 / testflag, sum_n30 / testflag))
            with open(logPath +'mean_std.txt', 'a+') as f:
                details = str(iterflag) + '\t' + str(sum_mean / testflag) + '\t' + str(sum_std /testflag) + '\t' + str(sum_n20 /testflag) + '\t' + str(sum_n25 /testflag) + '\t' + str(sum_n30 /testflag) + '\n'
                f.write(details)   
        iterflag += 1
        epoch += 1

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--lossflag", type=int, default=0)
    parser.add_argument("--refR", type=int, default=20)
    parser.add_argument("--refG", type=int, default=1)
    parser.add_argument("--refD", type=float, default=0.01)
    parser.add_argument("--refVF", type=int, default=1)
    parser.add_argument("--gpuID", type=int, default=1)

    args = parser.parse_args()
    device = "cuda:" + str(args.gpuID)

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.start_iter = 0
    GenCycle = GenCycle(1, 3)
    GenCycle = GenCycle.to(device)
    
    g_optim = optim.Adam(
        list(GenCycle.parameters()),
        lr=args.lr,
        betas=(0.9, 0.99),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass
        
        GenCycle.load_state_dict(ckpt["GenCycle"])
        g_optim.load_state_dict(ckpt["g_optim"])

    datasets = []
    pathd = '/home/hdr/autohdr/2022/CrossN_V1/data/csv/ALL_Face_train.csv'
    train_dataset, _ = get_Pre_trainData(csvPath=pathd, validation_split=0)
    train_dl  = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    pathd = '/home/hdr/autohdr/2022/CrossN_V1/data/csv/ALL_Face_test.csv'
    test_dataset, _ = get_Pre_trainData(csvPath=pathd, validation_split=0)
    valtrain_dl  = DataLoader(test_dataset, batch_size=args.batch, shuffle=True)

    pathd = '/home/hdr/autohdr/2022/CrossN_V1/data/csv/Phdb_test.csv'
    val_dataset, _ = get_PhotoDB(csvPath=pathd, validation_split=0)
    ph_val_dl  = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    pathd = '/home/hdr/autohdr/2022/CrossN_V1/data/csv/Phdb_train.csv'
    val_dataset, _ = get_PhotoDB(csvPath=pathd, validation_split=0)
    ph_train_dl  = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    train(
        args,
        train_dl,
        ph_val_dl,
        ph_train_dl,
        GenCycle,
        g_optim,
        device,
    )

