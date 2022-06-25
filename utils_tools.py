import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torch.nn import *

def applyMask(input_img, mask):
    if mask is None:
        return input_img
    return input_img * mask

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def get_normal_in_range(normal):
    new_normal = normal * 128 + 128
    new_normal = new_normal.clamp(0, 255) / 255
    return new_normal

def get_image_grid(pic, denormalize=False, mask=None):
    if denormalize:
        pic = denorm(pic)
    
    if mask is not None:
        pic = pic * mask

    grid = torchvision.utils.make_grid(pic, nrow=8, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return ndarr

def save_image(pic, denormalize=False, path=None, mask=None):
    ndarr = get_image_grid(pic, denormalize=denormalize, mask=mask)    
    
    if path == None:
        plt.imshow(ndarr)
        plt.show()
    else:
        im = Image.fromarray(ndarr)
        im.save(path)

def wandb_log_images(img, mask, pathName=None, denormalize=False):
    ndarr = get_image_grid(img, denormalize=denormalize, mask=mask)

    # save image if path is provided
    if pathName is not None:
        im = Image.fromarray(ndarr)
        im.save(pathName)


def weights_init(m):
    if isinstance(m, Conv2d) or isinstance(m, Conv1d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            init.constant_(m.bias, 0)
    elif isinstance(m, Linear):
        init.normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

import os, torch
def wandb_log_images_Single(img, mask, index, path, flag):
    b,c,w,h = img.shape
    if mask is not None:
        unmask = torch.where(mask<0.3, torch.ones_like(mask),torch.zeros_like(mask))
        img = img + unmask
        for i in range(b):
            tpimg = img[i,:,:,:]
            tppath = path + flag + '/'
            if not os.path.exists(tppath):
                os.makedirs(tppath)
            tppath = tppath + index[i].split('/')[-1]
            im = Image.fromarray(tpimg.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
            im.save(tppath)
    else:
        img = img
        for i in range(b):
            tpimg = img[i, :, :, :].expand([3,256,256])
            tppath = path + flag + '/'
            if not os.path.exists(tppath):
                os.makedirs(tppath)
            tppath = tppath + index[i].split('/')[-1]
            im = Image.fromarray(tpimg.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
            im.save(tppath)

import os, tarfile

def make_targz(output_filename, source_dir):
    try:
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

        return True
    except Exception as e:
        print(e)
        return False

def ImageBatchNormalization(input):
    [b,c,w,h] = input.size()
    tp_max = input.max(dim=1).values.max(dim=1).values.max(dim=1).values.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand([b,c,w,h])
    tp_min = input.min(dim=1).values.min(dim=1).values.min(dim=1).values.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand([b,c,w,h])
    tp_data =  (input - tp_min) / (tp_max - tp_min + 0.00001)
    return tp_data

