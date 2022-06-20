import re, glob, os, random
import pandas as pd
from shutil import copyfile
import numpy as np
from fnmatch import fnmatch, fnmatchcase
import torch
from PIL import Image
from torchvision import transforms
from torchvision import utils
def get_shading_DPR_0802(N, L):
    b, c, h, w = N.shape
    c1 = 0.8862269254527579
    c2 = 1.0233267079464883
    c3 = 0.24770795610037571
    c4 = 0.8580855308097834
    c5 = 0.4290427654048917
    for i in range(b):
        N1 = N[i,:,:,:]
        N2 = torch.zeros_like(N1)
        N2[2,:,:] = N1[0,:,:]
        N2[1,:,:] = N1[1,:,:]
        N2[0,:,:] = N1[2,:,:]

        N3 = torch.zeros_like(N2)
        N3[0,:,:] = N2[0,:,:]
        N3[1,:,:] = N2[2,:,:]
        N3[2,:,:] = -1 * N2[1,:,:]

        N3=N3.permute([1,2,0]).reshape([-1,3])

        norm_X = N3[:,0]
        norm_Y = N3[:,1]
        norm_Z = N3[:,2]
        numElem = N3.shape[0]
        sh_basis = torch.from_numpy(np.zeros([numElem, 9])).type(torch.FloatTensor)
        att= torch.from_numpy(np.pi*np.array([1, 2.0/3.0, 1/4.0])).type(torch.FloatTensor)
        sh_basis[:,0] = torch.from_numpy(np.array(0.5/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)*att[0]

        sh_basis[:,1] = torch.from_numpy(np.array(np.sqrt(3)/2/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)*norm_Y*att[1]
        sh_basis[:,2] = torch.from_numpy(np.array(np.sqrt(3)/2/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)*norm_Z*att[1]
        sh_basis[:,3] = torch.from_numpy(np.array(np.sqrt(3)/2/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)*norm_X*att[1]

        sh_basis[:,4] = torch.from_numpy(np.array(np.sqrt(15)/2/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)*norm_Y*norm_X*att[2]
        sh_basis[:,5] = torch.from_numpy(np.array(np.sqrt(15)/2/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)*norm_Y*norm_Z*att[2]
        sh_basis[:,6] = torch.from_numpy(np.array(np.sqrt(5)/4/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)*(3*norm_Z**2-1)*att[2]
        sh_basis[:,7] = torch.from_numpy(np.array(np.sqrt(15)/2/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)*norm_X*norm_Z*att[2]
        sh_basis[:,8] = torch.from_numpy(np.array(np.sqrt(15)/4/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)*(norm_X**2-norm_Y**2)*att[2]

        light = L[i,:]
        shading = torch.matmul(sh_basis, light)
        myshading = (shading - torch.min(shading) )/ (torch.max(shading)-torch.min(shading))

        tp = myshading.reshape([-1,h,w])
        if i == 0 :
            result = tp
        else:
            result = torch.cat([result,tp],axis=0)

    b, w, h = result.shape
    return result.reshape([b, 1, w, h])
IMAGE_SIZE = 256
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])
import subprocess

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def get_image_grid(pic, path, denormalize=False, mask=None):
    if denormalize:
        pic = denorm(pic)

    if mask is not None:
        pic = pic * mask

    grid = utils.make_grid(pic, nrow=8, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

    im = Image.fromarray(ndarr)
    im.save(path)
    return ''
path = '/media/hdr/oo/result/1_4_V2/test_old_5/'
apath =  path + '/albedo/'
npath = path + '/normal/'
fpath = path + '/albedo/'
mpath = path + '/mask/'

relighting_path = '/media/hdr/oo/result/1_4_V2/test_old_5/relighting/'
facel = glob.glob(fpath+'/*')
facel.sort()
normall = glob.glob(npath+'/*')
normall.sort()
albedol = glob.glob(apath+'/*')
albedol.sort()
maskl = glob.glob(mpath+'/*')
maskl.sort()


l0 = 'rotate_light_00.txt'
l1 = 'rotate_light_01.txt'
l2 = 'rotate_light_02.txt'
l3 = 'rotate_light_03.txt'
l4 = 'rotate_light_04.txt'
l5 = 'rotate_light_05.txt'
l6 = 'rotate_light_06.txt'
pd_sh0 = pd.read_csv(l0, sep='\t', header=None, encoding=u'gbk')
sh0 = torch.tensor(pd_sh0.values).type(torch.float).reshape([1, 9])
pd_sh1 = pd.read_csv(l1, sep='\t', header=None, encoding=u'gbk')
sh1 = torch.tensor(pd_sh1.values).type(torch.float).reshape([1, 9])
pd_sh2 = pd.read_csv(l2, sep='\t', header=None, encoding=u'gbk')
sh2 = torch.tensor(pd_sh2.values).type(torch.float).reshape([1, 9])
pd_sh3 = pd.read_csv(l3, sep='\t', header=None, encoding=u'gbk')
sh3 = torch.tensor(pd_sh3.values).type(torch.float).reshape([1, 9])
pd_sh4 = pd.read_csv(l4, sep='\t', header=None, encoding=u'gbk')
sh4 = torch.tensor(pd_sh4.values).type(torch.float).reshape([1, 9])
pd_sh5 = pd.read_csv(l5, sep='\t', header=None, encoding=u'gbk')
sh5 = torch.tensor(pd_sh5.values).type(torch.float).reshape([1, 9])
pd_sh6 = pd.read_csv(l6, sep='\t', header=None, encoding=u'gbk')
sh6 = torch.tensor(pd_sh6.values).type(torch.float).reshape([1, 9])

options = '-delay 12 -loop 0 -layers optimize' # gif. need ImageMagick.

for i in range(0, 50, 1):
    tpf = facel[i]
    tpf = transform(Image.open(tpf))
    tp_a = albedol[i]
    albedo = transform(Image.open(tp_a))
    tp_n = normall[i]
    normal = transform(Image.open(tp_n))
    normal = normal.reshape([1, 3, 256, 256])
    mask = transform(Image.open(maskl[i]))
    unmask = torch.where(mask >0.1, torch.zeros_like(normal),torch.ones_like(normal))

    shading0 = get_shading_DPR_0802(2 * (normal - 0.5), sh0)
    shading1 = get_shading_DPR_0802(2 * (normal - 0.5), sh1)
    shading2 = get_shading_DPR_0802(2 * (normal - 0.5), sh2)
    shading3 = get_shading_DPR_0802(2 * (normal - 0.5), sh3)
    shading4 = get_shading_DPR_0802(2 * (normal - 0.5), sh4)
    shading5 = get_shading_DPR_0802(2 * (normal - 0.5), sh5)
    shading6 = get_shading_DPR_0802(2 * (normal - 0.5), sh6)

    rec0 = shading0*albedo
    rec1 = shading1*albedo
    rec2 = shading2*albedo
    rec3 = shading3*albedo
    rec4 = shading4*albedo
    rec5 = shading5*albedo
    rec6 = shading6*albedo

    recon_path = relighting_path + '/' + tp_a.split('/')[-1].split('.')[0].split('_')[0] +'/'
    os.system('mkdir -p {}'.format(recon_path))

    get_image_grid(rec0+unmask,recon_path +facel[i].split('.')[0].split('/')[-1]+ '_0.png')
    get_image_grid(rec1+unmask,recon_path +facel[i].split('.')[0].split('/')[-1]+ '_1.png')
    get_image_grid(rec2+unmask,recon_path +facel[i].split('.')[0].split('/')[-1]+ '_2.png')
    get_image_grid(rec3+unmask,recon_path + facel[i].split('.')[0].split('/')[-1]+'_3.png')
    get_image_grid(rec4+unmask,recon_path + facel[i].split('.')[0].split('/')[-1]+'_4.png')
    get_image_grid(rec5+unmask,recon_path +facel[i].split('.')[0].split('/')[-1]+ '_5.png')
    get_image_grid(rec6+unmask,recon_path + facel[i].split('.')[0].split('/')[-1]+'_6.png')


    subprocess.call('convert {} {}/*.png {}'.format(options, recon_path, recon_path + '/relighting.gif'), shell=True)

    # get_image_grid(albedo+unmask,recon_path +facel[i].split('.')[0].split('/')[-1]+ '_0_albedo.png')
    # get_image_grid(normal+unmask,recon_path + facel[i].split('.')[0].split('/')[-1]+'_1_normal.png')
    # get_image_grid(tpf+unmask,recon_path + facel[i].split('.')[0].split('/')[-1]+'_1_face.png')
    print('')


