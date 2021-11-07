import os, sys, argparse, shutil
import sklearn
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import glob
import time
import datetime
import imageio
import argparse
from tqdm import tqdm
from scipy import linalg
from skimage.io import imsave
from IPython import embed
from PIL import Image, ImageFilter

from tqdm import tqdm
from sklearn.neighbors import KDTree
from scipy.special import softmax
from scipy.spatial import ConvexHull


# libraries for face landmarks
sys.path.insert(1, "/home-local/demi/Research/TextureGen/face-alignment")
import face_alignment
from skimage import io
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp_dir", type=str, default="data/FT/merged_overlay")
    parser.add_argument("--uv_dir", type=str, default="data/FT/R200_metadata")
    parser.add_argument("--o_dir", type=str, default="data/FT")
    parser.add_argument("--resolution",type=int,default=2048)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--grid_knn", type=int, default=10)
    parser.add_argument("--color_knn", type=int, default=8)
    parser.add_argument("--subsample_freq", type=int, default=1)
    parser.add_argument("--subsample_start", type=int, default=50)
    parser.add_argument("--n_eye_samples", type=int, default=1)
    parser.add_argument("--dist_T", type=float, default=10.0)
    parser.add_argument("--black_t", type=int, default=150)
    parser.add_argument("--dist_div", type=float, default=8)
    parser.add_argument("--eye_mask",type=int, default=1)
    parser.add_argument("--mind_t", type=float,default=0.6)
    parser.add_argument("--blur",type=int,default=-1)
    args = parser.parse_args()
    return args

def debug_uv(res, uvs, pixs, logdir):
    test=np.zeros((res,res,3))
    test[...,2]=0.5
    test_uvs=np.round(res*uvs).astype(np.int)
    test[res-1-test_uvs[:,1],test_uvs[:,0]]=pixs
    img=Image.fromarray((255*test).astype(np.uint8))
    img.save(f"{logdir}/df_debug_full.png")


def generate_pairings(args, comp_dir, uv_dir):
    uv_files=glob.glob(f"{uv_dir}/*.npy")
    uv_files.sort()
    comp_files=glob.glob(f"{comp_dir}/*.png")
    comp_files.sort()
    nframes=len(uv_files)
    assert(nframes==len(comp_files))

    subsample_freq = args.subsample_freq
    subsample_start = args.subsample_start

    if subsample_freq > 0:
        idxs = []
        for i in range((nframes-1)//2,-1,-subsample_freq):
            if i >= subsample_start and i < nframes-subsample_start:
                idxs.append(i)
        for i in range(nframes//2,nframes,subsample_freq):
             if i >= subsample_start and i < nframes-subsample_start:
                idxs.append(i)
        idxs.sort()          
    else:
        idxs=list(range(0,nframes))

    print(f"USING FRAMES: {idxs}")
    center_sample=len(idxs)//2
    
    center_sample=43  # HACK(demi)
    for idx in range(len(idxs)):
        if "94" in comp_files[idx]:
            center_sample=idx
    """
    print("comp_files=",comp_files)
    print("len(idxs=",len(idxs))
    print("comp_files[idx[center_sample=", comp_files[idxs[center_sample]])
    print("eyes center=", comp_files[idxs[center_sample]])
    """

    n_eye_samples = args.n_eye_samples
    if n_eye_samples is None:
        pairs=[(comp_files[idx], uv_files[idx], "DEFAULT") for idx in idxs]
    else:
        pairs=[(comp_files[idxs[i]], uv_files[idxs[i]], 
                "DEFAULT" if abs(i-center_sample)<n_eye_samples else "NO_EYE") 
                for i in range(len(idxs))]
    return pairs

def generate_pixmap(args, uv_im_pairs, logdir):
    #print("GENERATING PIXMAP...")
    uvs=[]
    pixs=[]
    lefteye=[]
    righteye=[]
    os.makedirs(logdir,exist_ok=True)
    for im_cmp, im_uv, mode in uv_im_pairs:
        uv=np.load(im_uv,allow_pickle=True)
        img=np.asarray(Image.open(im_cmp))
        imgmsk=(np.sum(img,axis=-1)>args.black_t)
        if mode=="NO_EYE":
            ys,xs=np.nonzero(np.multiply(imgmsk, 
                                         np.multiply(uv[:,:,0]>0, uv[:,:,1]<(1-(250/2048)))))
        else:
            ys,xs=np.nonzero(np.multiply(imgmsk , uv[:,:,0]>0))
        uvs.append(uv[ys,xs,:2])
        pixs.append(img[ys,xs,:3]/255)
        uv[ys,xs]=img[ys,xs,:3]/255
        img=Image.fromarray((255*uv).astype(np.uint8))
        img.save(f"{logdir}/overlay.{os.path.basename(im_cmp)}")

        ys,xs=np.nonzero(np.multiply(uv[:,:,0]>0.85, uv[:,:,1]>(1-(250/2048))))
        lefteye.append(uv[ys,xs])
        ys,xs=np.nonzero(np.multiply(uv[:,:,0]<0.15, uv[:,:,1]>(1-(250/2048))))
        righteye.append(uv[ys,xs])

    uvs=np.concatenate(uvs,axis=0)
    pixs=np.concatenate(pixs,axis=0)
    lefteye=np.concatenate(lefteye,axis=0)
    righteye=np.concatenate(righteye,axis=0)
    debug_uv(1024,uvs,pixs,logdir)
    return uvs, pixs, lefteye, righteye

def main():
    args=parse_cmd_line()
    comp_dir = args.comp_dir
    uv_dir   = args.uv_dir
    o_dir    = args.o_dir
    res      = args.resolution
    debug    = args.debug
    print("args=",args)

    #subsample=10, n_eye_samples=2
    files=generate_pairings(args, comp_dir, uv_dir)

    if not os.path.exists(f"{o_dir}/sandbox"):
        # directory for sanity checks
        os.makedirs(f"{o_dir}/sandbox")
    
    textures = []
    # format for FLAME
    for im_cmp, im_uv, mode in tqdm(files, desc="running generation"):
        name = im_cmp.split("/")[-1].split(".")[0]

        
        # check if masked aggregation makes sense
        uvs, pixs, le, re=generate_pixmap(args, [(im_cmp, im_uv, mode)],o_dir)
        mask = np.zeros((res, res))
        masked_img = np.zeros((res, res, 3))
        if not os.path.exists(f"{o_dir}/sandbox/masked_img"):
            os.makedirs(f"{o_dir}/sandbox/masked_img")

        for i in range(uvs.shape[0]):
            
            xx, yy = np.rint(uvs[i] * res).astype(np.int32).clip(0, res-1)
            x = res-1-yy
            y = xx
            mask[x][y] = 1
            masked_img[x,y,:] = pixs[i]
        Image.fromarray((masked_img*255).astype(np.uint8)).save(f"{o_dir}/sandbox/masked_img/{name}.png")
        if not os.path.exists(f"{o_dir}/agg_masks"):
            os.makedirs(f"{o_dir}/agg_masks")
        np.save(f"{o_dir}/agg_masks/{name}.npy", mask)
 
        # resize and copy image file to FFHQ folder
        img = Image.open(im_cmp).resize((256, 256))
        img.save(f"FFHQ/{name}.png")

        # get landmarks
        input = io.imread(f"FFHQ/{name}.png")
        preds = fa.get_landmarks(input)
        np.save(f"FFHQ/{name}.npy", preds[0])
    

        # get segmentation
        img_arr = np.array(img)
        assert img_arr.shape == (256, 256, 3)
        seg_mask = img_arr.sum(axis=2) > 5
        assert seg_mask.shape == (256, 256)
        # DEBUG(demi): save segmask to cnofirm
        if not os.path.exists(f"{o_dir}/sandbox/seg_masks"):
            os.makedirs(f"{o_dir}/sandbox/seg_masks")
        Image.fromarray((seg_mask).astype(np.uint8)*255).save(f"{o_dir}/sandbox/seg_masks/{name}.png")
        np.save(f"FFHQ_seg/{name}.npy", seg_mask.astype(np.float64))

        # run FLAME to get texture maps
        if not os.path.exists(f"{o_dir}/results"):
            os.makedirs(f"{o_dir}/results")
        os.system(f"python photometric_fitting.py {name} cuda {o_dir}/results 1>{o_dir}/log.out 2>{o_dir}/log.err")
        texture = np.load(f"{o_dir}/results/{name}_textures.npy")
        textures.append(texture)

        # TODO(demi): save [name]_textures.npy

    # average raw textures (before clipping), and save the final texture image
    avg_texture = np.stack(textures, axis=0).mean(axis=0)
    assert avg_texture.shape == (256, 256, 3)
    avg_texture_img = avg_texture.clip(0,1)
    avg_texture_img = (avg_texture_img*255).astype('uint8')
    imsave(f"{o_dir}/results/avg_texture.png", avg_texture_img)
    


if __name__ == "__main__":

    main()
