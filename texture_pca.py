import os, sys
import sklearn
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from glob import glob
import time
import datetime
import imageio
import argparse
from tqdm import tqdm
from scipy import linalg
from skimage.io import imsave
from IPython import embed
from PIL import Image, ImageFilter

sys.path.append('./models/')
from FLAME import FLAME, FLAMETex
from renderer import Renderer
import util
torch.backends.cudnn.benchmark = True


def convert_to_texture(args, input_path):
    image = cv2.resize(cv2.imread(input_path), (args.imsize, args.imsize)).astype(np.float32)
    #print("image.shape=", image.shape)
    image = image[:, :, [2,1,0]]
    #imsave("test_resize.png", image)
    texture = np.expand_dims(image.transpose(2,0,1), axis=0)  # todo(demi): do we need to transpose the rgb channels as well (see L193 in main py)
    assert texture.shape == (1, 3, args.imsize, args.imsize)
    return texture

def get_coefficient(args, texture):
    orig_texture = texture.copy()
   

    tex_space = np.load(args.tex_space_path)
    texture_mean = tex_space['mean'].reshape(1, -1)
    texture_basis = tex_space['tex_dir'].reshape(-1, 200)
    num_components = texture_basis.shape[1]
    # assume batch size = 1
    texture_mean = texture_mean.reshape(-1)
    texture_basis = texture_basis[:, :args.tex_params]

    def l2_norm(x,y):
        return np.sqrt(np.sum((x-y)**2))


    if args.mode == "dot_product":
        # the basis is not actually orthonormal here
        raise NotImplementedError
    elif args.mode == "linalg":
        print("texture.shape=", texture)
        assert texture.shape[1] == 3
        texture = texture[:, [2,1,0], :,:]
        texture = texture.transpose(0,2,3,1)
        assert texture.shape[0] == 1 and texture.shape[-1] == 3
        # NB(demi): batch size is 1
        texture = texture.reshape(-1)
        N = texture.shape[0]
        assert texture_mean.shape == (N,) and texture_basis.shape == (N, args.tex_params)
        texcode = linalg.solve(texture_basis, texture-texture_mean) + texture_mean
        return texcode    
    elif "optim_l2" in args.mode:
        assert texture.shape[1] == 3
        texture = texture[:, [2,1,0], :, :]
        texture = texture.transpose(0, 2, 3, 1)
        assert texture.shape[0] == 1 and texture.shape[-1] == 3
        if "wb_mul_mean" in args.mode:
            print("white balance - channel-wise multiplication to PCA mean")
            assert texture.reshape(-1,3).shape == texture_mean.reshape(-1,3).shape
            src_mean = np.mean(texture.reshape(-1, 3), axis=0, keepdims=True)
            tgt_mean = np.mean(texture_mean.reshape(-1, 3), axis=0, keepdims=True)
            assert src_mean.shape == (1,3) and tgt_mean.shape == (1,3)
            texture = texture.reshape(-1, 3)
            assert texture.shape[1] == 3
            texture = texture * (tgt_mean/src_mean)
            texture = np.clip(texture, 0, 255)

            # save image
            new_img = texture.reshape(args.imsize, args.imsize, 3)[:, :, [2,1,0]]
            imsave(f"{args.output}/new_texture.png", new_img)

            mean_img = texture_mean.reshape(args.imsize, args.imsize, 3)[:, :, [2,1,0]]
            imsave(f"{args.output}/mean_texture.png", mean_img)
        elif "wb_mul_ref" in args.mode:
            print("white balance -- channel-wise mul to reference image")
            ref_texture = convert_to_texture(args, "test_results/00000.png")
            ref_texture = ref_texture[:, [2,1,0], :, :].transpose(0,2,3,1).reshape(args.imsize, args.imsize, 3)

            src_mean = np.mean(texture.reshape(-1, 3), axis=0, keepdims=True)
            tgt_mean = np.mean(ref_texture.reshape(-1, 3), axis=0, keepdims=True)
            assert src_mean.shape == (1,3) and tgt_mean.shape == (1,3)
            texture = texture.reshape(-1, 3)
            assert texture.shape[1] == 3
            texture = texture * (tgt_mean/src_mean)
            texture = np.clip(texture, 0, 255)

            # save image
            new_img = texture.reshape(args.imsize, args.imsize, 3)[:, :, [2,1,0]]
            imsave(f"{args.output}/new_texture.png", new_img)
        else:
            print("no white balancing")
            pass

        texture = texture.reshape(-1)
        texture = texture - texture_mean

        if "face" in args.mode:
            print("optimize only on face pixels")
            # NB(demi): assume it's FT, we use a FT alpha mask
            alpha_mask = np.load(args.alpha_mask_path)
            # NB(demi): hacky now, use different image libraries
            alpha_mask_img = Image.fromarray((alpha_mask*255).astype(np.uint8)).resize((args.imsize, args.imsize))

            resized_alpha_mask = (np.array(alpha_mask_img)>=args.mask_t)
            Image.fromarray((resized_alpha_mask.astype(np.uint8)*255)).save(f"{args.output}/resized_binary_alpha_mask.png")

            resized_alpha_mask = np.stack([resized_alpha_mask,resized_alpha_mask,resized_alpha_mask],axis=2)
            assert resized_alpha_mask.shape == (args.imsize, args.imsize, 3)
            resized_alpha_mask = resized_alpha_mask.reshape(-1)
            assert resized_alpha_mask.shape[0] == texture_basis.shape[0] and texture_basis.shape[0] == texture.shape[0]
            texcode, residues, _, _ = linalg.lstsq(texture_basis[resized_alpha_mask], texture[resized_alpha_mask])
        else:
            print("optimize on all pixels")
            texcode, residues, _, _ = linalg.lstsq(texture_basis, texture)
        assert texcode.shape == (args.tex_params,)
        print("residues=", residues, " residuces avg=", np.sqrt(residues))
        return texcode
    elif args.mode == "sgd_regressor":
        from sklearn.linear_model import SGDRegressor
        assert texture.shape[1] == 3
        texture = texture[:, [2,1,0], :, :]
        texture = texture.transpose(0, 2, 3, 1)
        assert texture.shape[0] == 1 and texture.shape[-1] == 3
        texture = texture.reshape(-1)
        texture = texture - texture_mean

        reg = SGDRegressor(max_iter=1000, tol=1e-3, fit_intercept=False)
        reg.fit(texture_basis, texture)
        texcode = reg.get_params(deep=False)
        embed()
        ttexture = np.einsum("ij,j->i", texture_basis, texcode)
        loss = l2_norm(ttexture, texture)
        print("loss=",loss)
    else:
        raise NotImplementedError
    return None

def visualize(args, tex_code):
    tex_space = np.load(args.tex_space_path)
    texture_mean = tex_space['mean'].reshape(1, -1)
    texture_basis = tex_space['tex_dir'].reshape(-1, 200)
    num_components = texture_basis.shape[1]
    # assume batch size = 1
    texture_mean = texture_mean.reshape(-1)
    texture_basis = texture_basis[:, :args.tex_params]


    def viz(tex_code, filename):
        texture = texture_mean + (texture_basis*tex_code).sum(-1)
        texture = texture.reshape(512, 512, 3)
        texture = texture[:, :, [2,1,0]]
        texture = np.clip(texture, 0, 255)
        imsave(filename, texture)

    viz(tex_code, f"{args.output}/default.png") 
    for i in range(args.n_comp):
        tex_code[i] += args.vary
        viz(tex_code, f"{args.output}/vary_pos{i}.png")
        tex_code[i] -= args.vary * 2
        viz(tex_code, f"{args.output}/vary_neg{i}.png")
        tex_code[i] += args.vary

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texture Map PCA")

    # optimization
    parser.add_argument("--input", type=str, required=True, help="input texture map")
    parser.add_argument("--tex_space_path", type=str, required=False, default="./data/FLAME_texture.npz", help="texture space path (.npz)")
    parser.add_argument("--tex_params", type=int, required=False, default=50, help="number of dimensions for optimization")  # todo(demi): double check
    parser.add_argument("--imsize", type=int, required=False, default=512, help="image size")  # todo(demi): required? otherwise tex space path will need to change?
    parser.add_argument("--mode", type=str, required=False, default="dot_product", help="ways to get coefficients [dot_product, optim_l2]")
    parser.add_argument("--alpha_mask_path", type=str, required=False, default="data/FT_default_sandbox/alpha_mask.npy", help="path of alpha mask")
    parser.add_argument("--mask_t", type=int, required=False, default=100, help="threshold for face mask: [0, 255]")

    # visualization
    parser.add_argument("--vary", type=float, required=False, default=2.0, help="how much to vary per component for visualization")
    parser.add_argument("--n_comp", type=int, required=False, default=5, help="number of components to vary for visualization")
    parser.add_argument("--output", type=str, required=True, help="output varied texture map directory")

    args = parser.parse_args()
    
    # main
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    texture = convert_to_texture(args, args.input)
    tex_code = get_coefficient(args, texture)
    visualize(args, tex_code)

    

