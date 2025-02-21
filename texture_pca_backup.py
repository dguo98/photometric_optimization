import os, sys
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

sys.path.append('./models/')
from FLAME import FLAME, FLAMETex
from renderer import Renderer
import util
torch.backends.cudnn.benchmark = True


def convert_to_texture(args, input_path):
    image = cv2.resize(cv2.imread(input_path), (args.imsize, args.imsize)).astype(np.float32)
    print("image.shape=", image.shape)
    image = image[:, :, [2,1,0]]
    imsave("test_resize.png", image)
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

    np.save("orig_texture.npy", orig_texture)
    # hack(demi): check original texture
    def l2_norm(x,y):
        return np.sqrt(np.sum((x-y)**2))

    o_texture = np.load("test_results/o_texture.npy")
    s_texture = np.load("test_results/s_texture.npy")
    if args.imsize == 256:
        print("diff original = ", l2_norm(o_texture, orig_texture))
    elif args.imsize == 512:
        print("diff original s = ", l2_norm(s_texture, orig_texture))
    else:
        pass
    embed()
    
    i_texture = np.load("test_results/i_texture.npy")
 
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
    elif args.mode == "optim_l2":
        assert texture.shape[1] == 3
        texture = texture[:, [2,1,0], :, :]
        texture = texture.transpose(0, 2, 3, 1)
        assert texture.shape[0] == 1 and texture.shape[-1] == 3
        texture = texture.reshape(-1)
        texture = texture - texture_mean
        np.save("intermediate_texture.npy", texture)
        texcode, residues, _, _ = linalg.lstsq(texture_basis, texture)
        assert texcode.shape == (args.tex_params,)
        print("residues=", residues, " residuces avg=", np.sqrt(residues))

        # HACk(demi): look into gt
        ttexcode = np.load("test_results/00000_texcode.npy")
        ttexture = np.einsum('ij,j->i',texture_basis, ttexcode)
        assert ttexture.shape == texture.shape
        lsq = np.mean((texture - ttexture)**2)
        print("gt avg (x-x')^2=", lsq)
        print("gt 2-norm=", np.sqrt(np.sum((texture-ttexture)**2)))

        ttexcode = texcode
        ttexture = np.einsum('ij,j->i',texture_basis, ttexcode)
        assert ttexture.shape == texture.shape
        lsq = np.mean((texture - ttexture)**2)
        print("predict avg (x-x')^2=", lsq)
        print("predict 2-norm=", np.sqrt(np.sum((texture-ttexture)**2)))


        return texcode
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

    if not os.path.exists(args.output):
        os.makedirs(args.output)

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

    # visualization
    parser.add_argument("--vary", type=float, required=False, default=2.0, help="how much to vary per component for visualization")
    parser.add_argument("--n_comp", type=int, required=False, default=5, help="number of components to vary for visualization")
    parser.add_argument("--output", type=str, required=True, help="output varied texture map directory")

    args = parser.parse_args()
    
    # main
    texture = convert_to_texture(args, args.input)
    tex_code = get_coefficient(args, texture)

    # HACK(demi): sanity check
    #tex_code = np.load("test_results/00000_texcode.npy")

    visualize(args, tex_code)

    

