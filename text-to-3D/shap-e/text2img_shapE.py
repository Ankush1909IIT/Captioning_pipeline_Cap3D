# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: September 04, 2023
#
# This code is licensed under the MIT License.
# ==============================================================================
import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.models.configs import model_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from IPython import embed
import os
import time
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--render_type', type = str, default='stf', choices=['stf','nerf'])
parser.add_argument('--ckpt', default='shapE_finetuned_with_330kdata.pth', type=str, help="path to finetuned model")
parser.add_argument('--save_name', default='Cap3D_test1', type=str, help="result files save to here")
parser.add_argument('--test_type', default='300', type=str, choices=['300','2k'], help="300 or 2k test sets")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
model.load_state_dict(torch.load('./model_ckpts/%s'%args.ckpt, map_location=device)['model_state_dict'])

diffusion = diffusion_from_config(load_config('diffusion'))

batch_size = 1
guidance_scale = 15.0


import pickle
import pandas as pd
test_uids = pickle.load(open('../example_material/test_uids_%s.pkl'%args.test_type, 'rb'))
captions = pd.read_csv('../example_material/Cap3D_automated_Objaverse.csv', header=None)

outdir = './shapE_inference/%s_%s'%(args.save_name, args.render_type)
os.makedirs(outdir, exist_ok=True)

for i in range(len(test_uids)):
    if os.path.exists(os.path.join(outdir, '%s_%d.png'%(test_uids[i],7))):
        continue
    prompt = captions[captions[0] == test_uids[i]][1].values[0]
    assert captions[captions[0] == test_uids[i]][0].values[0] == test_uids[i]

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    with torch.no_grad():
        render_mode = args.render_type
        size = 512

        cameras = create_pan_cameras(size, device)
        images = decode_latent_images(xm, latents, cameras, rendering_mode=render_mode)
        for j, image in enumerate(images):
        #    # Save each image to a file. You can choose the format you prefer (png, jpg, etc.)
            image.save(os.path.join(outdir,'%s_%d.png'%(test_uids[i],j)))
