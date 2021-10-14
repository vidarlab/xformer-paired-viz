import argparse
import os

import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

from dataset_norms import dataset_norms
from models import VisionTransformer, ResNetV2
from visualization_functions import generate_sim_maps, show_cam_on_image, norm


def main():

    parser = argparse.ArgumentParser(description='Paired Image Visualization')
    parser.add_argument('--imageA', type=str, default='examples/Hotels-50k/images/img1.png')
    parser.add_argument('--imageB', type=str, default='examples/Hotels-50k/images/img2.png')
    parser.add_argument('--model_weights', type=str, default='weights/Hotels-50k/vit.pth')
    parser.add_argument('--model_type', type=str, default='ViT-B', help='"ViT-B" or "resnet-101"')
    parser.add_argument('--patch_size', type=int, default=16, help='ViT-B patch-size, either 16 or 32')
    parser.add_argument('--dataset', type=str, default='Hotels-50k', help='Choices: Hotels-50k, SOP, GLM')
    parser.add_argument('--device', type=str, default='gpu', help='use "GPU" or "CPU"')
    parser.add_argument('--save_dir', type=str, default='examples/Hotels-50k/results')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.model_type == 'ViT-B':
        config = {'output_size': 512, 'init_head': True, 'classifier': 'token', 'hidden_size': 768, 'img_size': 256,
                  'patch_size': (args.patch_size, args.patch_size), 'load_from': None, 'dropout_rate': 0, 'vis': True,
                  'num_layers': 12, 'mlp_dim': 3072, 'num_heads': 12, 'global_feature_embedding': 'mean',
                  'attention_dropout_rate': 0}
        model = VisionTransformer(config=config)
    elif args.model_type == 'resnet-101':
        config = {'arch': 'r101', 'width_factor': 1, 'output_size': 512, 'zero_head': False, 'weights_file': None}
        model = ResNetV2(config=config)
    else:
        raise ValueError(f"Invalid model_type argument: '{args.model_type}' "
                         "model_type argument must be either ViT-B or resnet-101")

    if args.device.lower() == 'gpu':
        use_gpu = True
        model.cuda()
    else:
        use_gpu = False

    weights = torch.load(args.model_weights)
    weights = {k.replace('module.', ''): weights[k] for k in weights.keys()}
    model.load_state_dict(weights)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(dataset_norms[args.dataset]['mean'], dataset_norms[args.dataset]['std'])
    ])

    transform_greyscale = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3)
    ])

    imageA_path = args.imageA
    imageB_path = args.imageB

    simmapA, simmapB, sim_score = generate_sim_maps(imageA_path, imageB_path, model, transform, use_gpu=use_gpu)
    imageA_gs = transform_greyscale(Image.open(imageA_path).convert("RGB"))
    imageA_gs = np.moveaxis(imageA_gs.numpy(), 0, 2)

    imageB_gs = transform_greyscale(Image.open(imageB_path).convert("RGB"))
    imageB_gs = np.moveaxis(imageB_gs.numpy(), 0, 2)

    simmapA = norm(cv2.resize(simmapA, (256, 256), interpolation=cv2.INTER_CUBIC))
    simmapA = show_cam_on_image(imageA_gs, simmapA, use_rgb=True)

    simmapB = norm(cv2.resize(simmapB, (256, 256), interpolation=cv2.INTER_CUBIC))
    simmapB = show_cam_on_image(imageB_gs, simmapB, use_rgb=True)

    simmapA = Image.fromarray(simmapA)
    simmapB = Image.fromarray(simmapB)

    imageA_name = os.path.split(imageA_path)[1].split('.')[0]
    imageB_name = os.path.split(imageB_path)[1].split('.')[0]

    simmapA.save(f'{args.save_dir}/{imageA_name}_simmap_with_{imageB_name}.png')
    simmapB.save(f'{args.save_dir}/{imageB_name}_simmap_with_{imageA_name}.png')


if __name__ == '__main__':
    main()
