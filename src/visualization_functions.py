import cv2
from PIL import Image
import torch
import numpy as np
from numpy import matlib as mb

from .models import VisionTransformer


def compute_spatial_similarity(prepooledA, pooledA, prepooledB, pooledB):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    score = np.dot(pooledA/np.linalg.norm(pooledA), pooledB/np.linalg.norm(pooledB))
    out_sz = (int(np.sqrt(prepooledA.shape[0])), int(np.sqrt(prepooledA.shape[0])))
    prepooledA_normed = prepooledA / np.linalg.norm(pooledA) / prepooledA.shape[0]
    prepooledB_normed = prepooledB / np.linalg.norm(pooledB) / prepooledB.shape[0]
    im_similarity = np.zeros((prepooledA_normed.shape[0], prepooledA_normed.shape[0]))
    for zz in range(prepooledA_normed.shape[0]):
        repPx = mb.repmat(prepooledA_normed[zz,:], prepooledA_normed.shape[0],1)
        t = np.multiply(repPx, prepooledB_normed)
        im_similarity[zz, :] = t.sum(axis=1)
    similarityA = np.reshape(np.sum(im_similarity, axis=1), out_sz)
    similarityB = np.reshape(np.sum(im_similarity, axis=0), out_sz)
    return similarityA, similarityB, score


def compute_rollout(attn_weights, start_layer=0, attn_head_agg='mean'):

    attn_weights = torch.tensor(attn_weights)
    if attn_head_agg == 'mean':
        attn_weights = torch.mean(attn_weights, dim=1)  # avg across heads
    elif attn_head_agg == 'min':
        attn_weights = torch.min(attn_weights, dim=1)[0]  # min across heads
    elif attn_head_agg == 'max':
        attn_weights = torch.max(attn_weights, dim=1)[0]  # max across heads
    else:
        raise ValueError('invalid input for "attn_head_agg", must be "mean", "min", or "max"')

    num_tokens = attn_weights[0].shape[1]
    eye = torch.eye(num_tokens).to(attn_weights[0].device)
    attn_weights = [attn_weights[i] + eye for i in range(len(attn_weights))]
    attn_weights = [attn_weights[i] / attn_weights[i].sum(dim=-1, keepdim=True) for i in range(len(attn_weights))]
    rollout_output = attn_weights[start_layer]
    for i in range(start_layer + 1, len(attn_weights)):
        rollout_output = attn_weights[i].matmul(rollout_output)
    return rollout_output


def generate_sim_maps(A_path, B_path, model, transform, use_gpu=True):

    model.eval()

    inpA = transform(Image.open(A_path).convert('RGB')).unsqueeze(0)
    inpB = transform(Image.open(B_path).convert('RGB')).unsqueeze(0)

    if use_gpu:
        inpA = inpA.cuda()
        inpB = inpB.cuda()

    with torch.no_grad():
        outputsA = list(model(inpA))
        outputsB = list(model(inpB))

    for i in range(len(outputsA)):
        outputsA[i] = outputsA[i].cpu().numpy().squeeze()
    for i in range(len(outputsB)):
        outputsB[i] = outputsB[i].cpu().numpy().squeeze()

    if type(model) == VisionTransformer:
        output_featA, prepooled_tokensA, attn_weightsA = outputsA
        output_featB, prepooled_tokensB, attn_weightsB = outputsB
    else:
        # ResNet
        output_featA, prepooled_tokensA = outputsA
        output_featB, prepooled_tokensB = outputsB

    simmapA, simmapB, score = compute_spatial_similarity(prepooled_tokensA, output_featA,
                                                         prepooled_tokensB, output_featB)

    if type(model) == VisionTransformer:

        original_shape = (simmapA.shape[0], simmapA.shape[1])

        rolloutA = compute_rollout(attn_weightsA).cpu()
        simmapA = torch.matmul(rolloutA, torch.tensor(simmapA.flatten()).float())
        simmapA = simmapA.detach().numpy().reshape(original_shape)

        rolloutB = compute_rollout(attn_weightsB).cpu()
        simmapB = torch.matmul(rolloutB, torch.tensor(simmapB.flatten()).float())
        simmapB = simmapB.detach().numpy().reshape(original_shape)

    return simmapA, simmapB, score


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def norm(a):
    c = a.copy()
    c = c - np.min(c)
    max_val = np.max(c)
    c = c / max_val
    return c
