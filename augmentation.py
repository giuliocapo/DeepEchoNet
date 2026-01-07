import cv2
import random
import numpy as np
from itertools import product
from globals import augmentation_parameters, dts_type, STRONG_AUG_DEFAULT


def pixel_shift(depth_img, shift):
    depth_img = depth_img + shift
    return depth_img


def apply_affine_pair(img, depth, max_rot=5, min_scale=0.97, max_scale=1.03):
    
    h, w = img.shape[:2]
    angle = random.uniform(-max_rot, max_rot)
    scale = random.uniform(min_scale, max_scale)
    
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    
    img_aug   = cv2.warpAffine(img,   M, (w, h), flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT_101)
    depth_aug = cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    
    return img_aug, depth_aug


def add_rgb_noise(img, sigma=0.02):
    
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    img_n = img + noise
    img_n = np.clip(img_n, 0.0, 1.0)
    return img_n


def augmentation2D(img, depth, print_info_aug, strong_aug=STRONG_AUG_DEFAULT):
    # Random flipping
    if random.uniform(0, 1) <= augmentation_parameters['flip']:
        img = img[::-1, :, :].copy()
        depth = depth[::-1, :, :].copy()
        if print_info_aug:
            print('--> Random flipped')
    
    # Random mirroring
    if random.uniform(0, 1) <= augmentation_parameters['mirror']:
        img = img[:, ::-1, :].copy()
        depth = depth[:, ::-1, :].copy()
        if print_info_aug:
            print('--> Random mirrored')
    
    # Channel swap
    if random.uniform(0, 1) <= augmentation_parameters['c_swap']:
        indices = list(product([0, 1, 2], repeat=3))
        policy_idx = random.randint(0, len(indices) - 1)
        img = img[..., list(indices[policy_idx])]
        if print_info_aug:
            print('--> Channel swapped')
    
    # Small rotation+scale solo se strong_aug=True
    if strong_aug and random.uniform(0, 1) <= augmentation_parameters.get('small_affine', 0.3):
        img, depth = apply_affine_pair(img, depth, max_rot=8, min_scale=0.95, max_scale=1.05)
        if print_info_aug:
            print('--> Small rotation+scale')
    
    
    # Shifting strategy
    if random.uniform(0, 1) <= augmentation_parameters['shifting_strategy']:
        gamma = random.uniform(0.9, 1.1)
        img = img ** gamma
        brightness = random.uniform(0.9, 1.1)
        img = img * brightness
        
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        img *= color_image
        img = np.clip(img, 0.0, 1.0)
        
        random_shift = random.randint(-10, 10)
        depth = pixel_shift(depth, shift=random_shift)
        if print_info_aug:
            print(f'--> Depth Shifted of {random_shift} cm/dm and Image randomly augmented')
    
    # RGB noise only if strong_aug=True
    if strong_aug and random.uniform(0, 1) <= augmentation_parameters.get('rgb_noise', 0.3):
        img = add_rgb_noise(img, sigma=0.02)
        if print_info_aug:
            print('--> RGB Gaussian noise')
    
    return img, depth
