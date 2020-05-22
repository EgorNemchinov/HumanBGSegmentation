import numpy as np
import cv2
import os.path as osp
import argparse
import os

def load_img_mask(img_dir, index):
    img = cv2.imread(osp.join(img_dir, f'{index:04d}_img.png'))
    mask = cv2.imread(osp.join(img_dir, f'{index:04d}_masksDL.png'), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None, None
    return img, mask[:, :, np.newaxis]


def load_img_mask(img_dir, index):
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    if index >= len(files):
        return None, None
    filename = files[index]
    img = cv2.imread(osp.join(img_dir, filename))
    mask = cv2.imread(osp.join(img_dir.replace('images', 'masks'), filename), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None, None
    return img, mask[:, :, np.newaxis]

def load_img_mask(img_dir, index):
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('_img.png')])
    if index >= len(files):
        return None, None
    filename = files[index]
    img = cv2.imread(osp.join(img_dir, filename))
    mask = cv2.imread(osp.join(img_dir, filename.replace('_img', '_masksDL')), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None, None
    return img, mask[:, :, np.newaxis]


def main(img_dir, bg_out_path, step=8, tf_masks=False):
    i = 1
    img, mask = load_img_mask(img_dir, i)
    frame_sum, mask_sum = 0, 0
    while (img is not None) and (mask is not None):
        frame_sum = frame_sum + img * (1 - mask / 255.)
        mask_sum = mask_sum + (1 - mask / 255.)
        i += step
        img, mask = load_img_mask(img_dir, i)
    print(f'Total images used for extracting bg: {i//step}')
    coords = np.argwhere(mask_sum > 0)
    frame_sum[coords[:, 0], coords[:, 1], :] = frame_sum[coords[:, 0], coords[:, 1], :] / mask_sum[coords[:, 0], coords[:, 1]]
    cv2.imwrite(bg_out_path, frame_sum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir')
    parser.add_argument('bg_out_path')
    parser.add_argument('--tf_masks', action='store_true')
    parser.add_argument('--step', type=int, default=8)
    args = parser.parse_args()

    main(args.img_dir, args.bg_out_path, args.step, tf_masks=args.tf_masks)
