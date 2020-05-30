import numpy as np
import cv2
import os.path as osp
import argparse
import os


def load_img_mask(img_dir, index, skip_last=0):
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('_img.png')])
    if skip_last > 0:
        files = files[:-skip_last]
    if index >= len(files):
        return None, None
    filename = files[index]
    img = cv2.imread(osp.join(img_dir, filename))
    mask = cv2.imread(osp.join(img_dir, filename.replace('_img', '_masksDL')), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None, None
    return img, mask[:, :, np.newaxis]


def main(img_dir, bg_out_path, step=8, inpaint=True, skip_first=0, skip_last=0):
    i = 1 + skip_first
    img, mask = load_img_mask(img_dir, i)
    frame_sum, mask_sum = 0, 0
    while (img is not None) and (mask is not None):
        frame_sum = frame_sum + img * (1 - mask / 255.)
        mask_sum = mask_sum + (1 - mask / 255.)
        i += step
        img, mask = load_img_mask(img_dir, i, skip_last=skip_last)
    print(f'Total images used for extracting bg: {i//step}')
    coords = np.argwhere(mask_sum > 0)
    frame_sum[coords[:, 0], coords[:, 1], :] = frame_sum[coords[:, 0], coords[:, 1], :] / mask_sum[coords[:, 0], coords[:, 1]]
    frame_sum = frame_sum.clip(0, 255).astype(np.uint8)
    if inpaint:
        inpaint_radius = int(0.03 * np.mean(frame_sum.shape[:2]) + 0.5)
        mask = cv2.dilate((mask_sum == 0).astype(np.uint8) * 255, np.ones((inpaint_radius // 2, inpaint_radius // 2)))
        frame_sum = cv2.inpaint(frame_sum, mask, inpaint_radius, cv2.INPAINT_TELEA)
    cv2.imwrite(bg_out_path, frame_sum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir')
    parser.add_argument('bg_out_path')
    parser.add_argument('--step', type=int, default=8)
    parser.add_argument('--no_inpainting', action='store_true')
    parser.add_argument('--skip_first', type=int, default=0, help='Exclude first N frames')
    parser.add_argument('--skip_last', type=int, default=0, help='Exclude last N frames')
    args = parser.parse_args()

    main(args.img_dir, args.bg_out_path, args.step, inpaint=(not args.no_inpainting), skip_first=args.skip_first, skip_last=args.skip_last)
