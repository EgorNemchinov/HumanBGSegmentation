import argparse
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import json
import cv2

from modeling.build_model import Pose2Seg


def infer(model, dataset='cocoVal', n_biggest_people=None):
    if osp.isdir(f'./data/{dataset}'):
        im_root_dir = f'./data/{dataset}/images'
    elif osp.isdir(dataset):
        im_root_dir = osp.join(dataset, 'images')
    else:
        raise ValueError(f'No dataset at {dataset} or data/{dataset}')

    im_paths = [os.path.join(im_root_dir, f) for f in os.listdir(im_root_dir)
                if f.endswith(('.jpg', '.png')) and
                not any((f.startswith('.'), 'masksDL' in f, 'gt' in f))]

    model.eval()

    for i in tqdm(range(len(im_paths))):
        im_path = im_paths[i]
        im = cv2.imread(im_path)
        kps_path = im_path.replace('images', 'keypoints')
        for ext in ('.jpg', '.png'):
            kps_path = kps_path.replace(ext, '_keypoints.json')

        with open(kps_path, 'r') as f:
            kps = json.load(f)['people']
            kps = [np.array(kps[i]['pose_keypoints_2d']).reshape(17, 3) for i in range(len(kps))]
            heights = [(kp[:, 1].max() - kp[:, 1].min()) for kp in kps]
            kps = [kps[i] for i in np.argsort(heights)[::-1]][:n_biggest_people]
        assert (len(kps) == 0) or all(len(kp) == 17 for kp in kps), [len(kp) for kp in kps]
        if len(kps) == 0:
            output = [[im[:, :, 0] * 0]]
        else:
            kps = np.concatenate([np.float32(kp).reshape(1, 17, 3) for kp in kps], axis=0)
            empty_masks = np.ones((len(kps),) + im.shape[:2], dtype=float) * 255
            output = model([im], [kps], [empty_masks])

        mask = np.sum(output[0], axis=0).clip(0, 1)
        dir_path = os.path.dirname(im_path).replace('images', 'masks')
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        mask_path = im_path.replace('images', 'masks').replace('jpg', 'png')
        mask_im = (mask * 255).astype(np.uint8)
        assert cv2.imwrite(mask_path, mask_im), (mask_im.shape, mask_path)
    print('Saved masks, finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pose2Seg Testing")
    parser.add_argument(
        "weights",
        help="path to .pkl model weight",
        type=str,
    )
    parser.add_argument(
        "input_dir",
        help="Test on custom input dir",
    )
    parser.add_argument(
        '--depth',
        help='How deep down the directory hole we go',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--n_biggest_people',
        help='How many biggest people to select (decided by kpts bbox area)',
        default=None,
        type=int,
    )
    args = parser.parse_args()

    print('===========> loading model <===========')
    model = Pose2Seg().cuda()
    model.init(args.weights)

    print('===========>   testing    <===========')
    if args.depth == 1:
        infer(model, dataset=args.input_dir, n_biggest_people=args.n_biggest_people)
    elif args.depth == 2:
        dirs = sorted([osp.join(args.input_dir, d) for d in os.listdir(args.input_dir) if
                       osp.isdir(osp.join(args.input_dir, d))])
        for d in dirs:
            imgs = [f for f in os.listdir(osp.join(d, 'images')) if f.endswith(('.png', '.jpg'))]
            mask_dir = osp.join(d, 'masks')
            if osp.exists(mask_dir) and \
                    len([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))]) >= len(imgs):
                print(f'Directory {d} already has masks, skipping')
                continue
            infer(model, dataset=d, n_biggest_people=args.n_biggest_people)
    else:
        raise ValueError(f'Unsupported depth={args.depth}')

