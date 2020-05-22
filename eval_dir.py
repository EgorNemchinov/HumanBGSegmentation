import argparse
import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import cv2


def calculate_metrics(gen_mask, gt_mask, threshold=0, skip_undetected=False):
    uncertain_zone = np.all(gt_mask == 128, axis=2)
    gt_human_mask = np.any(gt_mask > 0, axis=2) & (~uncertain_zone)

    gen_mask_bin = gen_mask > threshold
    gen_mask_bin = gen_mask_bin & (~uncertain_zone)

    if gt_human_mask.sum() == 0:
        return defaultdict(lambda: np.nan)
    if skip_undetected and gen_mask_bin.sum() == 0:
        return defaultdict(lambda: np.nan)

    overlap_sum = (gen_mask_bin & gt_human_mask).sum()
    union_sum = (gen_mask_bin | gt_human_mask).sum()

    gen_mask_soft = gen_mask / 255.0 * (~uncertain_zone).astype(int)
    overlap_sum_soft = (gen_mask_soft * gt_human_mask.astype(int)).sum()
    union_sum_soft = (gen_mask_soft + gt_human_mask.astype(int)).clip(0, 1).sum()

    metrics = dict()
    if overlap_sum == 0:
        return defaultdict(lambda: np.nan)
    metrics['iou'] = overlap_sum / union_sum
    metrics['f1'] = 2 * overlap_sum / (union_sum + overlap_sum)
    metrics['precision'] = overlap_sum / gen_mask_bin.sum()
    metrics['recall'] = overlap_sum / gt_human_mask.sum()

    metrics['iou_soft'] = overlap_sum_soft / union_sum_soft
    metrics['f1_soft'] = 2 * overlap_sum_soft / (union_sum_soft + overlap_sum_soft)
    # metrics['precision'] = overlap_sum_soft / gen_mask_soft.sum()
    # metrics['recall'] = overlap_sum_soft / gt_human_mask.sum()

    return metrics


def eval_dir(gen_masks_dir, gt_masks_dir, threshold=64):
    gen_masks_paths = [
        os.path.join(gen_masks_dir, f) for f in os.listdir(gen_masks_dir) if f.endswith('.png')
    ]
    assert len(os.listdir(gt_masks_dir)) >= len(
        gen_masks_paths
    ), f'More generated masks than gt: {len(gen_masks_paths)} vs {gt_masks_dir}'

    metric_list = defaultdict(list)
    for gen_mask_path in gen_masks_paths:
        gen_mask = cv2.imread(gen_mask_path, 0)
        assert gen_mask is not None, gen_mask_path

        gt_path = os.path.join(gt_masks_dir, os.path.basename(gen_mask_path))
        gt_mask = cv2.imread(gt_path)
        assert gt_mask is not None, gt_path
        assert gen_mask.shape == gt_mask.shape[:2], (gen_mask.shape, gt_mask.shape)

        metrics = calculate_metrics(gen_mask, gt_mask, threshold=threshold)
        for key, value in metrics.items():
            metric_list[key].append(value)

    return metric_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'gen_mask_dirnames', help='Names of folders with generated masks separated by a comma'
    )
    parser.add_argument('gt_mask_dirname', help='Folder name with ground truth masks')
    parser.add_argument('parent_dir', help='Folder name with ground truth masks')
    parser.add_argument('--csv_dir', help='Where to write results as .csv for each metric')
    args = parser.parse_args()

    os.makedirs(args.csv_dir, exist_ok=True)
    assert os.path.isdir(args.parent_dir), f'Not a directory: {args.parent_dir}'
    gen_mask_dirnames = [d.strip() for d in args.gen_mask_dirnames.split(',')]

    vid_names = sorted(
        [d for d in os.listdir(args.parent_dir) if os.path.isdir(os.path.join(args.parent_dir, d))]
    )
    metric_rows = defaultdict(lambda: [[''] + vid_names + ['Mean']])

    for method in tqdm(gen_mask_dirnames):
        tqdm.write(f'Method in progress: {method}')
        metric_lists = defaultdict(lambda: [method])
        for dir_name in vid_names:
            dir_metrics = eval_dir(
                os.path.join(args.parent_dir, dir_name, method),
                os.path.join(args.parent_dir, dir_name, args.gt_mask_dirname),
            )
            for key in set(metric_lists.keys()).union(dir_metrics.keys()):
                metric_lists[key].append(
                    np.nanmean(dir_metrics[key]) if len(dir_metrics[key]) > 0 else 0.0
                )
        metric_means = {key: np.nanmean(values[1:]) for key, values in metric_lists.items()}
        for metric in metric_lists:
            metric_rows[metric].append(metric_lists[metric] + [metric_means[metric]])
        tqdm.write(f'Finished!\n-------------')
    for metric, rows in metric_rows.items():
        np.savetxt(
            os.path.join(args.csv_dir, f'{metric}.csv'),
            np.array(rows).astype(str),
            fmt='%s',
            delimiter=',',
        )
