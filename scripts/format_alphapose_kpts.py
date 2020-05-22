import argparse
import os
from tqdm import tqdm
from collections import defaultdict
import json


def format_alphapose_to_openpose(alphapose_json, out_kpts_dir, frames_dir):
    os.makedirs(out_kpts_dir, exist_ok=True)
    with open(alphapose_json, 'r') as f:
        kpts = json.load(f)
    kpts_by_frame = defaultdict(list)
    for i in kpts:
        kpts_by_frame[int(i['image_id'].split('.')[0])].append(i)

    def convert_to_openpose_dict(frame_ap_kpts):
        people = [{'pose_keypoints_2d': kp['keypoints']} for kp in frame_ap_kpts]
        return {'version': 1.0, 'people': people}

    frames = sorted(os.listdir(frames_dir))
    for ind in tqdm(range(0, len(frames))):
        kp = kpts_by_frame[ind]
        path = os.path.join(out_kpts_dir, frames[ind].replace('.png', '_keypoints.json'))
        with open(path, 'w') as f:
            json.dump(convert_to_openpose_dict(kp), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'alphapose_results_json', help='AlphaPose output json'
    )
    parser.add_argument('out_kpts_dir', help='Folder for output keypoints in .json format for each frame')
    parser.add_argument('--frames_dir', help='Reference video frames')
    args = parser.parse_args()

    format_alphapose_to_openpose(args.alphapose_results_json, args.out_kpts_dir, args.frames_dir)