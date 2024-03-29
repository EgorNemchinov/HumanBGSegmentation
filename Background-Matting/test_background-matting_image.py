from __future__ import print_function

import os, glob, time, argparse, pdb, cv2
# import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from functions import *
from networks import ResnetConditionHR

from skeleton_feat import read_kpts_json, gen_skeletons, apply_crop_kpts

torch.set_num_threads(1)
# os.environ["CUDA_VISIBLE_DEVICES"]="4"
print('CUDA Device: ' + os.environ["CUDA_VISIBLE_DEVICES"])

"""Parses arguments."""
parser = argparse.ArgumentParser(description='Background Matting.')
parser.add_argument('-m', '--trained_model', type=str, default='real-fixed-cam',
                    help='Trained background matting model')
parser.add_argument('-o', '--output_dir', type=str, required=True,
                    help='Directory to save the output results. (required)')
parser.add_argument('-i', '--input_dir', type=str, required=True, help='Directory to load input images. (required)')
parser.add_argument('-tb', '--target_back', type=str, help='Directory to load the target background.')
parser.add_argument('-b', '--back', type=str, default=None,
                    help='Captured background image. (only use for inference on videos with fixed camera')
parser.add_argument('-k', '--use_kpts', action='store_true',
                    help='Whether to load keypoints from input_dir')

args = parser.parse_args()


def to_tensor(pic):
    if len(pic.shape) >= 3:
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
    else:
        img = torch.from_numpy(pic)
        img = img.unsqueeze(0)
    # backward compatibility

    return 2 * (img.float().div(255)) - 1


# input data path
data_path = args.input_dir

if os.path.isdir(args.target_back):
    args.video = True
    print('Using video mode')
else:
    args.video = False
    print('Using image mode')
    # target background path
    back_img10 = cv2.imread(args.target_back);
    back_img10 = cv2.cvtColor(back_img10, cv2.COLOR_BGR2RGB);
    # Green-screen background
    back_img20 = np.zeros(back_img10.shape);
    back_img20[..., 0] = 120;
    back_img20[..., 1] = 255;
    back_img20[..., 2] = 155;

# input model
model_dir = os.path.join('Models', args.trained_model)
if os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 0:
    model_main_dir = 'Models/' + args.trained_model + '/'
    fo = glob.glob(model_main_dir + 'net*_epoch_*')
    model_name1 = fo[0]
else:
    models = glob.glob(f'Models/{args.trained_model}netG_epoch_*')
    models.sort(key=lambda k: int(k[:-4].split('_')[-1]))
    model_name1 = models[-1]
    assert os.path.exists(model_name1), model_name1
print(f'Using model"{args.trained_model}" with checkpoint at {model_name1}')
# initialize network
# fo = glob.glob(model_main_dir + 'netG_epoch_*')
# model_name1 = fo[0]
netM = ResnetConditionHR(input_nc=(3, 3, 1, 4), output_nc=4, n_blocks1=7, n_blocks2=3, kpts_nc=55 if args.use_kpts else None)
netM = nn.DataParallel(netM)
netM.load_state_dict(torch.load(model_name1))
netM.cuda();
netM.eval()
for m in netM.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats=False
cudnn.benchmark = True
reso = (512, 512)  # input reoslution to the network

# load captured background for video mode, fixed camera
if args.back is not None:
    bg_im0 = cv2.imread(args.back);
    bg_im0 = cv2.cvtColor(bg_im0, cv2.COLOR_BGR2RGB);

# Create a list of test images
test_imgs = [f for f in os.listdir(os.path.join(data_path, 'images')) if f.endswith('.png')]
test_imgs.sort()

# output directory
result_path = args.output_dir

if not os.path.exists(result_path):
    os.makedirs(result_path)

for i in range(0, len(test_imgs)):
    filename = test_imgs[i]
    # original image
    bgr_img = cv2.imread(os.path.join(data_path, 'images', filename));
    bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB);

    if args.back is None:
        # captured background image
        bg_im0 = cv2.imread(os.path.join(data_path, filename.replace('_img', '_back')));
        bg_im0 = cv2.cvtColor(bg_im0, cv2.COLOR_BGR2RGB);

    if args.use_kpts:
        kpts_path = os.path.join(data_path, 'keypoints', filename.replace('.png', '_keypoints.json'))
        kpts = read_kpts_json(kpts_path) if os.path.exists(kpts_path) else None
        # kpts_path = os.path.join(data_path, filename.replace('_img.png', '_keypoints.json'))
        # kpts = read_kpts_json(kpts_path) if (kpts is None) and os.path.exists(kpts_path) else kpts
        assert kpts is not None, kpts_path
    else:
        kpts = None

    # segmentation mask
    rcnn = cv2.imread(os.path.join(data_path, 'masks', filename), 0);

    if args.video:  # if video mode, load target background frames
        # target background path
        back_img10 = cv2.imread(os.path.join(args.target_back, filename.replace('_img.png', '.png')));
        back_img10 = cv2.cvtColor(back_img10, cv2.COLOR_BGR2RGB);
        # Green-screen background
        back_img20 = np.zeros(back_img10.shape);
        back_img20[..., 0] = 120;
        back_img20[..., 1] = 255;
        back_img20[..., 2] = 155;

        # create multiple frames with adjoining frames
        gap = 20
        multi_fr_w = np.zeros((bgr_img.shape[0], bgr_img.shape[1], 4))
        idx = [i - 2 * gap, i - gap, i + gap, i + 2 * gap]
        for t in range(0, 4):
            if idx[t] < 0:
                idx[t] = len(test_imgs) + idx[t]
            elif idx[t] >= len(test_imgs):
                idx[t] = idx[t] - len(test_imgs)

            file_tmp = test_imgs[idx[t]]
            bgr_img_mul = cv2.imread(os.path.join(data_path, 'images', file_tmp));
            multi_fr_w[..., t] = cv2.cvtColor(bgr_img_mul, cv2.COLOR_BGR2GRAY);

    else:
        ## create the multi-frame
        multi_fr_w = np.zeros((bgr_img.shape[0], bgr_img.shape[1], 4))
        multi_fr_w[..., 0] = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY);
        multi_fr_w[..., 1] = multi_fr_w[..., 0]
        multi_fr_w[..., 2] = multi_fr_w[..., 0]
        multi_fr_w[..., 3] = multi_fr_w[..., 0]

    # crop tightly
    bgr_img0 = bgr_img;
    if rcnn.sum() < 500 or (kpts is not None and len(kpts) == 0):
        for subdir in ('out', 'fg', 'compose', 'matte'):
            os.makedirs(subdir, exist_ok=True)
        cv2.imwrite(os.path.join(result_path, 'out', filename), bgr_img * 0)
        cv2.imwrite(os.path.join(result_path, 'fg', filename), bgr_img * 0)
        cv2.imwrite(os.path.join(result_path, 'compose', filename), back_img10)
        cv2.imwrite(os.path.join(result_path, 'matte', filename), back_img20[:, :, ::-1])
        continue
    bbox = get_bbox(rcnn, R=bgr_img0.shape[0], C=bgr_img0.shape[1])
    #bboxes = get_bboxes(rcnn, R=bgr_img0.shape[0], C=bgr_img0.shape[1], kpts=kpts)

    alphas, fgs = [], []
    for bbox in [bbox]:
        crop_list = [bgr_img, bg_im0, rcnn, back_img10, back_img20, multi_fr_w]
        assert not any(c is None for c in crop_list), [i for i in range(len(crop_list)) if crop_list[i] is None]
        crop_list = crop_images(crop_list, reso, bbox)
        bgr_img = crop_list[0];
        bg_im = crop_list[1];
        rcnn = crop_list[2];
        back_img1 = crop_list[3];
        back_img2 = crop_list[4];
        multi_fr = crop_list[5]
        if kpts is not None:
            kpts = apply_crop_kpts(kpts, bbox, reso)
            kpts = np.concatenate(tuple(kp[np.newaxis, ...] for kp in kpts), axis=0)
            skeleton_feats = gen_skeletons(kpts, reso[0], reso[1], stride=1, sigma=6., threshold=4., visdiff=False)
            skeleton_feats = to_tensor(skeleton_feats).unsqueeze(0)
            skeleton_feats = Variable(skeleton_feats.cuda())

        # process segmentation mask
        kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        rcnn = rcnn.astype(np.float32) / 255;
        rcnn[rcnn > 0.2] = 1;
        K = 25

        zero_id = np.nonzero(np.sum(rcnn, axis=1) == 0)
        del_id = zero_id[0][zero_id[0] > 250]
        if len(del_id) > 0:
            del_id = [del_id[0] - 2, del_id[0] - 1, *del_id]
            rcnn = np.delete(rcnn, del_id, 0)
        rcnn = cv2.copyMakeBorder(rcnn, 0, K + len(del_id), 0, 0, cv2.BORDER_REPLICATE)

        rcnn = cv2.erode(rcnn, kernel_er, iterations=10)
        rcnn = cv2.dilate(rcnn, kernel_dil, iterations=5)
        rcnn = cv2.GaussianBlur(rcnn.astype(np.float32), (31, 31), 0)
        rcnn = (255 * rcnn).astype(np.uint8)
        rcnn = np.delete(rcnn, range(reso[0], reso[0] + K), 0)

        # convert to torch
        img = torch.from_numpy(bgr_img.transpose((2, 0, 1))).unsqueeze(0);
        img = 2 * img.float().div(255) - 1
        bg = torch.from_numpy(bg_im.transpose((2, 0, 1))).unsqueeze(0);
        bg = 2 * bg.float().div(255) - 1
        rcnn_al = torch.from_numpy(rcnn).unsqueeze(0).unsqueeze(0);
        rcnn_al = 2 * rcnn_al.float().div(255) - 1
        multi_fr = torch.from_numpy(multi_fr.transpose((2, 0, 1))).unsqueeze(0);
        multi_fr = 2 * multi_fr.float().div(255) - 1

        with torch.no_grad():
            img, bg, rcnn_al, multi_fr = Variable(img.cuda()), Variable(bg.cuda()), Variable(rcnn_al.cuda()), Variable(
                multi_fr.cuda())
            input_im = torch.cat([img, bg, rcnn_al, multi_fr], dim=1)

            alpha_pred, fg_pred_tmp = netM(img, bg, rcnn_al, multi_fr, kp=skeleton_feats if kpts is not None else None)

            al_mask = (alpha_pred > 0.95).type(torch.cuda.FloatTensor)

            # for regions with alpha>0.95, simply use the image as fg
            fg_pred = img * al_mask + fg_pred_tmp * (1 - al_mask)

            alpha_out = to_image(alpha_pred[0, ...]);

            # refine alpha with connected component
            labels = label((alpha_out > 0.05).astype(int))
            try:
                assert (labels.max() != 0)
            except:
                continue
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
            #		alpha_out=alpha_out*largestCC

            alpha_out = (255 * alpha_out[..., 0]).astype(np.uint8)

            fg_out = to_image(fg_pred[0, ...]);
            fg_out = fg_out * np.expand_dims((alpha_out.astype(float) / 255 > 0.01).astype(float), axis=2);
            fg_out = (255 * fg_out).astype(np.uint8)

            # Uncrop
            R0 = bgr_img0.shape[0];
            C0 = bgr_img0.shape[1]
            alphas.append(uncrop(alpha_out, bbox, R0, C0))
            fgs.append(uncrop(fg_out, bbox, R0, C0))

    alpha_out0 = alphas[0] #sum(alphas)
    fg_out0 = fgs[0]
    for a, f in zip(alphas[1:], fgs[1:]):
        a = a / 255.
        fg_out0[a > 0.01] = f[a > 0.01]

    # compose
    back_img10 = cv2.resize(back_img10, (C0, R0));
    back_img20 = cv2.resize(back_img20, (C0, R0))
    comp_im_tr1 = composite4(fg_out0, back_img10, alpha_out0)
    comp_im_tr2 = composite4(fg_out0, back_img20, alpha_out0)

    for subdir in ('out', 'fg', 'compose', 'matte'):
        os.makedirs(os.path.join(result_path, subdir), exist_ok=True)
    cv2.imwrite(os.path.join(result_path, 'out', filename), alpha_out0)
    cv2.imwrite(os.path.join(result_path, 'fg', filename), cv2.cvtColor(fg_out0, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(result_path, 'compose', filename), cv2.cvtColor(comp_im_tr1, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(result_path, 'matte', filename),
                cv2.cvtColor(comp_im_tr2, cv2.COLOR_BGR2RGB))

    #print('Done: ' + str(i + 1) + '/' + str(len(test_imgs)))
