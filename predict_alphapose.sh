source ~/.zshrc
conda activate alphapose

ALPHAPOSE_DIR=$HOME/AlphaPose

if [ $# -ne 1 ]; then
    echo "Usage: bash predict_alphapose.sh <path-to-dir>"
    exit 1
fi

d=`readlink -f $1`
if [ -f $d/`basename $d`.mp4 ]; then
    VID_PATH=$d/`basename $d`.mp4
elif [ -f $d/source.mp4 ]; then
    VID_PATH=$d/source.mp4
elif [ -d $d/images ]; then
    VID_PATH=$d/`basename $d`.mp4
    ffmpeg -i $d/images/%05d.png -pix_fmt yuv420p -c:v libx264 $VID_PATH || \
    ffmpeg -i $d/images/%04d.png -pix_fmt yuv420p -c:v libx264 $VID_PATH || \
    ffmpeg -i $d/images/%05d_img.png -pix_fmt yuv420p -c:v libx264 $VID_PATH || \
    ffmpeg -i $d/images/%04d_img.png -pix_fmt yuv420p -c:v libx264 $VID_PATH || \
    printf "Couldn't figure out formatting of images in $d/images folder\nConvert yourself to $d/`basename $d`.mp4" && \
    exit 1
else
    echo "Directory $d must contain either source.mp4/"`basename $d`.mp4" or folder 'images'"
    exit 1
fi


echo "--> Running AlphaPose on folder $d/"
CUR_DIR=$PWD
cd $ALPHAPOSE_DIR
./scripts/inference.sh configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml \
    pretrained_models/fast_421_res152_256x192.pth \
    $VID_PATH $d/alphapose_results || \
    echo "Failed to run alphapose.." && exit 1
cd $CUR_DIR

echo "--> Converting AlphaPose to OpenPose format"
if [ ! -d $d/images ]; then
    echo "To convert to keypoints, unpack video into frames into folder $d/images"
    exit 1
fi
python utils/format_alphapose_kpts.py $d/alphapose_results $d/keypoints --frames_dir $d/images