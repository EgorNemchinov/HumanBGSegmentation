set -e
WORKDIR=$HOME/workdir

VID_PATH=$1

if [ $# -eq 0 ]; then
    echo "Usage: bash main.sh <VID_PATH> [DIR_NAME]"
fi
if [ ! -e $VID_PATH ]; then
    echo "Failed as path $VID_PATH doesn't exist"
    exit 1
fi

NAME=$(date +'%Y-%m-%d_%H%M%S')
if [ $# -ge 2 ]; then
    NAME=$2
fi
DIR=$WORKDIR/$NAME
mkdir -p $DIR || echo "$DIR already exists"

if [ $(ls -1 $DIR/images | wc -l) -eq 0 ]; then
  echo "--> Extracting frames"
  mkdir $DIR/images || echo "$DIR/images already exists"
  ffmpeg -i $VID_PATH $DIR/images/%04d.png
fi

if [ $(ls -1 $DIR/images | wc -l) -ne $(ls -1 $DIR/keypoints | wc -l) ]; then
  echo "--> Running AlphaPose"
  zsh predict_alphapose.sh $DIR
fi

if [ $(ls -1 $DIR/images | wc -l) -ne $(ls -1 $DIR/masks | wc -l) ]; then
  echo "--> Running Pose2Seg"
  zsh run_pose2seg.sh $DIR
fi

if [ ! -e $DIR/$NAME.png ]; then
  echo "--> Extracting background"
  SKIP_AMOUNT=0
  if [ $(ls -1 $DIR/images | wc -l) -ge 1000 ]; then
      SKIP_AMOUNT=100
  fi
  python extract_bg.py $DIR/images $DIR/$NAME.png --skip_first $SKIP_AMOUNT --skip_last $SKIP_AMOUNT
fi

if [ ! -e $DIR/results/out.mp4 ] || [ ! -e $DIR/results/fg.mp4 ]; then
  echo "--> Run matting"
  zsh run_bg_matting.sh $DIR results Real_fixed_mix_kpts_div_big
fi

echo "--> Archive videos"
zip $DIR/results.zip -j $DIR/results/out.mp4 $DIR/results/fg.mp4

echo "--------------------"
echo "Finished! Result archive is at $DIR/results.zip"
