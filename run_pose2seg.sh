source ~/.zshrc
conda activate pose2seg

POSE2SEG_DIR=$HOME/Pose2Seg

if [ $# -eq 0 ]; then
    echo "Usage: bash run_pose2seg.sh <path-to-dir> [depth=1]"
    exit 1
fi

d=$(readlink -f $1)
depth=1
if [ $# -ge 2 ]; then
    depth=$2
    echo "Depth is set to $depth"
fi

if [ $depth -eq 1 ] && [ $(ls -1 "$d"/images | wc -l) -ne $(ls -1 "$d"/keypoints | wc -l) ]; then
    echo "Amount of files in $d/images & $d/keypoints must match. "\
         "Currently it's: "$(ls -1 $d/images | wc -l)" vs. "$(ls -1 $d/keypoints | wc -l)
    exit 1
fi

echo "--> Running Pose2Seg on folder $d/"
CUR_DIR=$PWD
cd "$POSE2SEG_DIR" || exit 1
python infer.py pose2seg_release.pkl $d --depth $depth
if [ $? -ne 0 ]; then
  echo "Failed to run pose2seg.."
  exit 1
fi
cd "$CUR_DIR" || exit 1
