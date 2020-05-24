source ~/.zshrc
conda activate back-matting
set -e

BG_MATTING_DIR=$(readlink -f Background-Matting/)

if [ $# -le 1 ]; then
    echo "Usage: bash run_bg_matting.sh <path-to-input-dir> <name-of-out-dir>"
    exit 1
fi

d=$(readlink -f $1)
if [ $(ls -1 "$d" | grep -c "_img.png") -ne $(ls -1 "$d" | grep -c "_masksDL.png") ]; then
    echo "Amount of _img.png files & _masksDL.png in $d must match. "\
         "Currently it's: "$(ls -1 "$d" | grep -c "_img.png")" vs. "$(ls -1 "$d" | grep -c "_masksDL.png")
    exit 1
fi

out_dir=$(readlink -f $d/../$2)
mkdir $out_dir && echo "----> Created $out_dir" || echo "----> $out_dir already exists"
par_name=$(readlink -f "$d/.." | xargs basename)

echo "--> Running matting in docker"
docker run --gpus all -v $BG_MATTING_DIR:/bg-matting -v $(readlink -f "$d"/..):/data back-mat \
      bash -c "cd /bg-matting/; export CUDA_VISIBLE_DEVICES=0,1; python test_background-matting_image.py -m real-fixed-cam -i /data/$(basename "$d") -o /data/$(basename "$out_dir") -b /data/$par_name.png  -tb /data/$par_name.png"

echo "--> Organize into folders & create vids"
for n in out matte compose fg; do
    mkdir "$out_dir"/"$n" && echo "----> Created " "$out_dir"/"$n" || echo "----> " "$out_dir"/"$n" " already exists";
    for i in $(ls -1 $out_dir | grep "$n.png"); do
        sudo mv "$out_dir"/$i "$out_dir"/$n/"$(echo $i | sed "s/$n/img/g")"
    done
    ffmpeg -i "$out_dir"/$n/%04d_img.png -pix_fmt yuv420p -c:v libx264 "$out_dir"/fg.mp4 -loglevel panic -y || \
    ffmpeg -i "$out_dir"/$n/%05d_img.png -pix_fmt yuv420p -c:v libx264 "$out_dir"/fg.mp4 -loglevel panic -y
done

# TODO: add variable for checkpoint