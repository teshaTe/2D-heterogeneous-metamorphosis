export directory=$1

cd $directory
ffmpeg -r 24 -f image2 -s 512x512 -i %d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p fin.mpg
