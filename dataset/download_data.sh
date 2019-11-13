URL=https://www.dropbox.com/s/5g2vcl3xtbt9ljm/no_weight_processed_stitch.zip?dl=1
TAR_FILE=./dataset/omnipush.zip
wget -N $URL -O $TAR_FILE
unzip $TAR_FILE -d ./dataset/
mv ./dataset/no_weight_processed_stitch ./dataset/omnipush
rm $TAR_FILE
