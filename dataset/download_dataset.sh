wget https://huggingface.co/datasets/jens-lundell/cong/resolve/main/full_dataset.zip -O $1"full_dataset.zip"
unzip -n -q $1"full_dataset.zip" -d $1"/dataset/"
rm $1"full_dataset.zip"