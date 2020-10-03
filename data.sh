mkdir input
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip -P input/
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip -P input/

cd input
unzip dogImages.zip
unzip lfw.zip

cd ..