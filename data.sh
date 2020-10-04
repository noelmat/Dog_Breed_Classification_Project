mkdir input
if which wget
then 
    wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip -P input/
    wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip -P input/
else
    cd input && { curl -O https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip ; cd -; }
    cd input && { curl -O https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip ; cd -; }
cd input
unzip dogImages.zip
unzip lfw.zip

cd ..