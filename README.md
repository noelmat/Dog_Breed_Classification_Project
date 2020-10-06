# Machine Learning Nanodegree Capstone Project

## Dog Breed Classification

Computer vision is a field of artificial intelligence that trains computers to interpret and
understand the visual world. The idea of building intelligent machines that can understand and
interpret visual world existed from 1960s. It all started with Hubel and Wiesel's experiment to
understand how a cat's neuron responded to visual stimuli. MIT started a Summer Vision Project
in the summer of 1966, with a goal to solve the vision problem in a single summer. Though it was
not a success, it led to people getting interested and further pushed the field. Today we have not
solved the problem yet, but we have surely come far. Far enough be used with medical imaging,
object detection, image captioning and so on. The field was revolutionized with the introduction
of Deep Learning and the hardware support provided by Nvidia GPUs and was demonstrated by
Alexnet winning the Imagenet Competition in 2012.
My whole journey with Machine Learning started with being amused by what deep learning has
made possible in the fields of Computer Vision. I will be using Deep learning to estimate dog
breeds which is not a trivial problem to solve due to the interclass variance and the intraclass
variance between the dog breeds.

**The Project notebook can be accessed** [here](Final_Notebook.ipynb)


**The Project report can be accessed** [here](report.pdf)

**The Project proposal can be accessed**[here](capstone_proposal.pdf)

### Problem Statement
The goal of the project is to determine which dog breed a given image contains. I will use
Convolutional Neural Network (CNN) to classify the images by dog breeds. If the image contains
a dog, the output will be the identified dog breed. If the image contains a human then output will
the resembling dog breed. If the image contains neither of the two then it will report an error. I
will be using a custom loss function, a combination of two Binary Cross Entropy loss dog_loss &
face_loss and a Cross Entropy loss for breed_loss. The breed_loss will be conditional; i.e., the
breed_loss will only be calculated in case the image is actually a dog.
Furthermore, One Cycle Scheduling with cosine annealing for scheduling learning rates and
momentums will be used.



## Steps to setup the project.

### Train Models with this project:

1. Download and extract the dataset using `data.sh`.
2. Create a python virtual env and install the dependencies using `requirements-dev.txt`
3. To train the model from scratch, run the train_model_scratch.py script. This script accepts the following command line arguments:
    * `--path_dogs` : path to the [dogImages](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) folder. If you have used data.sh to setup the data, path_dogs will be `input/dogImages`.
    * `--path_human` : path to [human](http://vis-www.cs.umass.edu/lfw/lfw.tgz) dataset. data.sh will set up the dataset in `input/lfw`.
    * `--n_epoch` : number of epochs to train the model. 
    * `--batch_size` : batch size for train set.
    * `--lr` : The learning rate defaults to 0.1
    * `--max_lr` : The `max_lr` defaults to 0.1
    * `--img_size` : The size of input to the model.
    Example command :
    ```
    python scripts/train_model_scratch.py --path_dogs input/dogImages --path_human input/lfw --batch_size 64 --n_epochs 1 --img_size 64
    ```

4. The train script saves the model with the name: `model_scratch_{n_epochs}_{breed_accuracy}`
5. To train the model using transfer learning, run the train_model_transfer.py script.  This script accepts the following command line arguments:
    * `--path_dogs` : path to the [dogImages](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) folder. If you have used data.sh to setup the data, path_dogs will be `input/dogImages`.
    * `--path_human` : path to [human](http://vis-www.cs.umass.edu/lfw/lfw.tgz) dataset. data.sh will set up the dataset in `input/lfw`.
    * `--n_epoch` : number of epochs to train the model. 
    * `--batch_size` : batch size for train set.
    * `--lr` : The learning rate defaults to 0.001
    * `--max_lr` : The `max_lr` defaults to 0.001
    * `--img_size` : The size of input to the model.

6. Once the model is trained, the test scripts can be used similarly with commandline arguments. It accepts `--model_path` as a path to the model file.

For model transfer, execute the follow command  example command:
```
python scripts/test_model_transfer.py --path_dogs input/dogImages --path_human input/lfw --batch_size 64 --model_path model_transfer_0.83
```
