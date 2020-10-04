from PIL import ImageFile
from src import models
from src import loss_func
from src import utils
from src.imports import Path, torch
from src.datasets import Dataset
from src.labeller import Labeller, human_or_dog, get_breed_label
from src import train, test
import argparse
ImageFile.LOAD_TRUNCATED_IMAGES = True


parser = argparse.ArgumentParser()
parser.add_argument('--path_dogs', metavar='Path_Dogs', type=str,
                    required=True)
parser.add_argument('--path_human', metavar='Path_Human', type=str,
                    required=True)
parser.add_argument('--batch_size', metavar='bs', type=int, required=True)
parser.add_argument('--img_size', metavar='imgsize', type=int, default=224)
parser.add_argument('--model_path',type=str, required=True)
args = parser.parse_args()

path_dogs = Path(args.path_dogs)
path_human = Path(args.path_human)
train.clear_output()
train.display_message('+++++++++++++Creating Splits+++++++++++++')
human_train, human_valid, \
    human_test = utils.create_splits_human_dataset(path_human)


learn_dict = torch.load(args.model_path, map_location='cpu')
model = models.ModelTransfer(pretrained=False)
model.load_state_dict(learn_dict['model'])
model.eval()
device = train.get_device()
model.to(device)
dog_human_labeller = learn_dict['dog_human_labeller']
breed_labeller = learn_dict['breed_labeller']
imagenet_stats = learn_dict['model_normalization_stats']
criterion = loss_func.CustomLoss(dog_human_labeller)
test_ds = Dataset(path_dogs, human_test, 'test', breed_labeller=breed_labeller, dog_human_labeller=dog_human_labeller, stats=imagenet_stats,size=args.img_size)
test.test(model, test_ds, criterion, device)
