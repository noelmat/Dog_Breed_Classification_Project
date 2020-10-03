from torch import optim
import argparse
from src.imports import Path
from src import utils
from src import train
from src import models
from src import loss_func
from src import metrics


parser = argparse.ArgumentParser()
parser.add_argument('--path_dogs', metavar='Path_Dogs', type=str,
                    required=True)
parser.add_argument('--path_human', metavar='Path_Human', type=str,
                    required=True)
parser.add_argument('--batch_size', metavar='bs', type=int, required=True)
parser.add_argument('--n_epochs', metavar='EPOCHS', type=int, required=True)
parser.add_argument('--img_size', metavar='imgsize', type=int, default=224)
parser.add_argument('--lr', metavar='LR', type=float, default=0.001)
parser.add_argument('--max_lr', metavar='MAX_LR_ONE_CYCLE', type=float,
                    default=0.001)
args = parser.parse_args()

path_dogs = Path(args.path_dogs)
path_human = Path(args.path_human)
train.clear_output()
train.display_message('+++++++++++++Creating Splits+++++++++++++')
human_train, human_valid, \
    human_test = utils.create_splits_human_dataset(path_human)
train.display_message(
    '+++++++++++++Calculating batch stats for normalizing+++++++++++++')
imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
train_ds, valid_ds = train.get_datasets(path_dogs, human_train, human_valid,
                                        stats=imagenet_stats,
                                        size=args.img_size)
bs = args.batch_size
dls = train.get_dls(train_ds, valid_ds, bs=bs)
train.display_message(
    '+++++++++++++Getting Model ready for training+++++++++++++')
model = models.ModelTransfer()
device = train.get_device()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = loss_func.CustomLoss(train_ds.dog_human_labeller)
recorder = metrics.Recorder()
n_epochs = args.n_epochs

train.run(n_epochs, model, optimizer, criterion, dls, device, recorder,
          max_lr=args.max_lr)
