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
args = parser.parse_args()
path_dogs = Path(args.path_dogs)
path_human = Path(args.path_human)
train.clear_output()
train.display_message('+++++++++++++Creating Splits+++++++++++++')
human_train, human_valid, \
    human_test = utils.create_splits_human_dataset(path_human)
train.display_message(
    '+++++++++++++Calculating batch stats for normalizing+++++++++++++')
train_ds, _ = train.get_datasets(path_dogs, human_train, human_valid)
batch_stat = utils.get_batch_stat(train_ds)
train.display_message('+++++++++++++Creating DataLoaders+++++++++++++')
train_ds, valid_ds = train.get_datasets(path_dogs, human_train, human_valid,
                                        stats=batch_stat)
dls = train.get_dls(train_ds, valid_ds, bs=args.batch_size)
train.display_message(
    '+++++++++++++Getting Model ready for training+++++++++++++')
model = models.ModelScratch()
optimizer = optim.Adam(model.parameters())
criterion = loss_func.CustomLoss(train_ds.dog_human_labeller)
recorder = metrics.Recorder()
device = train.get_device()
model.to(device)
n_epochs = args.n_epochs
train.run(n_epochs, model, optimizer, criterion, dls, device, recorder)
