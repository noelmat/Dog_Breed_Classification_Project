from . import utils
from . import datasets
from . import labeller
from . import engine
from . import metrics
import os
from torch import optim


def get_dls(train_ds, valid_ds, bs):
    """
    Creates and returns dataloaders for train and valid datasets
    with the input batchsize.
    """
    train_dl = utils.get_dl(train_ds, bs=bs)
    valid_dl = utils.get_dl(valid_ds, bs=2*bs)
    return train_dl, valid_dl


def get_datasets(path, human_train, human_valid,
                 train_folder='train', valid_folder='valid', stats=None,
                 dog_human_label_func=labeller.human_or_dog,
                 breed_label_func=labeller.get_breed_label, size=224):
    """
    Creates and returns train and valid datasets.
    """
    breed_labeller = labeller.Labeller(breed_label_func)
    dog_human_labeller = labeller.Labeller(dog_human_label_func)
    train_ds = datasets.Dataset(
        path, human_train, train_folder, 
        breed_labeller=breed_labeller, dog_human_labeller=dog_human_labeller,
        stats=stats, tfms=datasets.get_tfms(size, distortion_scale=0.5),
        size=size)
    valid_ds = datasets.Dataset(
        path, human_valid, valid_folder,
        breed_labeller=breed_labeller, dog_human_labeller=dog_human_labeller,
        stats=stats, size=size)
    return train_ds, valid_ds


def get_device(use_cuda):
    """
    Return device string if use_cuda is true.
    """
    return 'cuda' if use_cuda else 'cpu'


def clear_output():
    """
    Clear terminal output.
    """
    os.system('clear')


def run(n_epochs, model, optimizer, criterion, dls, device, recorder,
        max_lr=0.1):
    train_dl = dls[0]
    valid_dl = dls[1]
    total_steps = n_epochs * len(train_dl)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1,
                                              total_steps=total_steps)

    for epoch in range(n_epochs):
        train_losses, train_acts, train_dog_human_targets, \
            train_breed_targets = engine.train_loop(train_dl, model,
                                                    optimizer, scheduler,
                                                    criterion, device)
        valid_losses, valid_acts, valid_dog_human_targets, \
            valid_breed_targets = engine.eval_loop(valid_dl, model, criterion,
                                                   device)

        train_metrics = metrics.get_metrics(
            train_losses, train_acts, train_dog_human_targets,
            train_breed_targets, criterion.dog_idx, criterion.human_idx)
        valid_metrics = metrics.get_metrics(
            valid_losses, valid_acts, valid_dog_human_targets,
            valid_breed_targets, criterion.dog_idx, criterion.human_idx)

        recorder.update(train_metrics, valid_metrics)
        output = metrics.get_tab_output(recorder, epoch)
        clear_output()
        print()
        print(output)
