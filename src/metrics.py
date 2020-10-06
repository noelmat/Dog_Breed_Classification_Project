from .imports import tabulate, torch


def get_accuracy(acts, targets, sigmoid=True, thresh=0.5):
    """
    Calculates accuracy for the given activations and targets.
    Args:
        acts: model activations.
        targets: targets.
        sigmoid: Boolean to apply sigmoid to the activations.
        thresh: threshold for calculating accuracy.
    Returns:
        accuracy: calculated accuracy.
    """
    if sigmoid:
        acts = torch.sigmoid(acts)
        acc = ((acts > thresh) == targets).float().mean()
    else:
        mask = targets > -1
        acts = acts[mask]
        targets = targets[mask]
        acc = (acts.argmax(dim=1) == targets).float().mean()
    return acc


def get_metrics(losses, acts, dog_or_human, breed_targets, dog_idx, human_idx):
    """
    Calculates the accuracy for dogs, humans and breed classification.
    Args:
        losses: list of losses
        acts  : activations from the model.
        dog_or_human: targets for dog vs human classification.
        breed_targets: targets for breed classification.
        dog_idx: numeric label for dog in dog_human_labeller.
        human_idx: numeric label for human in dog_human_labeller.
    Returns:
        A dictionary of loss, accuracy_dog, accuracy_human, accuracy_breed.
    """
    losses = torch.tensor(losses)
    acts = torch.cat(acts, dim=0)
    dog_or_human = torch.cat(dog_or_human, dim=0)
    breed_targets = torch.cat(breed_targets, dim=0)
    return {
        'loss': losses.mean(),
        'accuracy_dog': get_accuracy(acts[:, [dog_idx]],
                                     dog_or_human[:, [dog_idx]], thresh=0.9),
        'accuracy_human': get_accuracy(acts[:, [human_idx]],
                                       dog_or_human[:, [human_idx]]),
        'accuracy_breed': get_accuracy(acts[:, 2:],
                                       breed_targets, sigmoid=False)
    }


class Recorder:
    def __init__(self):
        self.train_acc_human, self.train_acc_dog = [], []
        self.train_acc_breed, self.train_loss = [], []
        self.valid_acc_human, self.valid_acc_dog = [], []
        self.valid_acc_breed, self.valid_loss = [], []

    def update(self, train_metrics, valid_metrics):
        self.train_loss.append(train_metrics['loss'])
        self.train_acc_human.append(train_metrics['accuracy_human'])
        self.train_acc_dog.append(train_metrics['accuracy_dog'])
        self.train_acc_breed.append(train_metrics['accuracy_breed'])
        self.valid_loss.append(valid_metrics['loss'])
        self.valid_acc_human.append(valid_metrics['accuracy_human'])
        self.valid_acc_dog.append(valid_metrics['accuracy_dog'])
        self.valid_acc_breed.append(valid_metrics['accuracy_breed'])


def get_tab_output(recorder, epoch):
    """
    Creates a table of metrics from the recorder.
    Args:
        recorder: Recorder used for training.
        epoch: current epoch.
    Return:
        tabulate table
    """
    output = []
    output.append(['Epoch', 'T_loss', 'v_loss', 'va_human',
                   'va_dog', 'ta_breed', 'va_breed'])
    for i in range(epoch+1):
        output.append([
            f"{i+1}",
            f"{recorder.train_loss[i].item():.6f}",
            f"{recorder.valid_loss[i].item():.6f}",
            f"{recorder.valid_acc_human[i].item():.6f}",
            f"{recorder.valid_acc_dog[i].item():.6f}",
            f"{recorder.train_acc_breed[i].item():.6f}",
            f"{recorder.valid_acc_breed[i].item():.6f}"
        ])
    return tabulate(output)
