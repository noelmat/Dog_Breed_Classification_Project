import torch


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
                                     dog_or_human[:, [dog_idx]]),
        'accuracy_human': get_accuracy(acts[:, [human_idx]],
                                       dog_or_human[:, [human_idx]]),
        'accuracy_breed': get_accuracy(acts[:, 2:],
                                       breed_targets, sigmoid=False)
    }
