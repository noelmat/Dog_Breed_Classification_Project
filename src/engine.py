from .imports import torch, tqdm


def train_loop(dl, model, optimizer, scheduler, criterion, device):
    """
    Loop for training the model.
    Args:
        dl: train dataloader
        model: model for training. The model should be on
                the required device.
        optimizer: Pytorch Optimizer.
        scheduler: Scheduler
        criterion: Loss
        device: device for training. Expected 'cpu' or 'cuda'.

    Returns:
        losses: list of mean loss for each batch.
        outputs: list of model activation for each batch.
        dog_or_human: list of targets for dog or human for each batch
        breed_targets: list of targets for dog breeds for each batch
    """
    model.train()  # Put model in train mode.
    losses = []
    outputs = []
    dog_or_human = []
    breed_targets = []
    for x, y1, y2 in tqdm(dl, total=len(dl), position=0, leave=True):
        x = x.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        preds = model(x)
        loss = criterion(preds, y1, y2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
        outputs.append(preds.detach().cpu())
        dog_or_human.append(y1.detach().cpu())
        breed_targets.append(y2.detach().cpu())
    return losses, outputs, dog_or_human, breed_targets


def eval_loop(dl, model, criterion, device):
    """
    Loop for evaluting the model.
    Args:
        dl: dataloader for evaluation.
        model: model for training. The model should be on
                the required device.
        criterion: Loss
        device: device for training. Expected 'cpu' or 'cuda'.

    Returns:
        losses: list of mean loss for each batch.
        outputs: list of model activation for each batch.
        dog_or_human: list of targets for dog or human for each batch
        breed_targets: list of targets for dog breeds for each batch
    """
    losses = []
    outputs = []
    dog_or_human = []
    breed_targets = []
    model.eval()  # Put model in eval mode.
    with torch.no_grad():  # Allows for double batch size
        for x, y1, y2 in tqdm(dl, total=len(dl), position=0, leave=True):
            x = x.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            preds = model(x)
            loss = criterion(preds, y1, y2)
            losses.append(loss.item())
            outputs.append(preds.detach().cpu())
            dog_or_human.append(y1.detach().cpu())
            breed_targets.append(y2.detach().cpu())
    return losses, outputs, dog_or_human, breed_targets
