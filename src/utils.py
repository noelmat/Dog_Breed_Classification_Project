from .imports import mimetypes, Path, plt, torch, os

image_extension = [k for k, v in mimetypes.types_map.items() if 'image' in v]


def _get_files(path, extensions=image_extension):
    """
    Get files from the given path that matches the extensions.
    Args:
        path: path to a directory.
        extensions: list of extensions.
    """
    path = Path(path)
    files = [Path(file.path) for file in os.scandir(path) if Path(file).suffix
             in extensions]
    return files


def get_dirs(path):
    """
    Get all the directories in the provided path.
    Args:
        path: path to a directory.
    """
    path = Path(path)
    dirs = [file for file in os.scandir(path)]
    dirs = [Path(dir.path) for dir in dirs if dir.is_dir()]
    return dirs


def get_files(path, extensions=image_extension):
    """
    Get files from the given path that matches the extensions.
    Args:
        path: path to a directory.
        extensions: list of extensions.
    """
    dirs = get_dirs(path)
    files = []
    for dir in dirs:
        files.extend(_get_files(dir, extensions))
    return files


def get_dl(ds, bs=8, shuffle=True, num_workers=4):
    """
    Creates a torch.utils.data.DataLoader for the input dataset.
    Args:
        ds: dataset.
        bs: batch_size. default=8.
        shuffle: flag to indicate shuffle.
        num_workers: number of threads for the dataloader default=4.
    """
    return torch.utils.data.DataLoader(ds,
                                       batch_size=bs,
                                       shuffle=shuffle,
                                       num_workers=num_workers)


def get_one_batch(dl):
    """
    Gets the first batch from the dataloader.
    Args:
        dl: Dataloader
    """
    return next(iter(dl))


def denormalize(x, mean, std):
    """
    Denormalizes the data with the provided mean and std.
    Args:
        x: Batch of data.
        mean: mean used for normalizing.
        std: standard deviation used for normalizing.
    """
    means = torch.tensor(mean).T
    stds = torch.tensor(std).T
    return x * stds[None, :, None, None] + means[None, :, None, None]


def show_batch(dl, rows=3, cols=3, **kwargs):
    """
    Shows a batch in the jupyter notebook.
    Args:
        dl: Dataloader
        rows: number of rows to be displayed
        cols: number of cols to be displayed.
        stats: Stats for denormalizing (required)
    """
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8),
                             constrained_layout=True)
    dog_human_labeller = dl.dataset.dog_human_labeller
    breed_labeller = dl.dataset.breed_labeller
    dl = get_dl(dl.dataset, bs=rows*cols)
    x, y1, y2 = get_one_batch(dl)
    x = denormalize(x, **kwargs)
    axes = axes.flatten()
    for img, t1, t2, ax in zip(x, y1, y2, axes):
        img = img.permute(1, 2, 0)
        ax.imshow(img)
        label = dog_human_labeller.label_lookup[int(t1.argmax())]
        ax.set_title(f'{label} {breed_labeller.label_lookup[int(t2.item())] if label=="dog" else "" }')
        ax.axis('off')
    plt.show()


def create_splits_human_dataset(path):
    path = Path(path)
    dirs = get_dirs(path)
    human_train = dirs[:-2000]
    human_valid = dirs[-2000:-1000]
    human_test = dirs[-1000:]
    return human_train, human_valid, human_test


def get_batch_stat(ds):
    dl = get_dl(ds, bs=64, shuffle=False)
    x, y1, y2 = get_one_batch(dl)
    return {
        'mean': x.mean(dim=[0, 2, 3]).numpy().tolist(),
        'std': x.std(dim=[0, 2, 3]).numpy().tolist()
    }


def save_model(model, model_name, breed_labeller, dog_human_labeller,
               batch_stat):
    learner = {
        'model': model.state_dict(),
        'breed_labeller': breed_labeller,
        'dog_human_labeller': dog_human_labeller, 
        'model_normalization_stats': batch_stat,
    }
    torch.save(learner, model_name)
