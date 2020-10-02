from torchvision import transforms
from . import utils
from PIL import Image
import torch


class Dataset: 
    """
    Dataset for human vs dog classification along with dog breed
    classification
    Args:
        path_dogs: path to dog images dataset.
        human_paths: list of paths containing path to human face image.
        folder: folder name for dog images dataset. Expected values are 
                'train','valid' or 'test'
        breed_labeller: Labeller object for breed classification
        dog_human_labeller: Labeller object for dog vs human classification
        tfms : transforms to be applied on images.
        stats: batch stats for normalizing the images.
        size : size for resizing the images.
    """
    def __init__(self, path_dogs, human_paths, folder, breed_labeller,
                 dog_human_labeller, tfms=None, stats=None, size=224):
        self.files = utils.get_files(path_dogs/folder)
        self.breed_labeller = breed_labeller.get_labels(self.files)
        self.human_files = []
        [self.human_files.extend(utils._get_files(path))
         for path in human_paths]
        self.files += self.human_files
        self.dog_human_labeller = dog_human_labeller.get_labels(self.files)
        self.size = size
        if tfms is None:
            self.tfms = [
                transforms.Resize(size=(self.size, self.size)),
                transforms.ToTensor()
            ]
        else:
            self.tfms = tfms
        if stats is not None:
            self.tfms.append(transforms.Normalize(**stats))
        self.tfms = transforms.Compose(self.tfms)        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path)
        dog_human_label, lbl_str = self.dog_human_labeller.get_label(img_path)
        breed_label = -1
        if lbl_str == 'dog':
            breed_label, _ = self.breed_labeller.get_label(img_path)
        img = self.tfms(img)
        dog_human_target = torch.zeros(2)
        dog_human_target[dog_human_label] = 1
        breed_target = torch.tensor(breed_label, dtype=torch.long)

        return img, dog_human_target, breed_target
