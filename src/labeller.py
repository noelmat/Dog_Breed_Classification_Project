class Labeller:
    """
    Takes a labelling function and creates a labeller.
    Stores -
        label_dict: 'str' label to int map
        label_lookup: reverse map of label_dict
    Args:
        label_func: A function to label the data.
    """
    def __init__(self, label_func):
        self.label_dict = None
        self.label_func = label_func
        self.label_lookup = None

    def get_labels(self, files):
        if self.label_dict is not None:
            return self
        labels = set()
        for file in files:
            labels.add(self.label_func(file))
        self.label_dict = {label: i for i, label in enumerate(labels)}
        self.label_lookup = {v: k for k, v in self.label_dict.items()}
        return self

    def get_label(self, img_path):
        lbl_class = self.label_func(img_path)
        return self.label_dict[lbl_class], lbl_class


def get_breed_label(path):
    """
    Calculates the label for the breed from the given path.
    Args:
        path: path to the image.
    """
    return ' '.join(path.name.split('_')[:-1])


def human_or_dog(path):
    """
    Calculates the label from the given path. Labels are 'human'
    or 'dog'.
    Args:
        path: path to the image.
    """
    return 'human' if 'lfw' in str(path).split('/') else 'dog'
