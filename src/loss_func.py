from torch import nn


class CustomLoss(nn.Module):
    """
    Custom loss function, a combination of dog and human classification losses with
    breed classification loss. The loss function uses two BCEWithLogitsLoss, one for
    human and dog each, and a CrossEntropyLoss for the dog breed classification.
            loss = dog_loss + human_loss + breed_loss

    Args:
        dog_human_labeller : Labeller used for human vs dog classification.
    """
    def __init__(self, dog_human_labeller):
        """
        Sets up the loss function with:
            1. dog_loss: BCEWithLogitsLoss for dog class.
            2. human_loss: BCEWithLogitsLoss for human class.
            3. breed_loss: CrossEntropyLoss for dog breeds.
        Sets up indexes for dog and human from the labeller provided.
        """
        super().__init__()
        self.dog_loss = nn.BCEWithLogitsLoss()
        self.human_loss = nn.BCEWithLogitsLoss()
        self.breed_loss = nn.CrossEntropyLoss()
        self.dog_human_labeller = dog_human_labeller
        self.human_idx = self.dog_human_labeller.label_dict['human']
        self.dog_idx = self.dog_human_labeller.label_dict['dog']

    def forward(self, outputs, t1, t2):
        """
        Calculates human_loss, dog_loss and breed_loss and return a combination
        of the three.

        Args:
            outputs: activations from the model
            t1: OneHot Encoded Targets for human vs dog classification.
            t2: Targets for dog breed classification.
        """
        h_loss = self.human_loss(outputs[:, [self.human_idx]],
                                 t1[:, [self.human_idx]])
        d_loss = self.dog_loss(outputs[:, [self.dog_idx]],
                               t1[:, [self.dog_idx]])
        # non dog images have a breed label -1.
        # mask to get only dogs.
        mask = t2 >= 0
        b_loss = self.breed_loss(outputs[mask, 2:], t2[mask])
        loss = h_loss + d_loss + b_loss
        return loss
