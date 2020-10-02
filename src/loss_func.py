from torch import nn

class CustomLoss(nn.Module):
    def __init__(self, breed_labeller, dog_human_labeller):
        super().__init__()
        self.dog_loss = nn.BCEWithLogitsLoss()
        self.human_loss = nn.BCEWithLogitsLoss()
        self.breed_loss = nn.CrossEntropyLoss()
        self.breed_labeller = breed_labeller
        self.dog_human_labeller = dog_human_labeller
        self.human_idx = self.dog_human_labeller.label_dict['human']
        self.dog_idx = self.dog_human_labeller.label_dict['dog']
      
    def forward(self, outputs, t1, t2):
        h_loss = self.human_loss(outputs[:,[self.human_idx]], t1[:,[self.human_idx]])
        d_loss = self.dog_loss(outputs[:, [self.dog_idx]], t1[:, [self.dog_idx]])
        # non dog images have a breed label -1.
        # mask to get only dogs.
        mask = t2>=0  
        b_loss = self.breed_loss(outputs[t2>=0, 2:], t2[t2>=0])
        loss = h_loss + d_loss + b_loss
        return loss       