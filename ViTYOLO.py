from transformers import ViTModel
import torch.nn as nn

class ViTYOLO(nn.Module):
    def __init__(self):
        super(ViTYOLO
    , self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.head = nn.Sequential(
            nn.Linear(151296, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1470)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)['last_hidden_state']
        print(outputs.shape)
        outputs = outputs.view(1,-1)
        print(outputs.shape)
        outputs = self.head(outputs)
        print(outputs.shape)
        return outputs.view(-1,7,7,30)