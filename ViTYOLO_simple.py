import torch.nn as nn
from vit_pytorch import ViT

class ViTYOLO(nn.Module):
    def __init__(self):
        super(ViTYOLO, self).__init__()
        self.vit = ViT(
                        image_size = 224,
                        patch_size = 16,
                        num_classes = 5880,
                        channels=3,
                        dim = 512,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 512,
                        dropout = 0.1,
                        emb_dropout = 0.1
                    )

    def forward(self, input):
        outputs = self.vit(input)
        return outputs.view(-1,14,14,30)