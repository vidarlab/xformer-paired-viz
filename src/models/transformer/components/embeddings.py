import torch
from torch import nn
from torch.nn.modules.utils import _pair


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config['patch_size'])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.a = config['hidden_size']
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=config['hidden_size'],
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config['hidden_size']))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config['hidden_size']))

        self.dropout = nn.Dropout(config['dropout_rate'])
        self.use_cls = config['global_feature_embedding'] == 'cls'

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        if not self.use_cls:
            embeddings = embeddings[:, 1:, :]
        return embeddings
