from torch import nn

from .components import Embeddings, Encoder


class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, config['vis'])

    def forward(self, input_ids):
        encoder_input = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(encoder_input)
        return encoded, attn_weights
