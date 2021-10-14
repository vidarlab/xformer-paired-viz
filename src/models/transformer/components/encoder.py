from torch import nn
import copy

from .encoder_block import EncoderBlock


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        for _ in range(config['num_layers']):
            layer = EncoderBlock(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_state):
        attn_weights = []
        for layer_block in self.layer:
            hidden_state, weights = layer_block(hidden_state)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_state)
        return encoded, attn_weights