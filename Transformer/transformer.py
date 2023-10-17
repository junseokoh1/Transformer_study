import torch
import torch.nn as nn

class SelfAttention(nn.Moduel)
def __init__(self, embed_size, heads):    # embed_size : 256, heads : 8 -> 각 dim : 32
    super(SelfAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = head
    self.head_dim = embed_size // heads         # 이게 32

    assert (self.head_dim * heads = embed_size), "Embed size needs to be div by heads"

    self.values = nn.Linear(self.haed_dim, self.head_dim, bias=False)
    self.keys = nn.Linear(self.haed_dim, self.head_dim, bias=False)
    self.queries = nn.Linear(self.haed_dim, self.head_dim, bias=False)
    self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

def forward(self, values, keys, query, mask):
    N = query.shape[0] # batch size
    value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]          # 입력 sequence 개수

    # Split the embedding into self.heads different pieces
    values = values.reshape(N, value_len, self.heads, self.head_dim)
    keys = keys.reshape(N, key_len, self.heads, self.head_dim)
    query = query.reshape(N, query_len, self.heads, self.head_dim)