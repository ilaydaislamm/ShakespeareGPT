import torch
import torch.nn.functional as F
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self,
                 embeding_dim : int = 384,
                 heads : int = 6,
                 dropout : float = 0.1,
                 ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embeding_dim)

        self.attention = nn.MultiheadAttention(embed_dim=embeding_dim,
                                               num_heads=heads,
                                               dropout=dropout,
                                               batch_first=True
                                               )

        self.ln2 = nn.LayerNorm(embeding_dim)

        self.fnn = nn.Sequential(
            nn.Linear(embeding_dim, embeding_dim * 4),
            nn.GELU(),
            nn.Linear(embeding_dim * 4, embeding_dim),
            nn.Dropout(dropout),
        )

    def forward(self,x : torch.Tensor,casual_mask : torch.Tensor):

        x_norm = self.ln1(x)

        attn_output, _ = self.attention(query = x_norm,
                                                   key = x_norm,
                                                   value = x_norm,
                                                    attn_mask = casual_mask,
                                                   need_weights = False)

        x = x + attn_output

        x = x + self.fnn(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 block_size : int,
                 num_layers : int = 6,
                 embeding_dim : int = 384,
                 heads : int = 6,
                 dropout : float = 0.1):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embeding_dim)

        self.position_embedding = nn.Embedding(num_embeddings=block_size,
                                               embedding_dim=embeding_dim)

        self.transformers = nn.ModuleList([
            TransformerBlock(embeding_dim=embeding_dim,
                             heads=heads,
                             dropout=dropout) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.final_layer_norm = nn.LayerNorm(embeding_dim)

        self.output_projection = nn.Linear(in_features=embeding_dim,
                                           out_features=vocab_size)

        self.loss = nn.CrossEntropyLoss()

        casual_mask = torch.triu(
            torch.ones([block_size, block_size],dtype=torch.bool),
            diagonal=1)

        self.register_buffer('casual_mask', casual_mask)

        self.apply(self._init_weights)


    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids : torch.Tensor,target_ids : torch.Tensor = None):

        batch_size,seq_length = input_ids.shape
        device = input_ids.device


        token_embedding = self.token_embedding(input_ids)

        positions = torch.arange(seq_length,device=device)

        position_embedding = self.position_embedding(positions)

        x = self.dropout(token_embedding + position_embedding)

        mask = self.casual_mask[:seq_length,:seq_length]

        for transformer in self.transformers:
            x = transformer(x,mask)

        x = self.final_layer_norm(x)

        logits = self.output_projection(x)

        if target_ids is not None:
            batch_size,seq_length,vocab_size = logits.shape
            logit_flat = logits.reshape(batch_size*seq_length,vocab_size)
            target_flat = target_ids.reshape(batch_size*seq_length)

            loss = self.loss(logit_flat,target_flat)

            return logits,loss

        return logits


    #GENERATES NEXT TOKEN ID
    def generate(self,input_ids : torch.Tensor,max_new_tokens : int,temperature : float = 0.5):
        for _ in range(max_new_tokens):

            if input_ids.shape[1] <= self.block_size:
                current_input = input_ids
            else:
                current_input = input_ids[:,-self.block_size:]

            logits= self.forward(current_input)

            last_logits = logits[:,-1,:]

            last_logits = last_logits / temperature

            probs = F.softmax(last_logits, dim=-1)

            next_token = torch.multinomial(probs, 1)

            input_ids = torch.cat((input_ids,next_token),dim=1)

        return input_ids







