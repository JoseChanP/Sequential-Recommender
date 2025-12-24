import torch
import torch.nn as nn

class SASRecModel(nn.Module):
    def __init__(self, num_movies, num_genres, num_years, embed_dim=256, max_len=200, num_heads=4, num_layers=3, dropout=0.1):
        super(SASRecModel, self).__init__()
        self.movie_embedding = nn.Embedding(num_movies, embed_dim, padding_idx=0)
        self.genre_embedding = nn.Embedding(num_genres, embed_dim, padding_idx=0)
        self.year_embedding = nn.Embedding(num_years, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            dim_feedforward=embed_dim*4, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embed_dim, num_movies)
        self.max_len = max_len
        self.embed_dim = embed_dim
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_seq, input_genres, input_years):
        batch_size, seq_len = input_seq.size()
        
        movie_embeds = self.movie_embedding(input_seq)
        genre_embeds = self.genre_embedding(input_genres)
        year_embeds = self.year_embedding(input_years)
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_seq.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        x = movie_embeds + genre_embeds + year_embeds + position_embeds
        x = self.dropout(x)
        
        attn_mask = torch.triu(torch.ones((seq_len, seq_len), device=input_seq.device) * float('-inf'), diagonal=1).bool()
        
        x = self.transformer_encoder(x, mask=attn_mask)
        last_hidden = x[:, -1, :]
        logits = self.output_layer(last_hidden)
        
        return logits