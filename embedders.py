import torch
import torch.nn as nn
from overrides import overrides


class SingleEmbedder(nn.Module):
    def __init__(self, model_cls, vector='glove.840B.300d', device='cuda'):
        super(SingleEmbedder, self).__init__()
        model_cls.text_field.build_vocab(model_cls.train_data, vectors=vector)
        model_cls.label_field.build_vocab(model_cls.train_data)
        self.embed = nn.Embedding.from_pretrained(
            embeddings=model_cls.text_field.vocab.vectors).to(device)
        self.dim = self.embed.embedding_dim
             

    @overrides    
    def forward(self, batch):
        return self.embed(batch)



class ConcatEmbedder(nn.Module):
    def __init__(self, model_cls,
                 vectors=['glove.840B.300d', 'crawl-300d-2M.vec'], dropout=0.2,
                 device='cuda'):
        super(ConcatEmbedder, self).__init__()
        model_cls.text_field.build_vocab(model_cls.train_data, vectors=vectors)
        model_cls.label_field.build_vocab(model_cls.train_data)
        self.embed = nn.Embedding.from_pretrained(
            embeddings=model_cls.text_field.vocab.vectors).to(device)
        self.dropout = nn.Dropout(dropout)
        self.dim = self.embed.embedding_dim
                

    @overrides     
    def forward(self, batch):
        emb = self.embed(batch)
        out = self.dropout(emb)
        return out
        
        
        
class DMEmbedder(nn.Module):
    def __init__(self, model_cls,
                 vectors=['glove.840B.300d', 'crawl-300d-2M.vec'],
                 proj_dim=256, dropout=0.2, device='cuda'):
        super(DMEmbedder, self).__init__()
        self.model_cls = model_cls
        self.dim = proj_dim
        self.n_vectors = len(vectors)
        self.model_cls.text_field.build_vocab(model_cls.train_data,
                                              vectors=vectors)
        self.model_cls.label_field.build_vocab(model_cls.train_data)
        self.embed = nn.Embedding.from_pretrained(
            embeddings=model_cls.text_field.vocab.vectors).to(device)
        
        self.projectors = [None]*self.n_vectors
        for i, vector_cls in enumerate(self.model_cls.text_field.vocab.vectors_cls):
            self.projectors[i] = nn.Linear(vector_cls.dim, self.dim).to(device)
        
        self.attn_layer = nn.Linear(self.dim, 1).to(device)
        self.dropout = nn.Dropout(dropout)
            

    @overrides         
    def forward(self, batch):
        concat = self.embed(batch)
        start_pos = 0
        projected = [None]*self.n_vectors
        for i, vector_cls in enumerate(self.model_cls.text_field.vocab.vectors_cls):
            emb = concat[:, :, start_pos:start_pos+vector_cls.dim]
            start_pos += vector_cls.dim
            projected[i] = self.projectors[i](emb)
        projected_cat = torch.cat([p.unsqueeze(2) for p in projected], 2)
        attn = self.attn_layer(projected_cat)
        attn = torch.sigmoid(attn)
        attended = projected_cat * attn.expand_as(projected_cat)
        out = attended.sum(2)
        out = self.dropout(out)
        return out
        
        
        
class CDMEmbedder(nn.Module):
    def __init__(self, model_cls,
                 vectors=['glove.840B.300d', 'crawl-300d-2M.vec'],
                 proj_dim=256, dropout=0.2, device='cuda'):
        super(CDMEmbedder, self).__init__()
        self.model_cls = model_cls
        self.dim = proj_dim
        self.n_vectors = len(vectors)
        self.model_cls.text_field.build_vocab(model_cls.train_data,
                                              vectors=vectors)
        self.model_cls.label_field.build_vocab(model_cls.train_data)
        self.embed = nn.Embedding.from_pretrained(
            embeddings=model_cls.text_field.vocab.vectors).to(device)
        
        self.projectors = [None]*self.n_vectors
        for i, vector_cls in enumerate(self.model_cls.text_field.vocab.vectors_cls):
            self.projectors[i] = nn.Linear(vector_cls.dim, self.dim).to(device)
        
        self.attn_lstm = nn.LSTM(self.dim, 2, bidirectional=True).to(device)
        self.attn_linear = nn.Linear(2 * 2, 1).to(device)
        self.dropout = nn.Dropout(dropout)
            

    @overrides          
    def forward(self, batch):
        concat = self.embed(batch)
        start_pos = 0
        projected = [None]*self.n_vectors
        for i, vector_cls in enumerate(self.model_cls.text_field.vocab.vectors_cls):
            emb = concat[:, :, start_pos:start_pos+vector_cls.dim]
            start_pos += vector_cls.dim
            projected[i] = self.projectors[i](emb)
        projected_cat = torch.cat([p.unsqueeze(2) for p in projected], 2)
        attn_inp = projected_cat.view(projected_cat.shape[0], 
                                      projected_cat.shape[1]*projected_cat.shape[2], 
                                      -1)
        attn = self.attn_lstm(attn_inp)[0]
        attn = self.attn_linear(attn)
        attn = torch.sigmoid(attn)
        attn = attn.view(projected_cat.shape[0], projected_cat.shape[1], 
                         projected_cat.shape[2], 1)
        attended = projected_cat * attn.expand_as(projected_cat)
        out = attended.sum(2)
        out = self.dropout(out)
        return out
