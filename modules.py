import torch.nn as nn
from overrides import overrides

class BiLSTM_Max(nn.Module):
    def __init__(self, inp_dim, rnn_dim, device):
        super(BiLSTM_Max, self).__init__()
        
        self.rnn = nn.LSTM(input_size=inp_dim, 
                           hidden_size=rnn_dim, bidirectional=True).to(device)
    
    
    @overrides
    def forward(self, inp):
        context_embed = self.rnn(inp)[0]
        return context_embed.max(dim=0)[0]
    
    
    
class Classifier(nn.Module):
    def __init__(self, rnn_dim, fc_dim, clf_dropout, n_classes, device):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4 * 2 * rnn_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(p=clf_dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(p=clf_dropout),
            nn.Linear(fc_dim, n_classes),
        ).to(device)
    
    
    @overrides
    def forward(self, inp):
        return self.classifier(inp)
