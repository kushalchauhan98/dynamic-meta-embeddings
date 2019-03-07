import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from overrides import overrides
from torchtext import datasets, data
from modules import BiLSTM_Max, Classifier



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tr_acc = []
        self.val_acc = []
        self.tr_loss = []
        self.val_loss = []
        self.all_train_loss = []
        self.all_train_acc = []
        self.acc_this_epoch = {'train': [], 'val': []}
        self.loss_this_epoch = {'train': [], 'val': []}
        self.avg_loss = 0
        self.avg_acc = 0
    
    
    def train(self, bs=64, epochs=5, lr=0.0004, device='cuda'):
        self.create_batches(bs=bs, device=device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for i in range(epochs):
            with tqdm(self.get_batches(mode='train')) as t:
                for inputs, targets in t:
                    t.set_description('Epoch-%d of %d'%(i+1, epochs))
                    loss, logits = self.forward(inputs, targets)
                    loss.backward()
                    optimizer.step()
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == targets).float().mean()
                    
                    train_stats = self.on_batch_end(loss.item(), acc.item(),
                                                    'train')
                    t.set_postfix(**train_stats)
                t.set_postfix_str()
            
            for inputs, targets in self.get_batches(mode='val'):
                loss, logits = self.forward(inputs, targets)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == targets).float().mean()
                
                self.on_batch_end(loss.item(), acc.item(), 'val')
            
            self.on_epoch_end()
            
        test_acc = []
        for inputs, targets in self.get_batches(mode='test'):
            _, logits = self.forward(inputs, targets)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == targets).float().mean()
            test_acc.append(acc.item())
        
        print('Test Accuracy: %f'%np.mean(test_acc))
        torch.save(self.state_dict(), f'./{self.__class__.__name__}.pt')
        print(f'Model saved to ./{self.__class__.__name__}.pt')
                
    def on_batch_end(self, loss, acc, mode):
        self.loss_this_epoch[mode].append(loss)
        self.acc_this_epoch[mode].append(acc)
        self.all_train_loss.append(loss)
        self.all_train_acc.append(acc)

        if self.avg_loss == 0:
            self.avg_loss = loss
        else:
            self.avg_loss = 0.95*self.avg_loss + 0.05*loss
        if self.avg_acc == 0:
            self.avg_acc = acc
        else:
            self.avg_acc = 0.95*self.avg_acc + 0.05*acc
            
        return {'train_loss': self.avg_loss, 'train_accuracy': self.avg_acc}
    
    
    def on_epoch_end(self):
        self.tr_acc.append(np.mean(self.acc_this_epoch['train']))
        self.tr_loss.append(np.mean(self.loss_this_epoch['train']))
        self.val_loss.append(np.mean(self.loss_this_epoch['val']))
        self.val_acc.append(np.mean(self.acc_this_epoch['val']))
        
        self.acc_this_epoch = {'train': [], 'val': []}
        self.loss_this_epoch = {'train': [], 'val': []}
        self.avg_loss = 0
        self.avg_acc = 0
        
        print('Training Loss: %f'%self.tr_loss[-1])
        print('Training Accuracy: %f'%self.tr_acc[-1])
        print('Validation Loss: %f'%self.val_loss[-1])
        print('Validation Accuracy: %f'%self.val_acc[-1])
        print()
        
        
    def create_batches(self):
        raise NotImplementedError
    
    def get_batches(self):
        raise NotImplementedError

    @overrides
    def forward(self):
        raise NotImplementedError
        
        
        
class SNLIModel(Model):
    text_field = None
    label_field = None
    train_data = None
    val_data = None
    test_data = None
    
    def __init__(self, embedder, rnn_dim=512, fc_dim=1024, clf_dropout=0.2, 
                 n_classes=3, device='cuda'):
        super(SNLIModel, self).__init__()
        
        self.embedder = embedder
        self.sent_encoder = BiLSTM_Max(inp_dim=self.embedder.dim,
                                       rnn_dim=rnn_dim, device=device)
        self.classifier = Classifier(rnn_dim, fc_dim, clf_dropout, n_classes,
                                     device=device)
        self.criterion = nn.CrossEntropyLoss()
        
        
    @classmethod
    def read_data(cls):
        cls.text_field = data.Field(tokenize='spacy', lower=True)
        cls.label_field = data.Field(sequential=False)

        cls.train_data, cls.val_data, cls.test_data = datasets.SNLI.splits(
            cls.text_field, cls.label_field)
    
    
    def create_batches(self, bs=64, device='cuda'):
        cls = self.__class__
        self.train_iter, self.val_iter, self.test_iter =\
            data.BucketIterator.splits(
                (cls.train_data, cls.val_data, cls.test_data), 
                batch_size=bs,
                device=device)
        
        self.train_iter.create_batches()
        self.val_iter.create_batches()
        self.test_iter.create_batches()
    
    
    def get_batches(self, mode='train', device='cuda'):
        assert mode in ['train', 'val', 'test']
        cls = self.__class__
        if mode == 'train':
            iterator = self.train_iter
        elif mode == 'val':
            iterator = self.val_iter
        else:
            iterator = self.test_iter
            
        iterator.init_epoch()
            
        for batch in iterator.batches:
            premise = cls.text_field.process([e.premise for e in batch],
                                             device=device)
            hypothesis = cls.text_field.process([e.hypothesis for e in batch],
                                                device=device)
            labels = cls.label_field.process([e.label for e in batch],
                                             device=device) - 1
            yield ((premise, hypothesis), labels)
        

    @overrides
    def forward(self, inputs, labels=None):
        premise = self.embedder(inputs[0])
        hypothesis = self.embedder(inputs[1])
        enc_premise = self.sent_encoder(premise)
        enc_hypothesis = self.sent_encoder(hypothesis)
        combined = torch.cat([enc_premise, enc_hypothesis,
                              (enc_premise - enc_hypothesis).abs(),
                              enc_premise * enc_hypothesis],
                             1)
        logits = self.classifier(combined)
        if isinstance(labels, type(None)):
            return logits
        
        loss = self.criterion(logits, labels)
        return loss, logits
        
        
        

