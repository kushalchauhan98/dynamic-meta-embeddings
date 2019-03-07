import sys
import argparse
from functools import partial
from torchtext import data, vocab
from models import SNLIModel
from utils import CustomVocab, FastTextCC
from embedders import SingleEmbedder, ConcatEmbedder, DMEmbedder, CDMEmbedder

data.Field.vocab_cls = CustomVocab
vocab.pretrained_aliases['crawl-300d-2M'] = partial(FastTextCC)

embedders = ['single', 'concat', 'dme', 'cdme']
tasks = ['snli']

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', 
                        help='Name of task', 
                        choices=tasks,
                        default='snli')
    parser.add_argument('--embedder', 
                        help='Type of embedder to use',
                        choices=embedders,
                        default='cdme')
    parser.add_argument('--proj_dim', 
                        help='Dimension to which the embeddings should be projected to', 
                        default=256, 
                        type=int)
    parser.add_argument('--emb_dropout', 
                        help='Dropout probablity for the Embedding layer', 
                        default=0.2, 
                        type=float)
    parser.add_argument('--vectors', 
                        nargs='*', 
                        help='Pretrained word embeddings to use',
                        choices= vocab.pretrained_aliases.keys(),
                        default=['glove.840B.300d', 'crawl-300d-2M'])
    parser.add_argument('--rnn_dim', 
                        help='No. of hidden units in the sentence encoder LSTM',
                        default=512,
                        type=int)
    parser.add_argument('--fc_dim', 
                        help='No. of hidden units in the Classifier',
                        default=1024,
                        type=int)
    parser.add_argument('--clf_dropout', 
                        help='Dropout probablity for the Classifier',
                        default=0.2,
                        type=float)
    parser.add_argument('--n_classes', 
                        help='No. of classes in dataset',
                        default=3,
                        type=int)
    parser.add_argument('--bs', 
                        help='Batch size',
                        default=64,
                        type=int)
    parser.add_argument('--lr', 
                        help='Learning Rate',
                        default=0.0004,
                        type=float)
    parser.add_argument('--epochs', 
                        help='No. of epochs',
                        default=50,
                        type=int)
    parser.add_argument('--device',
                        help='Device to use',
                        choices=['cuda', 'cpu'],
                        default='cuda')

    args = parser.parse_args()

    if args.task == 'snli':
        model_cls = SNLIModel

    print('Reading data...')
    model_cls.read_data()

    print('Loading Word Embeddings...')
    if args.embedder == 'single':
        emb = SingleEmbedder(model_cls=model_cls, vector=args.vectors[0], 
                             device=args.device)
    elif args.embedder == 'concat':
        emb = ConcatEmbedder(model_cls=model_cls, vectors=args.vectors,
                             dropout=args.emb_dropout, device=args.device)
    elif args.embedder == 'dme':
        emb = DMEmbedder(model_cls=model_cls, vectors=args.vectors,
                         proj_dim=args.proj_dim, dropout=args.emb_dropout,
                         device=args.device)
    else:
        emb = CDMEmbedder(model_cls=model_cls, vectors=args.vectors,
                          proj_dim=args.proj_dim, dropout=args.emb_dropout,
                          device=args.device)

    model = model_cls(embedder=emb, rnn_dim=args.rnn_dim, fc_dim=args.fc_dim, 
                      clf_dropout=args.clf_dropout, n_classes=args.n_classes,
                      device=args.device)
    
    print('Training...')
    model.train(bs=args.bs, epochs=args.epochs, lr=args.lr, device=args.device)
