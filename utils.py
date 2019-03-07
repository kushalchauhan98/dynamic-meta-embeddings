import os
from torchtext import vocab
from functools import partial
from overrides import overrides


class CustomVocab(vocab.Vocab):
    def __init__(self, *args, **kwargs):
        super(CustomVocab, self).__init__(*args, **kwargs)
    
    @overrides
    def load_vectors(self, vectors, **kwargs):
        self.vectors_cls = vectors
        vocab.Vocab.load_vectors(self, self.vectors_cls, **kwargs)
        
        

class FastTextCC(vocab.Vectors):
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip'

    def __init__(self, **kwargs):
        name = os.path.basename(FastTextCC.url).rstrip('.zip')
        super(FastTextCC, self).__init__(name, url=FastTextCC.url, **kwargs)
