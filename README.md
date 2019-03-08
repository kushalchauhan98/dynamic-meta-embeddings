# Dynamic Meta-Embeddings for Improved Sentence Representations

<img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" width="12%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



This repository contains my PyTorch implementation of the paper:



**Dynamic Meta-Embeddings for Improved Sentence Representations**<br>
Douwe Kiela, Changhan Wang and Kyunghyun Cho<br>
*Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*<br>
[[arXiv](https://arxiv.org/abs/1804.07983)] [[GitHub](https://github.com/facebookresearch/DME)]



### Abstract

While one of the first steps in many NLP systems is selecting what pre-trained word embeddings to use, we argue that such a step is better left for neural networks to figure out by themselves. To that end, we introduce dynamic meta-embeddings, a simple yet effective method for the supervised learning of embedding ensembles, which leads to state of-the-art performance within the same model class on a variety of tasks. We subsequently show how the technique can be used to shed new light on the usage of word embeddings in NLP systems.



### Usage

- Clone this repository and install the necessary requirements. Do:

  ```bash
  git clone https://github.com/kushalchauhan98/dynamic-meta-embeddings.git
  cd dynamic-meta-embeddings
  pip install -r requirements.txt
  ```

- Train the model. The training script will take care of downloading the datasets and pre-trained word embeddings. Do:

  ```bash
  python main.py [arguments...]
  ```

  The arguments are listed as follows:

  ```
    -h, --help            show this help message and exit
    --task {snli}         Name of task (default: snli)
    --embedder {single,concat,dme,cdme}
                          Type of embedder to use (default: cdme)
    --proj_dim PROJ_DIM   Dimension to which the embeddings should be projected to (default: 256)
    --emb_dropout EMB_DROPOUT
                          Dropout probablity for the Embedding layer (default: 0.2)
    --vectors {charngram.100d,fasttext.en.300d,fasttext.simple.300d,glove.42B.300d,glove.840B.300d,
              crawl-300d-2M,glove.twitter.27B.25d,glove.twitter.27B.50d,glove.twitter.27B.100d,
              glove.twitter.27B.200d,glove.6B.50d,glove.6B.100d,glove.6B.200d,glove.6B.300d}
                          Pretrained word embeddings to use (default:
                          ['glove.840B.300d', 'crawl-300d-2M'])
    --rnn_dim RNN_DIM     No. of hidden units in the sentence encoder LSTM (default: 512)
    --fc_dim FC_DIM       No. of hidden units in the Classifier (default: 1024)
    --clf_dropout CLF_DROPOUT
                          Dropout probablity for the Classifier (default: 0.2)
    --n_classes N_CLASSES
                          No. of classes in dataset (default: 3)
    --bs BS               Batch size (default: 64)
    --lr LR               Learning Rate (default: 0.0004)
    --epochs EPOCHS       No. of epochs (default: 50)
    --device {cuda,cpu}   Device to use (default: cuda)
  ```

  For example:

  ```bash
  python main.py --task snli \
  	--embedder dme \
  	--vectors glove.840B.300d crawl-300d-2M \
  	--emb_dropout 0 \
  	--clf_dropout 0 \
  	--lr 0.000003 \
  	--epochs 5
  ```

  Without any arguments, the script will train on the SNLI dataset using Contextual Dynamic Meta-Embeddings with the network architecture and parameters as mentioned in the paper.
