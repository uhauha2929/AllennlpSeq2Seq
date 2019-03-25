# -*- coding: utf-8 -*-
# @Time    : 2019/3/14 13:44
# @Author  : uhauha2929
import itertools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data import Vocabulary

from allennlp.data.dataset_readers import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.models import SimpleSeq2Seq
from allennlp.modules import Embedding
from allennlp.modules.attention import DotProductAttention, BilinearAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.predictors import Seq2SeqPredictor
from allennlp.training import Trainer
from tqdm import tqdm

from metrics.rouge import eval_rouge

device = 1
torch.cuda.set_device(device)

train_file = 'data/train.txt'
valid_file = 'data/valid.txt'
test_file = 'data/test.txt'

vocab_dir = 'data/vocab/'
max_vocab_size = 50000
min_count = 5

batch_size = 64
INSTANCES_PER_EPOCH = batch_size * 5000  # 因为训练集太大, 一个epoch设定为5000个batch
num_epochs = 20

embedding_dim = 200
hidden_dim = 256
learning_rate = 1e-4
grad_clipping = 10

max_decoding_steps = 20
beam_size = 5

serialization_dir = 'checkpoints/seq2seq/'

reader = Seq2SeqDatasetReader(
    source_tokenizer=WordTokenizer(word_splitter=JustSpacesWordSplitter()),
    target_tokenizer=WordTokenizer(word_splitter=JustSpacesWordSplitter()),
    source_token_indexers={'tokens': SingleIdTokenIndexer(namespace='source_tokens')},
    target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')},
    lazy=True)

train_dataset = reader.read(train_file)
validation_dataset = reader.read(valid_file)

if os.path.exists(vocab_dir):
    vocab = Vocabulary.from_files(vocab_dir)
else:
    vocab = Vocabulary.from_instances(train_dataset,
                                      min_count={'source_tokens': min_count, 'target_tokens': min_count},
                                      max_vocab_size=max_vocab_size)
    vocab.save_to_files(vocab_dir)

en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('source_tokens'),
                         embedding_dim=embedding_dim)
source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})

encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True))
train_iterator = BucketIterator(batch_size=batch_size,
                                sorting_keys=[("source_tokens", "num_tokens")],
                                instances_per_epoch=INSTANCES_PER_EPOCH)

validation_iterator = BucketIterator(batch_size=batch_size,
                                     sorting_keys=[("source_tokens", "num_tokens")])

train_iterator.index_with(vocab)
validation_iterator.index_with(vocab)

model = SimpleSeq2Seq(vocab, source_embedder, encoder,
                      max_decoding_steps=max_decoding_steps,
                      target_embedding_dim=embedding_dim,
                      target_namespace='target_tokens',
                      attention=BilinearAttention(hidden_dim * 2, hidden_dim * 2),
                      beam_size=beam_size)


def train():
    model.cuda(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=train_iterator,
                      validation_iterator=validation_iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      num_epochs=num_epochs,
                      serialization_dir=serialization_dir,
                      grad_clipping=grad_clipping,
                      cuda_device=device)
    trainer.train()


def evaluate(model_path):
    model.load_state_dict(torch.load(model_path))
    model.cuda(device)
    reader.lazy = False
    test_dataset = reader.read(test_file)
    predictor = Seq2SeqPredictor(model, reader)

    def tokens2sent(tokens):
        return ' '.join([token
                         if isinstance(token, str) else token.text
                         for token in tokens])

    results, golds = [], []
    for instance in tqdm(test_dataset):
        results.append(tokens2sent(predictor.predict_instance(instance)['predicted_tokens']))
        golds.append(tokens2sent(instance.fields['target_tokens'].tokens[1:-1]))

    eval_rouge(results, golds)


if __name__ == '__main__':
    train()
    # evaluate('checkpoints/seq2seq/best.th')
