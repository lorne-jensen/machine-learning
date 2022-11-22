import random

import torch.cuda
from torch import nn

from src.eval import evaluate_input
from src.evaluate import evaluate_randomly
from src.model import Seq2Seq, Encoder, Decoder
from src.prepare_data import get_batches_from_dataset
from src.train import train_iterations, build_models
from src.train_model import train_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
dataset = 'squad1'

source, lengths, target, mask, max_target_len, voc, pairs = get_batches_from_dataset(dataset, batch_size)

randomize = random.choice(pairs)
print('random sentence {}'.format(randomize))

#print number of words
input_size = voc.num_words
output_size = voc.num_words
# print('Input : {} Output : {}'.format(input_size, output_size))

embed_size = 512
hidden_size = 512
num_layers = 1
num_iteration = 1000
learning_rate = 0.0001
teacher_forcing_ratio = 0.5
retrain = False

embedding = nn.Embedding(voc.num_words, embed_size)
if retrain:
    encoder = Encoder(input_size, hidden_size, embedding, num_layers)
    decoder = Decoder(hidden_size, voc.num_words, embedding, num_layers)

    train_iterations('boof', voc, pairs, encoder, decoder, learning_rate, embedding,
                     num_iteration, batch_size, 10, 10, dataset, teacher_forcing_ratio, False)

# reload the checkpoints:
encoder, decoder, _, _ = build_models(True)

evaluate_input(encoder, decoder, voc)
