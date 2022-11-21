import random

import torch.cuda
from torch import nn

from src.evaluate import evaluate_randomly
from src.model import Seq2Seq, Encoder, Decoder
from src.prepare_data import get_batches_from_dataset
from src.train import train_iterations
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

embedding = nn.Embedding(voc.num_words, embed_size)

encoder = Encoder(input_size, hidden_size, embedding, num_layers)
decoder = Decoder(hidden_size, voc.num_words, embedding, num_layers)

model = Seq2Seq(encoder, decoder)
# model = Seq2Seq(encoder_input_size=input_size,
#                 encoder_hidden_size=hidden_size,
#                 encoder_embedding_size=embed_size,
#                 decoder_hidden_size=hidden_size,
#                 decoder_output_size=output_size,
#                 decoder_embedding_size=embed_size,
#                 num_layers=num_layers).to(device)

# model = train_model(model, voc, pairs, batch_size, num_iteration)
#
# evaluate_randomly(model, source, target, pairs)

train_iterations('boof', voc, pairs, encoder, decoder, learning_rate, embedding,
                 num_iteration, batch_size, 10, 10, dataset, teacher_forcing_ratio, False)
