import random

import torch.cuda

from src.evaluate import evaluate_randomly
from src.model import Seq2Seq
from src.prepare_data import get_batches_from_dataset
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

embed_size = 256
hidden_size = 512
num_layers = 1
num_iteration = 100000

model = Seq2Seq(encoder_input_size=input_size,
                encoder_hidden_size=hidden_size,
                encoder_embedding_size=embed_size,
                decoder_hidden_size=hidden_size,
                decoder_output_size=output_size,
                decoder_embedding_size=embed_size,
                num_layers=num_layers).to(device)

model = train_model(model, source, target, pairs, num_iteration)

evaluate_randomly(model, source, target, pairs)
