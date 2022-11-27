import random

import torch.cuda
from torch import nn, optim

from src.eval import evaluate_input
# from src.evaluate import evaluate_randomly
from src.model import Encoder, Decoder
from src.prepare_data import get_vocab_and_sentence_pairs
from src.train import train_iterations, build_models
# from src.train_model import train_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
dataset = 'squad1'

voc, pairs_train, pairs_valid = get_vocab_and_sentence_pairs(dataset)

randomize = random.choice(pairs_train)
print('random sentence {}'.format(randomize))

#print number of words
input_size = voc.num_words
output_size = voc.num_words
# print('Input : {} Output : {}'.format(input_size, output_size))

embed_size = 256
hidden_size = 512
num_layers = 1
num_iteration = 50
num_epochs = 2
learning_rate = 0.0001
teacher_forcing_ratio = 0.5
retrain = True

embedding = nn.Embedding(voc.num_words, embed_size)

model_name = 'boof'

if retrain:
    encoder = Encoder(input_size, hidden_size, embedding, num_layers)
    decoder = Decoder(hidden_size, voc.num_words, embedding, num_layers)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * 5.0)
    scheduler_encoder = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', patience=5,
                                                                   factor=0.5, min_lr=0.0000001, verbose=True)
    scheduler_decoder = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', patience=5,
                                                                   factor=0.5, min_lr=0.0000001, verbose=True)

    for epoch in range(num_epochs):
        print('Beginning training for epoch {}'.format(epoch + 1))
        train_iterations(epoch, model_name, voc, pairs_train, pairs_valid, encoder, decoder, encoder_optimizer,
                         decoder_optimizer, scheduler_encoder, scheduler_decoder, embedding, num_iteration, batch_size,
                         10, 10, dataset, teacher_forcing_ratio, num_epochs)

# reload the checkpoints:
iteration = num_iteration
epoch = num_epochs - 1
encoder, decoder, _, _ = build_models(load_filename=True,
                                      hidden_size=hidden_size,
                                      encoder_n_layers=num_layers,
                                      decoder_n_layers=num_layers,
                                      batch_size=batch_size,
                                      dataset_name=dataset,
                                      embedding_size=embed_size,
                                      model_name=model_name,
                                      iteration=iteration,
                                      epoch=epoch)

evaluate_input(encoder, decoder, voc)
