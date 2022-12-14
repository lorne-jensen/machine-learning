import os

import torch
import random
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from src.data_to_tensors import batch_to_train_data
from src.eval import evaluate
from src.model import Encoder, Decoder, Seq2Seq
from src.prepare_data import MAX_LENGTH, get_vocab_and_sentence_pairs, DATA_HOME, normalize_string
from src.vocab import SOS_token
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    # crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    # crossEntropy = -torch.log(torch.gather(inp, 1, target.view(1, -1)).squeeze(1))
    crossEntropy = -torch.gather(inp, 1, target.view(1, -1).squeeze(1))
    # crossEntropy = -inp.masked_select(torch.broadcast_to((target.view(-1, 1) - 2).ge(0), inp.size()))
    loss = crossEntropy.masked_select(mask).mean()
    # loss = crossEntropy.mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def validate(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, batch_size):

    encoder.eval()
    decoder.eval()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = (torch.clone(encoder_hidden[0]).type(torch.float32),
                      torch.clone(encoder_hidden[1]).type(torch.float32))

    # Forward batch of sequences through decoder one time step at a time
    # criterion = nn.NLLLoss()
    with torch.no_grad():
        for t in range(max_target_len):
            decoder_hidden, pred = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # No teacher forcing: next input is decoder's own current output
            _, topi = pred.topk(1)
            topi = topi
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(pred, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    return sum(print_losses) / n_totals


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
          encoder_optimizer, decoder_optimizer, teacher_forcing_ratio, batch_size,
          max_length=MAX_LENGTH, clip=50, voc=None):

    encoder.train()
    decoder.train()
    # Zero gradients
    encoder.zero_grad()
    decoder.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = (torch.clone(encoder_hidden[0]).type(torch.float32),
                      torch.clone(encoder_hidden[1]).type(torch.float32))

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    # criterion = nn.NLLLoss()
    if True:
        for t in range(max_target_len):
            decoder_hidden, pred = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(pred, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
            # loss -= criterion(pred, target_variable[t])
    else:
        for t in range(max_target_len):
            decoder_hidden, pred = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # No teacher forcing: next input is decoder's own current output
            _, topi = pred.topk(1)
            topi = topi
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(pred, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
            # loss -= criterion(pred, target_variable[t])

    # Perform backpropagation)
    loss.backward()

    # plt.figure()
    # decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    # decoder_input = decoder_input.to(device)
    # _, pred = decoder(decoder_input, encoder_hidden)
    # p = pred.detach().numpy()[0].flatten()
    # plt.plot(np.arange(len(p)), p)
    # plt.show()

    # Clip gradients: gradients are modified in place
    # _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    # _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals
    # return loss.item() / max_target_len


def evaluate_iteration(encoder, decoder, voc, input_pair):
    encoder.eval()
    decoder.eval()
    seq2seq = Seq2Seq(encoder, decoder)

    # Normalize sentence
    input_sentence = normalize_string(input_pair[0])
    print('Input sentence: ', input_sentence)
    # Evaluate sentence
    output_words = evaluate(seq2seq, voc, input_sentence)
    # Format and print response sentence
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('Bot:', ' '.join(output_words))
    print('Expected: ', input_pair[1])


def train_iterations(epoch, model_name, voc, pairs_train, pairs_valid, encoder, decoder, encoder_optimizer,
                     decoder_optimizer, scheduler_encoder, scheduler_decoder, embedding,
                     n_iteration=0, batch_size=64, print_every=10,
                     save_every=10, corpus_name='', teacher_forcing_ratio=0.5, num_epochs=2):

    # Load batches for each iteration
    pairs_randomized = np.random.permutation(pairs_train)
    if n_iteration == 0:
        n_iteration = int(len(pairs_train) / batch_size)  # do the whole dataset
    training_batches = [batch_to_train_data(voc, [pairs_randomized[i * batch_size + k] for k in range(batch_size)])
                        for i in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    # if load_filename:
    #     start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len,
                     encoder, decoder, encoder_optimizer, decoder_optimizer, teacher_forcing_ratio,
                     batch_size, MAX_LENGTH, voc=voc)
        print_loss += loss

        # scheduler_encoder.step()
        # scheduler_decoder.step()

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Epoch: {}; Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}"
                  .format(epoch + 1, iteration, (epoch * n_iteration + iteration) / (num_epochs * n_iteration) * 100,
                          print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if iteration % save_every == 0:
            directory = os.path.join(DATA_HOME, model_name, corpus_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'epoch': epoch + 1,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}_{}.tar'.format(epoch, iteration, 'checkpoint')))

    val_loss = 0
    n_valid_iters = int(len(pairs_valid) / batch_size)
    valid_data = [batch_to_train_data(voc, [pairs_valid[i * batch_size + k] for k in range(batch_size)])
                  for i in range(n_valid_iters)]
    print('Beginning epoch validation...')
    for d in valid_data:
        input_variable, lengths, target_variable, mask, max_target_len = d
        val_loss += validate(input_variable, lengths, target_variable, mask, max_target_len,
                             encoder, decoder, batch_size)

    validation_loss = val_loss / len(valid_data)

    rand_pair = pairs_train[np.random.randint(0, len(pairs_train))]
    evaluate_iteration(encoder, decoder, voc, rand_pair)
    scheduler_encoder.step(validation_loss)
    scheduler_decoder.step(validation_loss)

    print('Epoch {} validation loss: {}'.format(epoch, validation_loss))


def build_models(load_filename: bool = False,
                 hidden_size=512,
                 encoder_n_layers=1,
                 decoder_n_layers=1,
                 batch_size=64,
                 embedding_size=256,
                 dataset_name='squad1',
                 model_name='boof',
                 iteration=1,
                 epoch=1):
    # Configure models

    voc, pairs_train, pairs_valid = get_vocab_and_sentence_pairs(dataset_name)

    # Load model if a loadFilename is provided
    if load_filename:
        directory = os.path.join(DATA_HOME, model_name, dataset_name)
        # If loading on same machine the model was trained on
        checkpoint = torch.load(os.path.join(directory, '{}_{}_{}.tar'.format(epoch, iteration, 'checkpoint')))
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, embedding_size)
    if load_filename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = Encoder(voc.num_words, hidden_size, embedding, encoder_n_layers)
    decoder = Decoder(hidden_size, voc.num_words, embedding, decoder_n_layers)
    if load_filename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')
    if not load_filename:
        return encoder, decoder, embedding
    else:
        return encoder, decoder, encoder_optimizer_sd, decoder_optimizer_sd
