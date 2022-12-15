import torch

from src.data_to_tensors import indexes_from_sentence
from src.model import Seq2Seq
from src.prepare_data import normalize_string, MAX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(seq2seq, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexes_from_sentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    if torch.cuda.is_available():
        input_batch = input_batch.to(device)
    # Decode sentence with searcher
    tokens, scores = seq2seq(input_batch, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluate_input(encoder, decoder, voc):
    encoder.eval()
    decoder.eval()
    input_sentence = ''
    seq2seq = Seq2Seq(encoder, decoder)
    while 1:
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            # Normalize sentence
            input_sentence = normalize_string(input_sentence)
            # Evaluate sentence
            output_words = evaluate(seq2seq, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")
