import torch
from numpy import random
from torch import nn
import torch.nn.functional as F

from src.vocab import SOS_token, EOS_token


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, embedding: nn.Embedding, num_layers, dropout=0.1):
        super(Encoder, self).__init__()

        # self.embedding provides a vector representation of the inputs to our model

        # self.lstm, accepts the vectorized input and passes a hidden state

        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.hidden = torch.zeros(1, 1, hidden_size)

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        # Initialize LSTM; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.lstm = nn.LSTM(input_size=embedding.embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=(0 if num_layers == 1 else dropout))

    def forward(self, i, input_lengths):
        '''
        Inputs: i, the src vector
        Outputs: o, the encoder outputs
                h, the hidden state
                c, the cell state
        '''
        # Convert word indexes to embeddings
        embedded = self.embedding(i)
        embedded = self.embedding_dropout(embedded)
        # Pack padded batch of sequences for RNN module
        # # Forward pass through LSTM, returns outputs, hidden, c
        # return self.lstm(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.lstm(packed)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        return outputs, hidden


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, embedding: nn.Embedding, num_layers, dropout=0.1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        # self.embedding provides a vector representation of the target to our model
        self.embedding = embedding

        self.embedding_dropout = nn.Dropout(dropout)

        # self.lstm, accepts the embeddings and outputs a hidden state
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_size, num_layers=num_layers,
                            dropout=(0 if num_layers == 1 else dropout))

        # self.ouput, predicts on the hidden state via a linear output layer
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, last_hidden):
        """
        Inputs: i, the target vector
        Outputs: o, the prediction
                h, the hidden state
        """
        v = self.embedding(input)
        v = self.embedding_dropout(v)
        out, hidden = self.lstm(v, last_hidden)
        prediction = F.softmax(self.output(out.squeeze(0)), dim=1)

        return out, hidden, prediction


class Seq2Seq(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(Seq2Seq, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder: Encoder = encoder
        self.encoder.to(self.device)
        self.decoder: Decoder = decoder
        self.decoder.to(self.device)

    def forward(self, input, input_length, max_length):

        encoder_outputs, encoder_hidden = self.encoder(input, input_length)

        decoder_hidden = encoder_hidden
        decoder_input = torch.ones(1, 1, device='cpu', dtype=torch.long) * SOS_token

        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)

        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden, prediction = self.decoder(decoder_input, decoder_hidden)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(prediction, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)

            if decoder_input.data == EOS_token:
                break

        # Return collections of word tokens and scores
        return all_tokens, all_scores
