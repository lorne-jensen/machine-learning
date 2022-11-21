import torch
from numpy import random
from torch import nn

from src.vocab import SOS_token, EOS_token


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, embedding: nn.Embedding, num_layers):
        super(Encoder, self).__init__()

        # self.embedding provides a vector representation of the inputs to our model

        # self.lstm, accepts the vectorized input and passes a hidden state

        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.hidden = torch.zeros(1, 1, hidden_size)

        self.embedding = embedding
        # Initialize LSTM; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.lstm = nn.LSTM(input_size=embedding.embedding_dim, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, i):
        '''
        Inputs: i, the src vector
        Outputs: o, the encoder outputs
                h, the hidden state
                c, the cell state
        '''
        # Convert word indexes to embeddings
        embedded = self.embedding(i)
        # Pack padded batch of sequences for RNN module
        # Forward pass through LSTM, returns outputs, hidden, c
        return self.lstm(embedded)


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, embedding: nn.Embedding, num_layers):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        # self.embedding provides a vector representation of the target to our model
        self.embedding = embedding

        # self.lstm, accepts the embeddings and outputs a hidden state
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_size, num_layers=num_layers)

        # self.ouput, predicts on the hidden state via a linear output layer
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, last_hidden):
        """
        Inputs: i, the target vector
        Outputs: o, the prediction
                h, the hidden state
        """
        v = self.embedding(input)
        out, hidden = self.lstm(v, last_hidden)
        prediction = self.output(out.squeeze(0))

        return out, hidden, prediction


class Seq2Seq(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(Seq2Seq, self).__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder

    def forward(self, input, input_length, max_length):

        encoder_outputs, encoder_hidden = self.encoder(input, input_length)

        decoder_hidden = encoder_hidden
        decoder_input = torch.ones(1, 1, device='cpu', dtype=torch.long) * SOS_token

        for _ in range(max_length):

            decoder_output, decoder_hidden, dec_pred = self.decoder(decoder_input, decoder_hidden, encoder_outputs)



        return 0
