import torch
from numpy import random
from torch import nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size):
        super(Encoder, self).__init__()

        # self.embedding provides a vector representation of the inputs to our model

        # self.lstm, accepts the vectorized input and passes a hidden state

        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.hidden = torch.zeros(1, 1, hidden_size)

        self.embedding = nn.Embedding(num_embeddings=self.input_size,
                                      embedding_dim=self.embedding_size)
        # Initialize LSTM; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)

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
        # Forward pass through LSTM
        return self.lstm(embedded)


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, embedding_size):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size

        # self.embedding provides a vector representation of the target to our model
        self.embedding = nn.Embedding(num_embeddings=self.output_size,
                                      embedding_dim=self.embedding_size)

        # self.lstm, accepts the embeddings and outputs a hidden state
        self.lstm = nn.LSTM(self.embedding_size, hidden_size, num_layers=3)

        # self.ouput, predicts on the hidden state via a linear output layer
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, i, h, c):
        """
        Inputs: i, the target vector
        Outputs: o, the prediction
                h, the hidden state
        """
        v = i.unsqueeze(0)

        v = self.embedding(v)

        o, h, c = self.lstm(v, (h, c))

        p = self.output(o.squeeze(0))

        return o, h, p, c


class Seq2Seq(nn.Module):

    def __init__(self, encoder_input_size, encoder_hidden_size, encoder_embedding_size,
                 decoder_hidden_size, decoder_output_size, decoder_embedding_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(encoder_input_size, encoder_hidden_size, encoder_embedding_size)
        self.decoder = Decoder(decoder_hidden_size, decoder_output_size, decoder_embedding_size)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size

        o = torch.zeros(trg_len, batch_size, trg_vocab_size)

        h, c = self.encoder(src)

        x = trg[0, :]
        for t in range(1, trg_len):
            o, h, c = self.decoder(x, h, c)
            o[t] = o
            teacher_force = random.random() < teacher_forcing_ratio
            top_trg = o.argmax(1)
            x = trg[t] if teacher_force else top_trg

        return o
