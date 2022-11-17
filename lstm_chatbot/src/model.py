import torch
from numpy import random
from torch import nn

from src.vocab import SOS_token, EOS_token


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, num_layers):
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
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

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
        # Forward pass through LSTM, returns outputs, hidden
        return self.lstm(embedded)


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, embedding_size, num_layers):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size

        # self.embedding provides a vector representation of the target to our model
        self.embedding = nn.Embedding(num_embeddings=self.output_size,
                                      embedding_dim=self.embedding_size)

        # self.lstm, accepts the embeddings and outputs a hidden state
        self.lstm = nn.LSTM(self.embedding_size, hidden_size, num_layers=num_layers)

        # self.ouput, predicts on the hidden state via a linear output layer
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        """
        Inputs: i, the target vector
        Outputs: o, the prediction
                h, the hidden state
        """
        v = input.unsqueeze(0)
        v = self.embedding(v)
        out, hidden = self.lstm(v, hidden)
        prediction = self.output(out.squeeze(0))

        return out, hidden, prediction


class Seq2Seq(nn.Module):

    def __init__(self, encoder_input_size, encoder_hidden_size, encoder_embedding_size,
                 decoder_hidden_size, decoder_output_size, decoder_embedding_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(encoder_input_size, encoder_hidden_size, encoder_embedding_size, num_layers)
        self.decoder = Decoder(decoder_hidden_size, decoder_output_size, decoder_embedding_size, num_layers)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)

        encoder_out, encoder_hidden = self.encoder(src)

        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([SOS_token])

        for t in range(1, trg_len):
            decoder_output, decoder_hidden, decoder_pred = \
                self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (trg[t] if teacher_force else topi)
            if teacher_force == False and input.item() == EOS_token:
                break

        return outputs
