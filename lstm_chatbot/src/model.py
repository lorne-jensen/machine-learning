from torch import nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        # self.embedding provides a vector representation of the inputs to our model

        # self.lstm, accepts the vectorized input and passes a hidden state

    def forward(self, i):
        '''
        Inputs: i, the src vector
        Outputs: o, the encoder outputs
                h, the hidden state
                c, the cell state
        '''

        return o, h, c


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()

        # self.embedding provides a vector representation of the target to our model

        # self.lstm, accepts the embeddings and outputs a hidden state

        # self.ouput, predicts on the hidden state via a linear output layer

    def forward(self, i, h):
        '''
        Inputs: i, the target vector
        Outputs: o, the prediction
                h, the hidden state
        '''

        return o, h


class Seq2Seq(nn.Module):

    def __init__(self, encoder_input_size, encoder_hidden_size, decoder_hidden_size, decoder_output_size):
        super(Seq2Seq, self).__init__()

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        return o
