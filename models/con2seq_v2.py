import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from models.attentions import MultiHeadAttention



class RnnType:
    GRU = 1
    LSTM = 2


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=None, stride=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), kernel_size=3, stride=None):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1], kernel_size=kernel_size, stride=stride) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x.flatten(1, -1))
            x = self.pool(x)
        return ftrs


class AttentionModel:
    NONE = 0
    DOT = 1
    GENERAL = 2
    CONCAT = 3


class Parameters:
    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec("self.%s=%s" % (k, v))

    def forward(self, input):
        hidden = self.init_hidden(input.shape[0])
        output, hidden = self.net(input,
                                  hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        if self.method == 'GRU':
            return torch.zeros(self.n_layers * self.n_directions,
                               batch_size,
                               self.hidden_size).to(self.device)
        elif self.method == 'LSTM':
            return (torch.zeros(self.n_layers * self.n_directions,
                                batch_size,
                                self.hidden_size).to(self.device),
                    torch.zeros(self.n_layers * self.n_directions,
                                batch_size,
                                self.hidden_size).to(self.device))
        else:
            raise Exception('Unknown rnn_type. Valid options: "gru", "lstm"')


class Decoder(nn.Module):
    def __init__(self, device, params):
        super(Decoder, self).__init__()
        self.params = params
        self.device = device
        # Calculate number of directions
        self.num_directions = 2 if self.params.bidirectional == True else 1
        self.encoder = Encoder(chs=self.params.chs_encoder,
                               kernel_size=self.params.kernel_size,
                               stride=self.params.stride)
        # Attention layer
        self.attention = MultiHeadAttention(d_model=self.params.rnn_hidden_dim * self.num_directions, num_heads=4)
        # RNN layer
        rnn = None
        if self.params.rnn_type == RnnType.GRU:
            rnn = nn.GRU
            self.decoder_hidden = nn.Sequential(
                torch.nn.LazyLinear(self.params.layer_width),
                torch.nn.ReLU(),
                torch.nn.Linear(self.params.layer_width,
                                self.num_directions * self.params.num_layers * self.params.rnn_hidden_dim),
                torch.nn.Sigmoid()
            )
        elif self.params.rnn_type == RnnType.LSTM:
            rnn = nn.LSTM
            self.decoder_hidden1 = nn.Sequential(
                torch.nn.LazyLinear(self.params.layer_width),
                torch.nn.ReLU(),
                torch.nn.Linear(self.params.layer_width,
                                self.num_directions * self.params.num_layers * self.params.rnn_hidden_dim),
                torch.nn.Sigmoid()
            )
            self.decoder_hidden2 = nn.Sequential(
                torch.nn.LazyLinear(self.params.layer_width),
                torch.nn.ReLU(),
                torch.nn.Linear(self.params.layer_width,
                                self.num_directions * self.params.num_layers * self.params.rnn_hidden_dim),
                torch.nn.Sigmoid()
            )
        else:
            raise Exception("[Error] Unknown RnnType. Currently supported: RnnType.GRU=1, RnnType.LSTM=2")
        self.rnn = rnn(input_size=self.params.rnn_hidden_dim,
                       hidden_size=self.params.rnn_hidden_dim,
                       num_layers=self.params.num_layers,
                       bidirectional=self.params.bidirectional,
                       dropout=self.params.dropout,
                       batch_first=True)
        # self.rnn2 = rnn(input_size=self.params.rnn_hidden_dim*self.num_directions,
        #                hidden_size=self.params.rnn_hidden_dim,
        #                num_layers=self.params.num_layers,
        #                bidirectional=self.params.bidirectional,
        #                dropout=self.params.dropout,
        #                batch_first=True)
        self.decoder_input = nn.Sequential(
            torch.nn.LazyLinear(self.params.layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.params.layer_width, self.params.output_sequence_length * self.params.rnn_hidden_dim),
            torch.nn.Sigmoid()
        )
        self.linear = nn.Sequential(
            torch.nn.Linear(self.params.rnn_hidden_dim * self.num_directions, self.params.layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.params.layer_width, 1),
            torch.nn.Sigmoid()
        )
        self.encoder_output_linear = nn.Sequential(
            nn.LazyLinear(self.params.output_sequence_length * self.params.rnn_hidden_dim * 2)
        )

        # self.kernel = nn.Parameter(torch.randn(self.params.num_of_wavelength, self.params.num_of_filter))
        self.kernel = nn.Linear(self.params.num_of_wavelength, self.params.num_of_filter)

        self.to(device)

    def init_hidden(self, encoder_outputs):
        if self.params.rnn_type == RnnType.GRU:
            hidden = self.decoder_hidden(encoder_outputs).view(self.batch_size,
                                                               self.params.num_layers * self.num_directions,
                                                               self.params.rnn_hidden_dim).transpose(0, 1)
            return hidden.contiguous().to(self.device)
        elif self.params.rnn_type == RnnType.LSTM:
            hidden1 = self.decoder_hidden1(encoder_outputs).view(self.batch_size,
                                                                 self.params.num_layers * self.num_directions,
                                                                 self.params.rnn_hidden_dim).transpose(0, 1)
            hidden2 = self.decoder_hidden2(encoder_outputs).view(self.batch_size,
                                                                 self.params.num_layers * self.num_directions,
                                                                 self.params.rnn_hidden_dim).transpose(0, 1)
            return (hidden1.contiguous().to(self.device),
                    hidden2.contiguous().to(self.device)
                    )
        else:
            raise Exception('Unknown rnn_type. Valid options: "gru", "lstm"')

    def forward(self, inputs):
        self.batch_size, seq_len = inputs.shape  # to encoder
        encoder_outputs = self.encoder(inputs.unsqueeze(1))
        encoder_outputs = torch.cat(encoder_outputs, 1)
        # encoder_outputs (batch_size, N)
        # decoder_inputs: (batch_size, sequence length, hidden)
        decoder_inputs = self.decoder_input(encoder_outputs).view(self.batch_size,
                                                                  self.params.output_sequence_length,
                                                                  self.params.rnn_hidden_dim)
        self.hidden = self.init_hidden(encoder_outputs)
        # Push through RNN layer
        rnn_output, _ = self.rnn(decoder_inputs, self.hidden)
        # (batch_size, output_seq_len, num_directions*hidden)
        decoder_inputs_ = torch.cat([decoder_inputs, decoder_inputs], 2)
        encoder_output_ = self.encoder_output_linear(encoder_outputs).view(self.batch_size,
                                                                           self.params.output_sequence_length,
                                                                           self.params.rnn_hidden_dim * 2)
        context, att = self.attention(rnn_output, encoder_output_, encoder_output_)
        # context, att = self.attention(rnn_output, decoder_inputs_, decoder_inputs_)
        # context = self.rnn2(context, self.hidden)[0]
        # residual connection
        # context = context + rnn_output
        # print(rnn_output.shape)
        spectrum_out = self.linear(context).squeeze(2) + rnn_output.mean(dim=2)
        # response_out = torch.matmul(spectrum_out, self.kernel)
        response_out = self.kernel(spectrum_out)

        return spectrum_out, response_out
