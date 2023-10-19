import torch
import torch.nn as nn

class AutoencoderFC(nn.Module):

    def __init__(self, dim_code, emb_dim=2):
        super(AutoencoderFC, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dim_code, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=emb_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features = 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features = 1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=dim_code),
        )

    def forward(self, x):
        out_enc = self.encoder(x)
        out = self.decoder(out_enc)
        return out, out_enc


class AutoencoderConv1D(nn.Module):

    def __init__(self, input_dim, emb_dim=2):
        super(AutoencoderConv1D, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(16, emb_dim, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Linear(30,1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1,30),
            nn.ReLU(),
            nn.ConvTranspose1d(emb_dim, 16, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch = x.shape[0]
        x = torch.unsqueeze(x, 1)
        enc_out = self.encoder(x)
        out = self.decoder(enc_out)
        return torch.squeeze(out), enc_out.reshape(batch, 2)

class AutoencoderRNN(nn.Module):

    def __init__(self, seq_len, n_features, emb_dim=2):
        super(AutoencoderRNN, self).__init__()

        self.encoder = self.EncoderRNN(seq_len, n_features, emb_dim)
        self.decoder = self.DecoderRNN(seq_len, n_features, emb_dim)

    def forward(self, x):
        enc_out = self.encoder(x)

        out = self.decoder(enc_out)

        return torch.transpose(out, 0, 1), enc_out

    class EncoderRNN(nn.Module):

        def __init__(self,  seq_len, n_features, emb_dim=2):
            super().__init__()
            self.seq_len = seq_len
            self.n_features = n_features
            self.emb_dim = emb_dim
            self.hidden_dim = 2 * emb_dim

            self.rnn1 = nn.LSTM(
                input_size=n_features,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True
            )

            self.rnn2 = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=emb_dim,
                num_layers=1,
                batch_first=True
            )

        def forward(self, x):
            x = x.unsqueeze(-1)
            x, (_, _) = self.rnn1(x)
            x, (hidden_n, _) = self.rnn2(x)

            return hidden_n.reshape((self.n_features, self.emb_dim))

    class DecoderRNN(nn.Module):

        def __init__(self, seq_len, n_features, emb_dim=2):
            super().__init__()

            self.seq_len = seq_len
            self.input_dim = emb_dim
            self.hidden_dim = 2 * emb_dim
            self.n_features = n_features

            self.rnn1 = nn.LSTM(
                input_size=emb_dim,
                hidden_size=emb_dim,
                num_layers=1,
                batch_first=True
            )

            self.rnn2 = nn.LSTM(
                input_size=emb_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True
            )

            self.output_layer = nn.Linear(self.hidden_dim, n_features)

        def forward(self, x):
            x = x.repeat(self.seq_len, self.n_features)
            x = x.reshape((self.n_features, self.seq_len, self.input_dim))

            x, (hidden_n, cell_n) = self.rnn1(x)
            x, (hidden_n, cell_n) = self.rnn2(x)
            x = x.reshape((self.seq_len, self.hidden_dim))

            return self.output_layer(x)