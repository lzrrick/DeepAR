import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import MyLoader, conditioning_range, prediction_range


class DeepAR(nn.Module):

    def __init__(self,
                 embedding_size=20,
                 input_size=3,
                 hidden_size=40,
                 num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(1, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size + input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.gaussian_mu = nn.Linear(hidden_size, 1)
        self.gaussian_sigma = nn.Sequential(nn.Linear(hidden_size, 1),
                                            nn.Softplus())

    def forward(self, z1, xc, device):
        batch_size = z1.shape[0]
        # scale factor
        v = torch.mean(z1, dim=1, keepdim=True) + 1
        # encode
        h_t = torch.zeros((1, batch_size, self.hidden_size)).to(
            device)  # h_t c_t (1, batch_szie, hidden_size)
        c_t = torch.zeros((1, batch_size, self.hidden_size)).to(device)

        for t in range(1, conditioning_range):
            inputs = torch.cat(
                [self.embedding(z1[:, t - 1:t, :] / v), xc[:, t:t + 1, :]],
                dim=-1)
            _, (h_t, c_t) = self.lstm(
                inputs,
                (h_t, c_t))  # output (batch_size, sql_len, hidden_size)

        # decode
        outputs = torch.empty((batch_size, prediction_range, 1)).to(device)
        mus = torch.empty((batch_size, prediction_range, 1)).to(device)
        sigmas = torch.empty((batch_size, prediction_range, 1)).to(device)
        y_pre = z1[:, -1:, :]
        for t in range(conditioning_range,
                       conditioning_range + prediction_range):
            inputs = torch.cat([
                self.embedding(y_pre.reshape(-1, 1, 1) / v), xc[:, t:t + 1, :]
            ],
                               dim=-1)
            _, (h_t, c_t) = self.lstm(inputs, (h_t, c_t))
            mu = self.gaussian_mu(h_t[0])
            sigma = self.gaussian_sigma(h_t[0])
            # gauss sample
            gaussian = torch.distributions.normal.Normal(mu, sigma)
            y_pre = gaussian.sample()
            outputs[:, t - conditioning_range, :] = y_pre * v.reshape(-1, 1)
            mus[:, t - conditioning_range, :] = mu * v.reshape(-1, 1)
            sigmas[:, t - conditioning_range, :] = sigma * v.reshape(-1, 1)

        return outputs, mus, sigmas


def gaussian_likelihood_loss(z, mu, sigma):
    '''
    Gaussian Liklihood Loss
    likelihood: 
    (2 pi sigma^2)^(-1/2) exp(-(z - mu)^2 / (2 sigma^2))
    log likelihood:
    -1/2 * (log (2 pi) + 2 * log (sigma)) - (z - mu)^2 / (2 sigma^2)
    '''
    negative_likelihood = torch.log(sigma) + (z - mu)**2 / (2 * sigma**2)
    return negative_likelihood.mean()


def train(model, loader, device):
    model.to(device)
    num_epoch = 500
    optim = torch.optim.Adam(model.parameters(), 1e-3)
    loss_array = []
    min_loss = 1e8
    for _ in tqdm(range(num_epoch)):
        ave_loss = 0
        for z1, z2, xc in tqdm(loader):
            z1 = z1.permute(2, 1, 0).to(device)
            z2 = z2.permute(2, 1, 0).to(device)
            xc = xc.repeat(z1.shape[0], 1, 1).to(device)
            _, mu, sigma = model(z1, xc, device)
            loss = gaussian_likelihood_loss(z2, mu, sigma)
            ave_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
        loss_array.append(ave_loss / len(loader))
        plt.cla()
        plt.plot(loss_array)
        plt.savefig('img/img.png')
        if loss_array[-1] < min_loss:
            min_loss = loss_array[-1]
            torch.save(model, 'model/model.pth')


if __name__ == '__main__':
    loader = MyLoader('train')
    model = DeepAR()
    # model = torch.load('model/model.pth')
    train(model, loader, 'cuda' if torch.cuda.is_available() else 'cpu')
