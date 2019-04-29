"""Author: Brandon Trabucco, Copyright 2019
Implements a CNN VAE for mnist."""


import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.optim as optim


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__(*args, **kwargs)

    def forward(self, x):
        return x


class ConvNet(nn.Module):

    def __init__(self, input_size, kernel_sizes, num_channels, strides, paddings, activations):
        super(ConvNet, self).__init__()
        self.cnn = nn.Sequential(
            *(x for cin, cout, k, s, p, a in zip([input_size] + num_channels[:-1], num_channels, 
                    kernel_sizes, strides, paddings, activations)
                for x in (nn.Conv2d(cin, cout, k, stride=s, padding=p), a())))

    def forward(self, x):
        return self.cnn(x)


class ConvTransposeNet(nn.Module):

    def __init__(self, input_size, kernel_sizes, num_channels, strides, paddings, output_paddings, activations):
        super(ConvTransposeNet, self).__init__()
        self.cnn_transpose = nn.Sequential(
            *(x for cin, cout, k, s, p, o, a in zip([input_size] + num_channels[:-1], num_channels, 
                    kernel_sizes, strides, paddings, output_paddings, activations)
                    for x in (nn.ConvTranspose2d(cin, cout, k, stride=s, padding=p, output_padding=o), a())))

    def forward(self, x):
        return self.cnn_transpose(x)


BATCH_SIZE = 512
EPOCHS = 2


if __name__ == "__main__":


    data_dict = io.loadmat("mnist_data.mat")
    train_images = (np.reshape(data_dict["training_data"], 
        [60000, 1, 28, 28]).astype(np.float32) / 127.5 - 1.0)
    train_labels = data_dict["training_labels"].astype(np.int32)[:, 0]
    dataset = utils.TensorDataset(torch.Tensor(train_images), torch.Tensor(train_labels))
    loader = utils.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    encoder = ConvNet(1, [5, 5], [32, 64], [2, 2], [2, 2], [nn.ReLU, nn.ReLU])
    mean = ConvNet(64, [1], [64], [1], [0], [Identity])
    stddev = ConvNet(64, [1], [64], [1], [0], [Identity])
    decoder = ConvTransposeNet(64, [5, 5], [32, 1], [2, 2], [2, 2], [1, 1], [nn.ReLU, nn.ReLU])

    criterion = nn.MSELoss()
    optimizer = optim.Adam([{"params": encoder.parameters(), "lr": 0.001}, {
        "params": mean.parameters(), "lr": 0.001}, {
        "params": stddev.parameters(), "lr": 0.001}, {
        "params": decoder.parameters(), "lr": 0.001}])

    for e in range(EPOCHS):

        for i, (image, label) in enumerate(loader):

            features = encoder(image * 2.0 - 1.0)
            mm, log_ss = mean(features), stddev(features)
            ss = torch.exp(log_ss)
            reconstruction = decoder(torch.randn(mm.shape) * ss + mm)

            kldiv = torch.mean(0.5 * (ss**2 + mm**2) - 0.5 - log_ss)
            loss = criterion(image, reconstruction) + kldiv
            print("Epoch {0:05d} Iteration {1:05d} loss was {2:.5f}".format(e, i, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                plt.imshow((decoder(torch.randn([1, 64, 7, 7])) / 2.0 + 0.5).view(-1, 28, 28).detach().numpy()[0])
                plt.axis("off")
                plt.savefig("{0:05d}-{1:05d}.png".format(e, i))
                plt.close()


