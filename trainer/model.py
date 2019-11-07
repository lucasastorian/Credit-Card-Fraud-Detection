import torch


class AutoEncoder(torch.nn.Module):

    def __init__(self, num_features, first_compression, second_compression):
        super(AutoEncoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_features, first_compression),
            torch.nn.ReLU(True),
            torch.nn.Linear(first_compression, second_compression))
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(second_compression, first_compression),
            torch.nn.ReLU(True),
            torch.nn.Linear(first_compression, num_features),
            torch.nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def compile_model(num_features, first_compression, second_compression):
    """Defines and compiles a 4-Layer Autoencoder Architecture


    :param args:
    :return:
    """
    model = AutoEncoder(num_features, first_compression, second_compression)

    return model




