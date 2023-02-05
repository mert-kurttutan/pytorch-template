import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(nn.Module):
    """
    Standard Baseline CNN.
    Basic unit is conv->BN->pool->activation
    """

    @staticmethod
    def get_activation(name: str):

        activation_map = {
            "relu": F.relu,
            "elu": F.elu,
            "gelu": F.gelu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid
        }

        return activation_map[name]

    def __init__(self, config):
        """
        We define an convolutional network that predicts
        the sign from an image. The components required are:
        Args:
            params: (Params) contains n_channels
        """
        super(BaseCNN, self).__init__()
        self.n_channel = config["n_channel"]

        self.activation = (
            F.relu if config["activation"] is None
            else self.get_activation(config["activation"])
        )

        # each of the convolution layers below have the arguments
        # (input_channels, output_channels, filter_size, stride, padding).
        # We also include batch normalisation layers that help stabilise
        # training. For more details on how to use these layers, check out
        # the documentation.
        self.conv1 = nn.Conv2d(config["img_channel"], self.n_channel, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.n_channel)
        self.conv2 = nn.Conv2d(self.n_channel, self.n_channel*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.n_channel*2)
        self.conv3 = nn.Conv2d(self.n_channel*2, self.n_channel*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.n_channel*4)

        # 2 fully connected layers to transform the output of
        # the convolution layers to the final output
        self.fc1 = nn.Linear(4*4*self.n_channel*4, self.n_channel*4)
        self.fcbn1 = nn.BatchNorm1d(self.n_channel*4)
        self.fc2 = nn.Linear(self.n_channel*4, config["n_class"])
        self.dropout_rate = config["fc_pdrop"]

    def forward(self, s):
        """
        This function defines how we use the components of our
        network to operate on an input batch.
        Args:
            s: contains a batch of images, of dimension batch_size x 3 x 32 x 32 .
        Returns:
            out: dimension batch_size x class_num with the
                log prob for the image labels.
        Note: the dimensions after each step are provided
        """
        # we apply the conv layers, followed by batch normalisation, maxpool and activation x 3
        s = self.bn1(self.conv1(s))                     # b_size x n_ch x 32 x 32
        s = self.activation(F.max_pool2d(s, 2))         # b_size x n_ch x 16 x 16
        s = self.bn2(self.conv2(s))                     # b_size x n_ch*2 x 16 x 16
        s = self.activation(F.max_pool2d(s, 2))         # b_size x n_ch*2 x 8 x 8
        s = self.bn3(self.conv3(s))                     # b_size x n_ch*4 x 8 x 8
        s = self.activation(F.max_pool2d(s, 2))         # b_size x n_ch*4 x 4 x 4

        # flatten the output for each image
        s = s.view(-1, 4*4*self.n_channel*4)            # b_size x 4*4*n_channels*4

        # apply 2 fully connected layers with dropout
        s = self.activation(self.fcbn1(self.fc1(s)))
        s = F.dropout(s, p=self.dropout_rate, training=self.training)  # b_size x self.n_channels*4
        s = self.fc2(s)                                 # b_size x 10

        return s
