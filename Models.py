import torch
import torch.nn as nn
import unittest 

class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
                    nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, padding_mode='reflect'),
                    nn.InstanceNorm2d(input_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, padding_mode='reflect'),
                    nn.InstanceNorm2d(input_dim)
                )

    def forward(self, x):
        identity_x = x.clone()
        return self.block(x) + identity_x

class ContractingBlock(nn.Module):
    def __init__(self, input_dim):
        super(ContractingBlock, self).__init__()
        self.block = nn.Sequential(
                    nn.Conv2d(input_dim, input_dim * 2, kernel_size=3, padding=1, stride=2, padding_mode='reflect'),
                    nn.InstanceNorm2d(input_dim * 2),
                    nn.ReLU(inplace=True)
                )

    def forward(self, x):
        return self.block(x)


class ExpandingBlock(nn.Module):
    def __init__(self, input_dim):
        super(ExpandingBlock, self).__init__()
        self.block = nn.Sequential(
                    nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(input_dim // 2),
                    nn.ReLU(inplace=True)
                )

    def forward(self, x):
        return self.block(x)


class ResUNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ResUNet, self).__init__()
        self.input_layer = nn.Conv2d(input_dim, hidden_dim, kernel_size=7, padding=3, padding_mode='reflect')
        self.contract1 = ContractingBlock(hidden_dim)
        self.contract2 = ContractingBlock(hidden_dim * 2)
        res_mult = 4
        self.res0 = ResidualBlock(hidden_dim * res_mult)
        self.res1 = ResidualBlock(hidden_dim * res_mult)
        self.res2 = ResidualBlock(hidden_dim * res_mult)
        self.res3 = ResidualBlock(hidden_dim * res_mult)
        self.res4 = ResidualBlock(hidden_dim * res_mult)
        self.res5 = ResidualBlock(hidden_dim * res_mult)
        self.res6 = ResidualBlock(hidden_dim * res_mult)
        self.res7 = ResidualBlock(hidden_dim * res_mult)
        self.res8 = ResidualBlock(hidden_dim * res_mult)
        self.expand2 = ExpandingBlock(hidden_dim * 4)
        self.expand3 = ExpandingBlock(hidden_dim * 2)
        self.output_layer = nn.Conv2d(hidden_dim, output_dim, kernel_size=7, padding=3, padding_mode='reflect')
        self.tanh = nn.Tanh()

    def forward(self, x):
        x0 = self.input_layer(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.output_layer(x13)
        return self.tanh(xn)

"""
    Unit Test class

"""
class TestResUNetOuput(unittest.TestCase):
    def test_ouput_shape(self):
        """
            Tests if the model preserves the input shape

        """
        model = ResUNet(3, 3)
        dummy_input = torch.randn(1, 3, 256, 512, dtype=torch.float)
        self.assertEqual(model(dummy_input).shape, dummy_input.shape)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

