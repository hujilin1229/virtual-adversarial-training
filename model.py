import torch.nn as nn

class VAT(nn.Module):

      def __init__(self, top_bn=True, input_channels=3, n_class=10):

            super(VAT, self).__init__()
            self.top_bn = top_bn
            self.main = nn.Sequential(
                  nn.Conv2d(input_channels, 128, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1),

                  nn.MaxPool2d(2, 2, 1),
                  nn.Dropout2d(p=0.5),

                  nn.Conv2d(128, 256, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1),

                  nn.MaxPool2d(2, 2, 1),
                  nn.Dropout2d(p=0.5),

                  nn.Conv2d(256, 512, 3, 1, 0, bias=False),
                  nn.BatchNorm2d(512),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(512, 256, 1, 1, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1),

                  nn.Conv2d(256, 128, 1, 1, 1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1),

                  nn.AdaptiveAvgPool2d((1, 1))
                  )

            self.linear = nn.Linear(128, n_class)
            self.bn = nn.BatchNorm1d(n_class)

      def forward(self, input, featmap_only=False):
            output = self.main(input)
            if not featmap_only:
                  output = self.linear(output.view(input.size()[0], -1))
                  if self.top_bn:
                        output = self.bn(output)
            return output