# Merging features from all block together
class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        for params in self.resnet.parameters():
            params.requires_grad = False

            # layers without learnable parameters
        self.pool = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten()

        # to transform r2
        self.r2_conv1 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.r2_conv2 = nn.Conv2d(64, 64, (1, 1), stride=(1, 1))

        # to transform r3
        self.r3_conv = nn.Conv2d(128, 64, (1, 1), stride=(1, 1))

        # to transform r4
        self.r4_deconv = nn.ConvTranspose2d(256, 64, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.r4_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))

        # to transform r5
        self.r5_deconv1 = nn.ConvTranspose2d(512, 64, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.r5_deconv2 = nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.r5_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 6))

    def forward(self, x):
        # Before 1st block
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # multi-level features --> store features after each block
        r2 = self.resnet.layer1(x)
        r3 = self.resnet.layer2(r2)
        r4 = self.resnet.layer3(r3)
        r5 = self.resnet.layer4(r4)

        # transform r2 (56x56x64 -> 28x28x64)
        b1 = nn.ReLU(inplace=True)(self.r2_conv1(r2))
        b1 = self.pool(b1)
        b1 = nn.ReLU(inplace=True)(self.r2_conv2(b1))

        # transform r3 (28x28x128 -> 28x28x64)
        b2 = nn.ReLU(inplace=True)(self.r3_conv(r3))

        # transform r4 (14x14x256 -> 28x28x64)
        b3 = nn.ReLU(inplace=True)(self.r4_deconv(r4))
        b3 = nn.ReLU(inplace=True)(self.r4_conv(b3))

        # transform r5 (7x7x512 -> 28x28x64)
        b4 = nn.ReLU(inplace=True)(self.r5_deconv1(r5))
        b4 = nn.ReLU(inplace=True)(self.r5_deconv2(b4))
        b4 = nn.ReLU(inplace=True)(self.r5_conv(b4))

        # append layer-wise features
        merged = torch.cat((b1, b2, b3, b4), dim=1)

        # classification
        flatten = self.flatten(merged)
        y = self.classifier(flatten)

        return y
