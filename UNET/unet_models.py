from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int):
        """UNet Model

        Args:
            n_channels (int): Channels of input
            n_classes (int): Channels of output. Number of classes if task is segmentation or 3 if task is image generation
            hidden_dim: (int): Number of hidden_dim
        """
        super(UNet, self).__init__()


        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.hidden_dims = [hidden_dim*(2**i) for i in range(5)]

        self.inc = DoubleConv(in_channels=n_channels, out_channels=self.hidden_dims[0])
        self.down1 = Down(in_channels=self.hidden_dims[0], out_channels=self.hidden_dims[1])
        self.down2 = Down(in_channels=self.hidden_dims[1], out_channels=self.hidden_dims[2])
        self.down3 = Down(in_channels=self.hidden_dims[2], out_channels=self.hidden_dims[3])
        self.down4 = Down(in_channels=self.hidden_dims[3], out_channels=self.hidden_dims[4])
        self.up1 = Up(in_channels=self.hidden_dims[4], out_channels=self.hidden_dims[3])
        self.up2 = Up(in_channels=self.hidden_dims[3], out_channels=self.hidden_dims[2])
        self.up3 = Up(in_channels=self.hidden_dims[2], out_channels=self.hidden_dims[1])
        self.up4 = Up(in_channels=self.hidden_dims[1], out_channels=self.hidden_dims[0])
        self.head = Head(in_channels=self.hidden_dims[0], out_channels=n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.head(x)

        return logits
    
class ResUNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int):
        """ResUNet Model

        Args:
            n_channels (int): Channels of input
            n_classes (int): Channels of output. Number of classes if task is segmentation or 3 if task is image generation
            hidden_dim (int): Number of hidden dim
        """
        super(ResUNet, self).__init__()


        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.hidden_dims = [hidden_dim*(2**i) for i in range(5)]

        self.inc = DoubleResConv(in_channels=n_channels, out_channels=self.hidden_dims[0])
        self.down1 = DownRes(in_channels=self.hidden_dims[0], out_channels=self.hidden_dims[1])
        self.down2 = DownRes(in_channels=self.hidden_dims[1], out_channels=self.hidden_dims[2])
        self.down3 = DownRes(in_channels=self.hidden_dims[2], out_channels=self.hidden_dims[3])
        self.down4 = DownRes(in_channels=self.hidden_dims[3], out_channels=self.hidden_dims[4])
        self.up1 = UpRes(in_channels=self.hidden_dims[4], out_channels=self.hidden_dims[3])
        self.up2 = UpRes(in_channels=self.hidden_dims[3], out_channels=self.hidden_dims[2])
        self.up3 = UpRes(in_channels=self.hidden_dims[2], out_channels=self.hidden_dims[1])
        self.up4 = UpRes(in_channels=self.hidden_dims[1], out_channels=self.hidden_dims[0])
        self.head = Head(in_channels=self.hidden_dims[0], out_channels=n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.head(x)

        return logits
    
class AttentionUNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int):
        """_summary_

        Args:
            n_channels (int): Channels of input
            n_classes (int): Channels of output. Number of classes if task is segmentation or 3 if task is image generation
            hidden_dim (int, optional): Number of hidden dim.
        """
        super(AttentionUNet, self).__init__()


        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.hidden_dims = [hidden_dim*(2**i) for i in range(5)]

        self.inc = DoubleConv(in_channels=n_channels, out_channels=self.hidden_dims[0])
        self.down1 = Down(in_channels=self.hidden_dims[0], out_channels=self.hidden_dims[1])
        self.down2 = Down(in_channels=self.hidden_dims[1], out_channels=self.hidden_dims[2])
        self.down3 = Down(in_channels=self.hidden_dims[2], out_channels=self.hidden_dims[3])
        self.down4 = Down(in_channels=self.hidden_dims[3], out_channels=self.hidden_dims[4])
        self.up1 = UpAttention(in_channels=self.hidden_dims[4], out_channels=self.hidden_dims[3])
        self.up2 = UpAttention(in_channels=self.hidden_dims[3], out_channels=self.hidden_dims[2])
        self.up3 = UpAttention(in_channels=self.hidden_dims[2], out_channels=self.hidden_dims[1])
        self.up4 = UpAttention(in_channels=self.hidden_dims[1], out_channels=self.hidden_dims[0])
        self.head = Head(in_channels=self.hidden_dims[0], out_channels=n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.head(x)

        return logits