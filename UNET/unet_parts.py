import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """Conv2d -> BatchNorm -> ReLu * 2: [N, in_channels, H, W] -> [N, out_channels, H, W]

        Args:
            in_channels (int): Channels of input
            out_channels (int): Channels of output
            mid_channels (int, optional): If defined, it will act as the number of channels of the first convolution output. Defaults to None.
        """

        super(DoubleConv, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels == None:
            mid_channels = out_channels
        self.mid_channels = mid_channels

        self.double_conv = nn.Sequential(
            # [N, in_channels, H, W] -> [N, mid_channels, H, W]
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # [N, mid_channels, H, W] -> [N, out_channels, H, W]
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [N, in_channels, H, W]

        Returns:
            torch.Tensor: [N, out_channels, H, W]
        """

        out = self.double_conv(x)
        return out
    
class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """Down part of the net: [N, self.in_channels, H, W] -> [N, self.out_channels, (H - (kernel_size - 1) - 1) / kernel_size + 1, (W - (kernel_size - 1) - 1) / kernel_size + 1]

        Args:
            in_channels (int): Channels of input
            out_channels (int): Channels of output
            mid_channels (int, optional): If defined, it will act as the number of channels of the first convolution output. Defaults to None.
        """

        super(Down, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels == None:
            mid_channels = out_channels
        self.mid_channels = mid_channels

        self.down = nn.Sequential(
            # [N, in_channels, H, W] -> [N, in_channels, (H - (kernel_size - 1) - 1) // kernel_size + 1, (W - (kernel_size - 1) - 1) // kernel_size + 1]
            # e.g. [N, C, 225, 225] -> [N, C, 112, 112]
            nn.MaxPool2d(kernel_size=2),
            # [N, in_channels, H, W] -> [N, out_channels, H, W]
            DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [N, in_channels, H, W]

        Returns:
            torch.Tensor: [N, out_channels, (H - (kernel_size - 1) - 1) / kernel_size + 1, (W - (kernel_size - 1) - 1) / kernel_size + 1]
        """

        out = self.down(x)
        return out
    
class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """Up part of the net: [N, in_channels, H_1, W_1], [N, in_channels // 2, H_2, W_2] -> [N, out_channels, H_2, W_2]

        Args:
            in_channels (int): Channels of input
            out_channels (int): Channels of output
            mid_channels (int, optional): If defined, it will act as the number of channels of the first convolution output. Defaults to None.
        """

        super(Up, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels == None:
            mid_channels = out_channels
        self.mid_channels = mid_channels

        # [N, in_channels, H, W] -> [N, in_channels // 2, (H - 1) * stride + kernel_size - 1 + 1, (W - 1) * stride + kernel_size - 1 + 1]
        # e.g. [N, C, 225, 225] -> [N, C, 450, 450]
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2,  stride=2)
        # [N, in_channels, H, W] -> [N, out_channels, H, W]
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (torch.Tensor): [N, in_channels, H_1, W_1]
            x2 (torch.Tensor): [N, in_channels // 2, H_2, W_2]

        Returns:
            torch.Tensor: [N, out_channels, H_2, W_2]
        """

        x1 = self.up(x1)
        
        # Take the size difference between x1 and x2
        x1_h = x1.shape[-2]
        x1_w = x1.shape[-1]
        x2_h = x2.shape[-2]
        x2_w = x2.shape[-1]

        h_diff = x2_h - x1_h
        w_diff = x2_w - x1_w

        # Pad x1 so x1 has the same size as x2
        pad_h = [h_diff // 2, h_diff - h_diff // 2]
        pad_w = [w_diff // 2, w_diff - w_diff // 2]
        x1 = F.pad(x1, pad_w + pad_h)

        # Concat by channels
        x = torch.cat([x2, x1], dim=1)
        out = self.double_conv(x)

        return out
    
class Head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Head of the net:

        Args:
            in_channels (int): Channels of input
            out_channels (int): Channels of output. Number of classes if task is segmentation or 3 if task is image generation
        """

        super(Head, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels

        # [N, in_channels, H, W] -> [N, out_channels, H, W]
        self.head = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [N, in_channels, H, W]

        Returns:
            torch.Tensor: [N, out_channels, H, W]
        """

        out = self.head(x)
        return out
    
class DoubleResConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """DoubleConv with skip connection: [N, in_channels, H, W] -> [N, out_channels, H, W]

        Args:
            in_channels (int): Channels of input
            out_channels (int): Channels of output
            mid_channels (int, optional): If defined, it will act as the number of channels of the first convolution output. Defaults to None.
        """

        super(DoubleResConv, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels == None:
            mid_channels = out_channels
        self.mid_channels = mid_channels

        self.double_conv = nn.Sequential(
            # [N, in_channels, H, W] -> [N, mid_channels, H, W]
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # [N, mid_channels, H, W] -> [N, out_channels, H, W]
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.skip_connection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.last_activation = nn.ReLU()
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [N, in_channels, H, W]

        Returns:
            torch.Tensor: [N, out_channels, H, W]
        """

        conv_out = self.double_conv(x)
        skip_out = self.skip_connection(x)
        out = torch.add(conv_out, skip_out)
        out = self.last_activation(out)

        return out
    
class DownRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """Down part of the net with skip connection: [N, self.in_channels, H, W] -> [N, self.out_channels, (H - (kernel_size - 1) - 1) / kernel_size + 1, (W - (kernel_size - 1) - 1) / kernel_size + 1]

        Args:
            in_channels (int): Channels of input
            out_channels (int): Channels of output
            mid_channels (int, optional): If defined, it will act as the number of channels of the first convolution output. Defaults to None.
        """

        super(DownRes, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels == None:
            mid_channels = out_channels
        self.mid_channels = mid_channels

        self.down = nn.Sequential(
            # [N, in_channels, H, W] -> [N, in_channels, (H - (kernel_size - 1) - 1) // kernel_size + 1, (W - (kernel_size - 1) - 1) // kernel_size + 1]
            # e.g. [N, C, 225, 225] -> [N, C, 112, 112]
            nn.MaxPool2d(kernel_size=2),
            # [N, in_channels, H, W] -> [N, out_channels, H, W]
            DoubleResConv(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [N, in_channels, H, W]

        Returns:
            torch.Tensor: [N, out_channels, (H - (kernel_size - 1) - 1) / kernel_size + 1, (W - (kernel_size - 1) - 1) / kernel_size + 1]
        """

        out = self.down(x)
        return out
    
class UpRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """Up part of the net with skip connection: [N, in_channels, H_1, W_1], [N, in_channels // 2, H_2, W_2] -> [N, out_channels, H_2, W_2]

        Args:
            in_channels (int): Channels of input
            out_channels (int): Channels of output
            mid_channels (int, optional): If defined, it will act as the number of channels of the first convolution output. Defaults to None.
        """

        super(UpRes, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels == None:
            mid_channels = out_channels
        self.mid_channels = mid_channels

        # [N, in_channels, H, W] -> [N, in_channels // 2, (H - 1) * stride + kernel_size - 1 + 1, (W - 1) * stride + kernel_size - 1 + 1]
        # e.g. [N, C, 225, 225] -> [N, C, 450, 450]
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2,  stride=2)
        # [N, in_channels, H, W] -> [N, out_channels, H, W]
        self.double_conv = DoubleResConv(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (torch.Tensor): [N, in_channels, H_1, W_1]
            x2 (torch.Tensor): [N, in_channels // 2, H_2, W_2]

        Returns:
            torch.Tensor: [N, out_channels, H_2, W_2]
        """

        x1 = self.up(x1)
        
        # Take the size difference between x1 and x2
        x1_h = x1.shape[-2]
        x1_w = x1.shape[-1]
        x2_h = x2.shape[-2]
        x2_w = x2.shape[-1]

        h_diff = x2_h - x1_h
        w_diff = x2_w - x1_w

        # Pad x1 so x1 has the same size as x2
        pad_h = [h_diff // 2, h_diff - h_diff // 2]
        pad_w = [w_diff // 2, w_diff - w_diff // 2]
        x1 = F.pad(x1, pad_w + pad_h)

        # Concat by channels
        x = torch.cat([x2, x1], dim=1)
        out = self.double_conv(x)

        return out
    
class UNetAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        """Attention part of Attention UNet: [N, channels, H, W] -> [N, channels, H, W]

        Args:
            channels (int): Number of channels
        """

        super(UNetAttentionBlock, self).__init__()

        self.channels = channels

        # [N, channels, H, W] -> [N, channels, H, W]
        self.g_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(num_features=channels)
        )

        # [N, channels, H, W] -> [N, channels, H, W]
        self.x_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.BatchNorm2d(num_features=channels)
        )

        # [N, channels, H, W] -> [N, 1, H, W] with value [0, 1]
        self.psi_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Skip connection input -> [N, channels, H, W]
            g (torch.Tensor): Gate input -> [N, channels, H, W]

        Returns:
            torch.Tensor: Weighted skip connection -> [N, channels, H, W]
        """

        g = self.g_conv(g)
        x = self.x_conv(x)
        psi = self.relu(torch.add(g, x))
        psi = self.psi_conv(psi)
        weighted_x = x * psi

        return weighted_x

class UpAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """Up part of the net with attention block: [N, in_channels, H_1, W_1], [N, in_channels // 2, H_2, W_2] -> [N, out_channels, H_2, W_2]

        Args:
            in_channels (int): Channels of input
            out_channels (int): Channels of output
            mid_channels (int, optional): If defined, it will act as the number of channels of the first convolution output. Defaults to None.
        """

        super(UpAttention, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels == None:
            mid_channels = out_channels
        self.mid_channels = mid_channels

        # [N, in_channels, H, W] -> [N, in_channels // 2, (H - 1) * stride + kernel_size - 1 + 1, (W - 1) * stride + kernel_size - 1 + 1]
        # e.g. [N, C, 225, 225] -> [N, C // 2, 450, 450]
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2,  stride=2)
        # [N, in_channels // 2, H, W] -> [N, in_channels // 2, H, W]
        self.attention_block = UNetAttentionBlock(channels=in_channels // 2)
        # [N, in_channels, H, W] -> [N, out_channels, H, W]
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (torch.Tensor): [N, in_channels, H_1, W_1]
            x2 (torch.Tensor): [N, in_channels // 2, H_2, W_2]

        Returns:
            torch.Tensor: [N, out_channels, H_2, W_2]
        """

        x1 = self.up(x1)
        
        # Take the size difference between x1 and x2
        x1_h = x1.shape[-2]
        x1_w = x1.shape[-1]
        x2_h = x2.shape[-2]
        x2_w = x2.shape[-1]

        h_diff = x2_h - x1_h
        w_diff = x2_w - x1_w

        # Pad x1 so x1 has the same size as x2
        pad_h = [h_diff // 2, h_diff - h_diff // 2]
        pad_w = [w_diff // 2, w_diff - w_diff // 2]
        x1 = F.pad(x1, pad_w + pad_h)

        # Get the weighted x2 using attention block
        x2 = self.attention_block(x2, x1)

        # Concat by channels
        x = torch.cat([x2, x1], dim=1)
        out = self.double_conv(x)

        return out