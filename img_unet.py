import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, shape):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(shape)
    
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class ImgUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_levels=4, num_convs_per_level=2, base_channels=64, image_size=81, text_embedding_dim=1024):
        super(ImgUNet, self).__init__()
        self.num_levels = num_levels
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.text_embedding_dim = text_embedding_dim

        # Downsampling path
        channels = base_channels
        for i in range(num_levels):
            convs = []
            in_ch = in_channels if i == 0 else channels // 2
            for _ in range(num_convs_per_level):
                convs.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding='same'))
                convs.append(nn.ReLU(inplace=True))
                in_ch = channels
            self.down_convs.append(nn.Sequential(*convs))
            channels *= 2  # Double the channels at each level
        print("channels is", channels)
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

        self.bottleneck_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1536, 1536, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                #LayerNorm(1536),
                nn.Conv2d(1536, 1536, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                LayerNorm(1536),
            ) for _ in range(3)
        ])
        self.bottleneck_convs_out = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            LayerNorm(512),
        )

        # Text embedding expansion
        reduced_size = image_size
        for _ in range(num_levels):
            reduced_size = reduced_size // 3

        # Upsampling path
        for i in range(num_levels):
            channels //= 2
            convs = []
            print("channels is", channels)
            convs.append(nn.Conv2d(channels * 2, channels, kernel_size=3, padding='same'))
            convs.append(nn.ReLU(inplace=True))
            in_ch = channels
            for _ in range(num_convs_per_level - 1):
                convs.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding='same'))
                convs.append(nn.ReLU(inplace=True))
            if i < num_levels - 1:
                convs.append(nn.Conv2d(channels, channels//2, kernel_size=1, padding=0))
                convs.append(nn.ReLU(inplace=True))
            self.up_convs.append(nn.Sequential(*convs))

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, text_embedding):
        skip_connections = []

        # Downsampling path
        for i in range(self.num_levels):
            x = self.down_convs[i](x)
            print(f"After down conv level {i}: {x.shape}")
            skip_connections.append(x)
            x = self.pool(x)
            print(f"After pooling level {i}: {x.shape}")

        # Incorporate text embedding
        text_features = text_embedding.reshape(text_embedding.shape[0], text_embedding.shape[1], 1, 1)
        x = torch.cat([x, text_features], dim=1)
        print(f"After concatenating text embedding: {x.shape}")

        # Bottleneck conv
        for conv in self.bottleneck_convs:
            x = x + conv(x)
        print(f"After bottleneck conv: {x.shape}")
        x = self.bottleneck_convs_out(x)

        # Upsampling path
        # Upsampling path
        for i in range(self.num_levels):
            x = F.interpolate(x, scale_factor=3, mode='nearest')
            skip = skip_connections[-(i+1)]
            print("skip shape", skip.shape, "x shape", x.shape)
            x = torch.cat([x, skip], dim=1)
            print(f"After concatenation with skip connection at level {i}: {x.shape}")
            x = self.up_convs[i](x)
            print(f"After up conv level {i}: {x.shape}")
        # Final output layer
        x = self.final_conv(x)
        print(f"After final conv: {x.shape}")
        return x

if __name__ == "__main__":
    # Test the model
    batch_size = 1
    in_channels = 3
    out_channels = 3
    image_size = 81  # 3**4
    num_levels = 4
    num_convs_per_level = 2
    base_channels = 64
    text_embedding_dim = 1024

    model = ImgUNet(in_channels, out_channels, num_levels, num_convs_per_level, base_channels, image_size, text_embedding_dim)

    x = torch.randn(batch_size, in_channels, image_size, image_size)
    text_embedding = torch.randn(batch_size, text_embedding_dim)
    output = model(x, text_embedding)
    print(f"Output shape: {output.shape}")