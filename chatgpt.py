import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, input_dim=1024, image_dim=64, num_layers=4, num_filters=64):
        super(UNet, self).__init__()
        
        self.input_dim = input_dim
        self.image_dim = image_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        
        # Initial fully connected layer to project the input to a feature map
        self.fc = nn.Linear(input_dim, num_filters * (image_dim // 2**num_layers) * (image_dim // 2**num_layers))
        
        # Encoder layers
        self.encoders = nn.ModuleList()
        for i in range(num_layers):
            in_channels = num_filters if i == 0 else num_filters * (2 ** (i - 1))
            out_channels = num_filters * (2 ** i)
            self.encoders.append(self._conv_block(in_channels, out_channels))
        
        # Bottleneck layer
        self.bottleneck = self._conv_block(num_filters * (2 ** (num_layers - 1)), num_filters * (2 ** num_layers))
        
        # Decoder layers
        self.decoders = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            in_channels = num_filters * (2 ** (i + 1))
            out_channels = num_filters * (2 ** i)
            self.decoders.append(self._upconv_block(in_channels, out_channels))
        
        # Final layer to output the image
        self.final_conv = nn.Conv2d(num_filters, 3, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x is of shape [batch_size, 1024]
        x = self.fc(x)  # Fully connected layer
        x = x.view(-1, self.num_filters, self.image_dim // 2**self.num_layers, self.image_dim // 2**self.num_layers)  # Reshape
        
        # Encoder path
        encoder_outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
            # Skip connection
            if i < len(encoder_outputs):
                x = x + encoder_outputs[-(i + 1)]  # Skip connection
        
        # Final convolution to get the output image
        x = self.final_conv(x)
        
        return x  # Output shape will be [batch_size, 3, image_dim, image_dim]

# Example usage
if __name__ == "__main__":
    batch_size = 8
    image_dim = 64
    model = UNet(image_dim=image_dim, num_layers=4, num_filters=64)
    
    # Create a random input tensor with shape [batch_size, 1024]
    input_tensor = torch.randn(batch_size, 1024)
    
    # Get the output image
    output_image = model(input_tensor)
    print(output_image.shape)  # Should be [batch_size, 3, image_dim, image_dim]