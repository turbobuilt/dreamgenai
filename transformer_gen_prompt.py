import torch
import math

import torch.nn as nn
import torch.nn.functional as F

class ImageGenerationModel(nn.Module):
    def __init__(self, d_model=512):
        super(ImageGenerationModel, self).__init__()
        self.d_model = d_model
        self.text_embedding_count = 55

        # Initial 3x3 conv layers with padding='same'
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        # Strided conv to get chunks of the image
        self.stride_conv = nn.Conv2d(512, d_model, kernel_size=3, stride=3)

        # Positional encoding
        self.pos_linear = nn.Linear(2, d_model)

        # Text embedding processing
        self.text_conv1 = nn.Conv1d(1, 1024*self.text_embedding_count, kernel_size=1024, stride=1024, padding=0)
        self.text_conv2 = nn.Conv1d(1024, d_model, kernel_size=1)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, norm_first=True)
            for _ in range(60)
        ])

        # Output linear layer
        self.output_linear = nn.Linear(d_model, 3 * 3 * 3)

    def forward(self, image, text_embeddings):
        batch_size = image.size(0)

        # Initial convolutions
        x = self.conv_layers(image)

        # Strided convolution to get chunks
        x = self.stride_conv(x)  # Shape: (batch_size, d_model, new_h, new_w)

        # Flatten each chunk to 1D
        x = x.flatten(2).permute(0, 2, 1)  # Shape: (batch_size, num_chunks, d_model)

        # Positional encoding
        seq_len = x.size(1)
        grid_size = int(math.sqrt(seq_len))
        positions = torch.stack(torch.meshgrid(
            torch.arange(grid_size), torch.arange(grid_size)), dim=-1)
        positions = positions.reshape(-1, 2).float().to(image.device)
        pos_enc = torch.sin(self.pos_linear(positions))  # Shape: (seq_len, d_model)
        pos_enc = pos_enc.unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + pos_enc
        print(x.shape, "pos_enc", pos_enc.shape)

        # Process text embeddings
        text_embeddings = text_embeddings.view(batch_size, 1, 1024)
        text_embeddings = self.text_conv1(text_embeddings)  # Shape: (batch_size, 55, 1)
        text_embeddings = text_embeddings.reshape(batch_size, 1024, self.text_embedding_count) # Shape: (batch_size, 1024, 55)
        text_embeddings = self.text_conv2(text_embeddings)  # Shape: (batch_size, d_model, 55)
        text_embeddings = text_embeddings.permute(0, 2, 1)  # Shape: (batch_size, 55, d_model)

        print("x shape", x.shape, "text_embeddings shape", text_embeddings.shape)

        # Concatenate image and text tensors
        x = torch.cat([x, text_embeddings], dim=1)  # Shape: (batch_size, 49, d_model)

        # Transformer layers with local and global attention
        seq_len = x.size(1)
        grid_size = int(math.sqrt(seq_len))
        for i, layer in enumerate(self.transformer_layers):
            if i % 3 != 2:
                # Local attention
                x = x.reshape(-1, grid_size, self.d_model)
                x = layer(x)
                # x_local = x[:, :grid_size**2, :].reshape(batch_size, grid_size, grid_size, -1)
                # x_local = x_local.reshape(-1, grid_size, self.d_model)
                # x_local = layer(x_local)
                # x[:, :grid_size**2, :] = x_local.reshape(batch_size, grid_size**2, -1)
            else:
                # Global attention
                x = x.reshape(batch_size, seq_len, self.d_model)
                x = layer(x)

        outputs = x[:, 1, :]
        
        output_section = self.output_linear(outputs).reshape(batch_size, 3, 3, 3)
        print("output_section", output_section.shape)
        return output_section

# Example usage:
if __name__ == "__main__":
    model = ImageGenerationModel()
    image = torch.randn(1, 3, 81, 81)  # Input image of size 3^4 x 3^4
    text_embeddings = torch.randn(1, 1024)
    output_image = model(image, text_embeddings)