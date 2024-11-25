import torch
import torch.nn as nn
import torch.nn.functional as F

from img_unet_loader import ImgUnetLoader

dtype = torch.bfloat16
device = torch.device("cuda:0")

class LayerNorm(nn.Module):
    def __init__(self, shape):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(shape, dtype=dtype, device=device)
    
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
                convs.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding='same', dtype=dtype, device=device))
                convs.append(nn.ReLU(inplace=True))
                in_ch = channels
            self.down_convs.append(nn.Sequential(*convs))
            channels *= 2  # Double the channels at each level
        # print("channels is", channels)
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

        self.bottleneck_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512+text_embedding_dim, 512+text_embedding_dim, kernel_size=1, padding=0, dtype=dtype, device=device),
                nn.ReLU(inplace=True),
                #LayerNorm(512+text_embedding_dim),
                nn.Conv2d(512+text_embedding_dim, 512+text_embedding_dim, kernel_size=1, padding=0, dtype=dtype, device=device),
                nn.ReLU(inplace=True),
                LayerNorm(512+text_embedding_dim),
            ) for _ in range(3)
        ])
        self.bottleneck_convs_out = nn.Sequential(
            nn.Conv2d(512+text_embedding_dim, 512, kernel_size=1, padding=0, dtype=dtype, device=device),
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
            # print("channels is", channels)
            convs.append(nn.Conv2d(channels * 2, channels, kernel_size=3, padding='same', dtype=dtype, device=device))
            convs.append(nn.ReLU(inplace=True))
            in_ch = channels
            for _ in range(num_convs_per_level - 1):
                convs.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding='same', dtype=dtype, device=device))
                convs.append(nn.ReLU(inplace=True))
            if i < num_levels - 1:
                convs.append(nn.Conv2d(channels, channels//2, kernel_size=1, padding=0, dtype=dtype, device=device))
                convs.append(nn.ReLU(inplace=True))
            self.up_convs.append(nn.Sequential(*convs))

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1, dtype=dtype, device=device)

    def forward(self, x, text_embedding):
        skip_connections = []

        # Downsampling path
        for i in range(self.num_levels):
            x = self.down_convs[i](x)
            # print(f"After down conv level {i}: {x.shape}")
            skip_connections.append(x)
            x = self.pool(x)
            # print(f"After pooling level {i}: {x.shape}")

        # Incorporate text embedding
        text_features = text_embedding.reshape(text_embedding.shape[0], text_embedding.shape[1], 1, 1)
        x = torch.cat([x, text_features], dim=1)
        # print(f"After concatenating text embedding: {x.shape}")

        # Bottleneck conv
        for conv in self.bottleneck_convs:
            x = x + conv(x)
        # print(f"After bottleneck conv: {x.shape}")
        x = self.bottleneck_convs_out(x)

        # Upsampling path
        # Upsampling path
        for i in range(self.num_levels):
            x = F.interpolate(x, scale_factor=3, mode='nearest')
            skip = skip_connections[-(i+1)]
            # print("skip shape", skip.shape, "x shape", x.shape)
            x = torch.cat([x, skip], dim=1)
            # print(f"After concatenation with skip connection at level {i}: {x.shape}")
            x = self.up_convs[i](x)
            # print(f"After up conv level {i}: {x.shape}")
        # Final output layer
        x = self.final_conv(x)
        # print(f"After final conv: {x.shape}")
        return x
    
model_name = "img_unet"
# create output dir
output_dir = f"output/{model_name}"
import os
os.makedirs(output_dir, exist_ok=True)
# clear dir
for file in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, file))

if __name__ == "__main__":
    # Test the model
    batch_size = 1
    in_channels = 3
    out_channels = 3
    image_size = 81  # 3**4
    num_levels = 4
    num_convs_per_level = 2
    base_channels = 64
    text_embedding_dim = 768

    model = ImgUNet(in_channels, out_channels, num_levels, num_convs_per_level, base_channels, image_size, text_embedding_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    #print parameter count
    print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")

    data_loader = ImgUnetLoader()
    total_steps = 0
    epoch = 0
    for epoch in range(epoch, 100):
        step = 0
        for inputs, targets, text_embedding, text in data_loader:
            step += 1
            total_steps += 1
            inputs = inputs.bfloat16().to(device)
            targets = targets.bfloat16().to(device)
            text_embedding = text_embedding.bfloat16().to(device)
            for i in range(len(inputs)):
                # text_embedding = torch.from_numpy(text_embedding.reshape(1, text_embedding_embedding_dim))
                optimizer.zero_grad()
                output = model(inputs[i:i+1], text_embedding.reshape(1, text_embedding_dim))
                loss = F.mse_loss(output, targets[i:i+1])
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    print(f"epoch: {epoch}, step: {step}, i: {i}/{inputs.shape[0]}, loss: {loss.item()}, ")
                if step == 10:
                    # save input/output for comparison using pillow
                    import numpy as np
                    input_img = inputs[i].float().cpu().numpy().transpose(1, 2, 0)
                    output_img = output[0].float().cpu().detach().numpy().transpose(1, 2, 0)
                    target_img = targets[i].float().cpu().numpy().transpose(1, 2, 0)
                    from PIL import Image
                    Image.fromarray((input_img * 255).astype(np.uint8)).save(f"{output_dir}/{total_steps}_{step}_{i}_input.png")
                    Image.fromarray((output_img * 255).astype(np.uint8)).save(f"{output_dir}/{total_steps}_{step}_{i}_output.png")
                    Image.fromarray((target_img * 255).astype(np.uint8)).save(f"{output_dir}/{total_steps}_{step}_{i}_target.png")
                    # write text
                    with open(f"{output_dir}/{total_steps}_text.txt", "w") as f:
                        f.write(text)

    # x = torch.randn(batch_size, in_channels, image_size, image_size)
    # text_embedding = torch.randn(batch_size, text_embedding_dim)
    # output = model(x, text_embedding)
    # print(f"Output shape: {output.shape}")