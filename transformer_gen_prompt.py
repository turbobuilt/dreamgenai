import torch
import math

import torch.nn as nn
import torch.nn.functional as F

dtype = torch.bfloat16
device = torch.device("cuda:0")

class ImageGenerationModel(nn.Module):
    def __init__(self, d_model=256):
        super(ImageGenerationModel, self).__init__()
        self.d_model = d_model
        self.text_embedding_count = 55

        # Initial 3x3 conv layers with padding='same'
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding='same', device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding='same', device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same', device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding='same', device=device, dtype=dtype),
            nn.ReLU(),
        )

        # Strided conv to get chunks of the image
        self.stride_conv = nn.Conv2d(256, d_model, kernel_size=3, stride=3, device=device, dtype=dtype)

        # Positional encoding
        self.pos_linear = nn.Linear(2, d_model, device=device, dtype=dtype)

        # Text embedding processing
        self.text_conv1 = nn.Conv1d(1, 768*self.text_embedding_count, kernel_size=768, stride=768, padding=0, device=device, dtype=dtype)
        self.text_conv2 = nn.Conv1d(768, d_model, kernel_size=1, device=device, dtype=dtype)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, norm_first=True, device=device, dtype=dtype)
            for _ in range(30)
        ])

        # Output linear layer
        self.output_linear = nn.Linear(d_model, 9 * 9 * 3, device=device, dtype=dtype)

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
            torch.arange(grid_size, device=device, dtype=dtype), torch.arange(grid_size, device=device, dtype=dtype)), dim=-1)
        positions = positions.reshape(-1, 2).bfloat16().to(image.device)
        pos_enc = torch.sin(self.pos_linear(positions))  # Shape: (seq_len, d_model)
        pos_enc = pos_enc.unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + pos_enc

        # Process text embeddings
        text_embeddings = text_embeddings.view(batch_size, 1, 768)
        text_embeddings = self.text_conv1(text_embeddings)  # Shape: (batch_size, 55, 1)
        text_embeddings = text_embeddings.reshape(batch_size, 768, self.text_embedding_count) # Shape: (batch_size, 1024, 55)
        text_embeddings = self.text_conv2(text_embeddings)  # Shape: (batch_size, d_model, 55)
        text_embeddings = text_embeddings.permute(0, 2, 1)  # Shape: (batch_size, 55, d_model)


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
        
        output_section = self.output_linear(outputs).reshape(batch_size, 3, 9, 9)
        return output_section

# # Example usage:
# if __name__ == "__main__":
#     model = ImageGenerationModel()
#     image = torch.randn(1, 3, 81, 81)  # Input image of size 3^4 x 3^4
#     text_embeddings = torch.randn(1, 1024)
#     output_image = model(image, text_embeddings)


    import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
from img_unet_loader import ImgUnetLoader
from torch.utils.tensorboard import SummaryWriter

    
model_name = "transformer_gen_prompt"
# create output dir
output_dir = f"{model_name}_output"
import os
os.makedirs(output_dir, exist_ok=True)
checkpoint_dir = f"{model_name}_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
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

    model = ImageGenerationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(f"runs/{model_name}")
    #print parameter count
    print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")
    # scaler = torch.GradScaler()
    data_loader = ImgUnetLoader()
    total_steps = -1
    epoch = 0
    step = 0

    # load checkpoint
    checkpoints_sorted = glob.glob(f'{checkpoint_dir}/*.pt')
    if len(checkpoints_sorted) > 0:
        checkpoints_sorted.sort(key=os.path.getmtime)
        print("loading checkpoint", checkpoints_sorted[-1],)
        checkpoint = torch.load(checkpoints_sorted[-1])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        total_steps = checkpoint['total_steps']
        print("loaded checkpoint", checkpoints_sorted[-1], "total steps", total_steps)


    for epoch in range(epoch, 5000):
        step = 0
        for inputs, targets, text_embedding, text in data_loader:
            step += 1
            # inputs = inputs.bfloat16().to(device)
            # targets = targets.bfloat16().to(device)
            text_embedding = text_embedding.bfloat16().to(device)
            for i in range(len(inputs)):
                total_steps += 1
                # text_embedding = torch.from_numpy(text_embedding.reshape(1, text_embedding_embedding_dim))
                optimizer.zero_grad()
                # with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(inputs[i:i+1].bfloat16().to(device), text_embedding.reshape(1, text_embedding_dim))
                # print("output is", output[0, 0, 0, :], output.shape)
                # loss = F.mse_loss(output, targets[i:i+1])
                loss = F.mse_loss(output, targets[i:i+1].bfloat16().to(device))
                loss.backward()
                optimizer.step()

                # scaler.scale(loss).backward()
                # scaler.unscale_(optimizer)
                # scaler.step(optimizer)
                # scaler.update()

                if i % 10 == 0:
                    print("output is", output[0, 0, 0, :], output.shape)
                    print(f"epoch: {epoch}, step: {step}, i: {i}/{inputs.shape[0]}, loss: {loss.item()}, total_steps: {total_steps}")
                    writer.add_scalar("loss", loss.item(), total_steps)
                if i == 9 or i == 12:
                    # save input/output for comparison using pillow
                    print("saving")
                    import numpy as np
                    input_img = inputs[i].float().cpu().numpy().transpose(1, 2, 0)
                    output_img = output[0].float().cpu().detach().numpy().transpose(1, 2, 0)
                    target_img = targets[i].float().cpu().numpy().transpose(1, 2, 0)
                    from PIL import Image
                    Image.fromarray((input_img * 255).astype(np.uint8)).save(f"{output_dir}/{step}_{i}_input.png")
                    Image.fromarray((output_img * 255).astype(np.uint8)).save(f"{output_dir}/{step}_{i}_output.png")
                    Image.fromarray((target_img * 255).astype(np.uint8)).save(f"{output_dir}/{step}_{i}_target.png")
                    # write text
                    with open(f"{output_dir}/{i}_text.txt", "w") as f:
                        f.write(text)
                if total_steps % 10000 == 0 and total_steps >= 0:
                    print("saving model!")
                    torch.save({
                        "total_steps": total_steps,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": step,
                    }, f"{checkpoint_dir}/model_{total_steps}.pt")
                    # delete all but last 2
                    checkpoints = os.listdir(checkpoint_dir)
                    checkpoints_sorted = sorted(checkpoints)
                    for checkpoint in checkpoints_sorted[:-2]:
                        os.remove(f"{checkpoint_dir}/{checkpoint}")
                    print("saved model")
                    # break

    # x = torch.randn(batch_size, in_channels, image_size, image_size)
    # text_embedding = torch.randn(batch_size, text_embedding_dim)
    # output = model(x, text_embedding)
    # print(f"Output shape: {output.shape}")