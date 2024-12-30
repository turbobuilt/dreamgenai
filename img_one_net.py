import glob
import json
import os
import torch
from PIL import Image
import shutil

import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from img_one_net_loader import create_loader
# import tensorboard
from torch.utils.tensorboard import SummaryWriter

import os
import math

image_dim = 32
out_convs_length = 3
embedding_dim = 384

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm_1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        r = self.norm_1(x)
        x = self.norm_2(self.act(self.fc1(x)))
        x = self.act(self.fc2(x))
        return x + r
    

class ImgOneNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_in = nn.Linear(embedding_dim, 1024)
        self.res1 = ResidualBlock(1024)
        self.res2 = ResidualBlock(1024)
        self.individualizer_pre_norm = nn.LayerNorm(1024)
        self.out_positions = (image_dim + 2 * out_convs_length)**2
        # self.individualizer = nn.Conv1d(self.out_positions, self.out_positions*32, kernel_size=1024, padding=0, groups=self.out_positions, bias=False)
        self.individualizer = nn.Linear(1024, self.out_positions*32)
        self.out_conv_channels = 32

        self.out_convs = nn.Sequential(
            nn.BatchNorm2d(self.out_conv_channels),
            nn.Conv2d(self.out_conv_channels, self.out_conv_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_conv_channels),
            nn.Conv2d(self.out_conv_channels, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=0)
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_emb):
        batch_size = text_emb.shape[0]
        x = self.act(self.lin_in(text_emb))
        x = self.res1(x)
        x = self.dropout(x)
        x = self.res2(x)
        
        x = x.reshape(batch_size, 1, 1024)
        x = self.individualizer_pre_norm(x)
        x = self.individualizer(x).reshape(batch_size, self.out_conv_channels, image_dim + 2 * out_convs_length, image_dim + 2 * out_convs_length)
        x = self.out_convs(x)
        return x

if __name__ == "__main__":
    model_name = ImgOneNet.__name__
    out_dir = f"{model_name}_out"
    checkpoint_dir = f"{model_name}_checkpoints"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(f"{model_name}_logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImgOneNet().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {param_count:,}")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    data_loader = create_loader(image_dim)
    total_steps = 0
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

        

    for epoch in range(epoch, 100):
        for step, (img, text_emb, text) in enumerate(data_loader, step):
            total_steps += 1
            # learning rate halves every 5000 steps
            learning_rate = 1e-3 * 0.5**(total_steps // 5000)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            img, text_emb = img.to(device), text_emb.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(text_emb)
                loss = loss_fn(out, img)
                loss.backward()
                optimizer.step()
            
            if step % 10 == 0:
                print(f"epoch {epoch} Step {step}: total steps: {total_steps} Loss: {loss.item()} lr: {learning_rate}")

            if step % 100 == 0:
                writer.add_scalar("Loss", loss.item(), total_steps)
                writer.add_scalar("Learning Rate", learning_rate, total_steps)
                vutils.save_image(img, f"{out_dir}/{step}_target.png", normalize=True)
                vutils.save_image(out, f"{out_dir}/{step}_output.png", normalize=True)
                with open(f"{out_dir}/{step}_text.txt", "w") as f:
                    f.write(json.dumps(text, indent=4))

                files = sorted(
                [f for f in os.listdir(out_dir) if f.endswith(".png") or f.endswith(".txt")],
                    key=lambda x: os.path.getmtime(os.path.join(out_dir, x))
                )
                excess = len(files) - 150
                for old_file in files[:excess if excess > 0 else 0]:
                    os.remove(os.path.join(out_dir, old_file))
            
            if total_steps % 1000 == 0:
                # save checkpoint including total_steps, epoch, and step
                print(f"Saving checkpoint at step {total_steps}")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "total_steps": total_steps,
                    "epoch": epoch,
                    "step": step
                }, f"{checkpoint_dir}/checkpoint_{total_steps}.pt")
                # delete all but last 2
                checkpoints = sorted(
                    [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")],
                    key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x))
                )
                excess = len(checkpoints) - 2
                for old_checkpoint in checkpoints[:excess if excess > 0 else 0]:
                    os.remove(os.path.join(checkpoint_dir, old_checkpoint))