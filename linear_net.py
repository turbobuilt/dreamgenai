import torch
import os
import glob
from torch import nn
from linearnet_img import ImageDatasetGenerative, TrainMode, Metadata, OutputType, text_length, img_output_shape, image_dim
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from deepspeed.ops.adam import FusedAdam as DeepSpeedCPUAdam
import time
from get_embedding import embedding_model
import torch.nn.functional as F


device = torch.device("cuda:0")
model_name = "linear_net"
out_dir = f"{model_name}/samples"
checkpoint_dir = f"{model_name}/checkpoints"
test_dir = "test_out_2"

if not os.path.exists(model_name):
    os.mkdir(model_name)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

class Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, dtype=torch.bfloat1, device=device)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, src):
        return self.norm(self.linear(src)+src)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.ModuleList([
            nn.Linear(1024, 1024*2, dtype=torch.bfloat16, device=device),
            nn.LayerNorm(1024*2, dtype=torch.bfloat16, device=device)
        ])
        layers = []
        for i in range(10):
            layers.extend([
                nn.Linear(1024*2, 1024*2, dtype=torch.bfloat16, device=device),
                nn.LayerNorm(1024*2, dtype=torch.bfloat16, device=device)
            ])
        self.layers_mid_1 = nn.ModuleList(layers)
        layers = []
        for i in range(5):
            layers.extend([
                nn.Linear(1024*2+1,1024*2+1, dtype=torch.bfloat16, device=device),
                nn.LayerNorm(1024*2+1, dtype=torch.bfloat16, device=device),
            ])
        self.layers_mid_2 = nn.ModuleList(layers)
        self.layers_mid_3 = nn.Sequential(*[
            nn.Linear(1024*2+1,64*64*3, dtype=torch.bfloat16, device=device),
            nn.ReLU(),
            nn.LayerNorm(64*64*3, dtype=torch.bfloat16, device=device),
        ])
        layers = []
        for i in range(8):
            layers.extend([
                nn.Linear(64*64*3,64*64*3, dtype=torch.bfloat16, device=device),
                nn.LayerNorm(64*64*3, dtype=torch.bfloat16, device=device),
            ])
        self.final_layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(64*64*3, 64*64*3, dtype=torch.bfloat16, device=device)


    #     self.upscale_window_size=5
    #     layers = []
    #     for i in range(7):
    #         layers.extend([
    #             nn.Linear(self.upscale_window_size**2*3,self.upscale_window_size**2*3, dtype=torch.bfloat16, device=device),
    #             nn.LayerNorm(self.upscale_window_size**2*3, dtype=torch.bfloat16, device=device),
    #         ])
    #     self.upscale_layers = nn.ModuleList(*layers)
    #     self.final_upscale_layer = nn.Linear(self.upscale_window_size**2*3, 2*2*3)

    # def upscale(self, src):
    #     src = F.unfold(src, 1, 0, self.upscale_window_size//2)
    #     original = src
    #     for i in range(0,len(self.upscale_layers),2):
    #         src = self.upscale_layers[i+1](F.relu(self.upscale_layers[i]) + original)
    #     src = self.final_upscale_layer(src)
    #     return src
        


        

    # @torch.compile(mode="reduce-overhead")
    def forward(self, src):
        with torch.no_grad():
            src = embedding_model.encode([src])
            src = torch.tensor(src).bfloat16().to(device)
        src = self.layers_1[1](F.relu(self.layers_1[0](src)))

        original = src.clone()
        for i in range(0, len(self.layers_mid_1), 2):
            src = self.layers_mid_1[i+1](F.relu(self.layers_mid_1[i](src))+original)

        random_seed = torch.randn((src.shape[0],1), device=device, dtype=torch.bfloat16)
        src = torch.cat([src, random_seed], dim=-1)

        original = src.clone()
        for i in range(0, len(self.layers_mid_2), 2):
            src = self.layers_mid_2[i+1](F.relu(self.layers_mid_2[i](src))+original)

        src = self.layers_mid_3(src)

        original = src.clone()
        for i in range(0, len(self.final_layers), 2):
            src = self.final_layers[i+1](F.relu(self.final_layers[i](src))+original)

        src = self.final_layer(src)

        return src.reshape(64,64,3)


        

torch.set_float32_matmul_precision('high')
if __name__ == "__main__":

    model = Model().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1)

    checkpoints_sorted = glob.glob(f'{checkpoint_dir}/*.pt')
    if len(checkpoints_sorted) > 0:
        checkpoints_sorted.sort(key=os.path.getmtime)
        print("loading checkpoint", checkpoints_sorted[-1],)
        checkpoint = torch.load(checkpoints_sorted[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_steps = checkpoint['total_steps']
        print("loaded checkpoint", checkpoints_sorted[-1], "total steps", total_steps)

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print("num trainable params is", params)

    loss_function_text = nn.CrossEntropyLoss()
    loss_function_image = nn.L1Loss()
    writer = SummaryWriter(log_dir=f"runs/{model_name}_128_.0001", flush_secs=10)

    dataset = ImageDatasetGenerative()
    computed_steps = 0
    for total_steps in range(5000000):
        sample_info = dataset[total_steps]
        if sample_info is None:
            print("skipping", total_steps)
            continue
        x, y = sample_info
        y = torch.tensor(y).bfloat16().to(device)

        optimizer.zero_grad()
        start = time.time()
        prediction = model.forward(x)
        loss = loss_function_image(prediction.to(device), y.to(device))
        writer.add_scalar("main_loss", loss.float().item(), total_steps)
        writer.add_scalar("min", torch.min(prediction).float(), total_steps)
        writer.add_scalar("max", torch.max(prediction).float(), total_steps)

        if computed_steps % 50 == 0:
            print("total steps", total_steps, "loss", loss.item())
            print("output", prediction[0,0], prediction.shape)
            print("target", y[0,0])

        if computed_steps % 200 == 0:
            print("printing prediction")
            prediction = prediction*255
            prediction = prediction.cpu().int().numpy().astype(np.uint8)
            Image.fromarray(prediction).save(f"test_out_2/prediction.png")

            target = y*255
            target = target.cpu().int().numpy().astype(np.uint8)
            Image.fromarray(target).save(f"test_out_2/target.png")

            with open(f"test_out_2/text.txt", "w") as f:
                f.write(x)

        loss.backward()
        optimizer.step()
        
        # save every 100,000 examples
        computed_steps += 1
        if computed_steps % 4000 == 0 and total_steps > 2:
            print("saving expoch")
            old_checkpoints = glob.glob(f"{checkpoint_dir}/*")
            old_checkpoints.sort(key=os.path.getmtime)
            for f in old_checkpoints[:-1]:
                os.remove(f)
            torch.save({
                'total_steps': total_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{checkpoint_dir}/{total_steps}.pt")


