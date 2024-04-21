import torch
from img_preprocess_infill import ImageDatasetInfill
from mamba_ssm import Mamba
import pyarrow.parquet as pq
import os
import glob
from torch import nn
from data_preprocess import MultiDataset
from img_preprocess_generative_embedding import ImageDatasetGenerative, TrainMode, Metadata, OutputType, text_length, img_output_shape, image_dim
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from deepspeed.ops.adam import FusedAdam as DeepSpeedCPUAdam
from deepspeed import DeepSpeedEngine
import time
from trainer_complicated import TrainLinear, TrainLayer, TrainModel
from text_preprocess import ArticleDataset
from get_embedding import embedding_model
import torch.nn.functional as F
from squarenet import SquareNet

import deepspeed

# Define the DeepSpeed configuration with ZeRO optimization
ds_config = {
    "train_batch_size": 1,  # Example, adjust based on your GPU memory and model size
    "fp16": {
        "enabled": True,  # Enable mixed precision training
    },
    # "zero_optimization": {
    #     "stage": 2,  # Using ZeRO Stage 2 as an example, adjust as needed
    #     "offload_optimizer": {
    #         "device": "cpu",  # Offload optimizer state to CPU
    #         "pin_memory": True
    #     },
    #     "allgather_partitions": True,
    #     "allgather_bucket_size": 2e6,
    #     "reduce_scatter": True,
    #     "reduce_bucket_size": 2e6,
    #     "overlap_comm": False,
    #     "contiguous_gradients": True,
    # },
    "optimizer": {
        "type": "OneBitAdam",
        "params": {
            "lr": 0.0001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.001,  # Adjust weight decay to your needs
        }
    }
}


device = torch.device("cuda:0")
model_name = "imagen_yeshua"
out_dir = f"{model_name}/samples"
checkpoint_dir = f"{model_name}/checkpoints"



# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=text_length, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
device = torch.device("cuda")

class Layer(nn.Module):
    def __init__(self, parent_model, in_dim, out_dim, kernel_size=4, activation=nn.functional.relu, add=False):
        super().__init__()
        self.conv = nn.Conv1d(1, out_channels=kernel_size, kernel_size=kernel_size, stride=kernel_size, device=device, dtype=torch.bfloat16, bias=True)
        self.layer_norm_1 = nn.LayerNorm(in_dim)
        self.in_dim = in_dim
        # self.linear_1 = nn.Linear(in_dim, in_dim, device=device, dtype=torch.bfloat16, bias=False)
        # self.linear_1.weight.data.uniform_(-1, 1)  # Initializes weights with uniform random values between -1 and 1
        # self.linear_1.weight.data /= out_dim*2
        self.linear_2 = nn.Linear(in_dim, out_dim, device=device, dtype=torch.bfloat16, bias=True)
        self.linear_2.weight.data.uniform_(-1, 1)  # Initializes weights with uniform random values between -1 and 1
        # self.linear_2.weight.data /= out_dim*2
        self.out_dim = out_dim
        self.layer_norm_2 = nn.LayerNorm(out_dim)
        self.activation = activation
        self.add = add

    # @torch.compile(mode="max-autotune")
    def forward(self, src, previous):
        src = self.conv(src.reshape(src.shape[0], 1, src.shape[1]))
        if self.activation is not None:
            src = self.layer_norm_1(self.activation(src).reshape(src.shape[0], -1))
        src = self.linear_2(src)
        if src.shape[-1] == previous.shape[-1]:
            src = src + previous
        # src = self.activation(src)
        src = torch.nn.functional.sigmoid(src)
            # if torch.isnan(src).any():
            #     print("SRC", src[0], "previous", previous[0])
            #     exit()
        # if self.activation is not None:
        src = self.layer_norm_2(src)
            # if torch.isnan(src).any():
            #     print("SRC", src[0], "previous", previous[0])
            #     exit()
        return src
    

class SigmoidOut(nn.Module):
    def __init__(self, parent_model, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, device=device, dtype=torch.bfloat16, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    # @torch.compile(mode="max-autotune")
    def forward(self, src):
        src = self.linear(src)
        src = self.sigmoid(src)
        return src.reshape(-1, img_output_shape, img_output_shape, 3)
    
class LinearWithActivation(nn.Module):
    def __init__(self, in_features=None, out_features=None, dtype=torch.bfloat16):
        super().__init__()
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features=out_features, dtype=dtype)
        self.norm = nn.LayerNorm(out_features, dtype=dtype)
    
    def forward(self, src, original=None):
        if original == None:
            original = 0
        return self.norm(F.relu(self.linear(src))+original)
    
class ImagePreprocessLayerGroup(nn.Module):
    def __init__(self, in_dim, in_channels, out_channels=None, n_layers=4):
        super().__init__()
        layers = []
        if out_channels is None:
            out_channels = in_channels // 2
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, dtype=torch.bfloat16),)
        layers.append(nn.BatchNorm2d(out_channels, dtype=torch.bfloat16))
        for i in range(n_layers-1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, dtype=torch.bfloat16))
            layers.append(nn.BatchNorm2d(out_channels, dtype=torch.bfloat16))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, dtype=torch.bfloat16))
        layers.append(nn.BatchNorm2d(out_channels, dtype=torch.bfloat16))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, src):
        src = self.layers[1](F.relu(self.layers[0](src)))
        original = src.clone()
        for i in range(2, len(self.layers)-2, 2):
            src = self.layers[i+1](F.relu(self.layers[i](src)) + original)
        src = self.layers[-1](self.layers[-2](src))
        return src

    
# class ImagePreprocess(nn.Module):
#     def __init__(self, in_dim=64, layers_per_group=4):
#         super().__init__()
#         layers = []
#         current_dim = in_dim
#         layers.append(ImagePreprocessLayerGroup(current_dim, 1, in_dim, in_dim, layers_per_group))
#         for i in range(7): # drop to 1024 data points
#             layers.append(ImagePreprocessLayerGroup(current_dim, in_dim, in_dim, layers_per_group))

    # def forward(self, src):

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_in_processing_dim = image_dim**2*3
        self.image_in = ImagePreprocessLayerGroup(image_dim, 3, 64)
        
        self.text_embedding_dim=1024
        self.text_in = nn.ModuleList([
            LinearWithActivation(self.text_embedding_dim, self.text_embedding_dim*4),
            *[LinearWithActivation(self.text_embedding_dim*4, self.text_embedding_dim*4) for _ in range(15)],
            nn.Linear(self.text_embedding_dim*4, self.text_embedding_dim, dtype=torch.bfloat16),
        ])
        dim = image_dim // 2
        self.master_layers = nn.ModuleList([
            *[SquareNet(dim*dim*64+1024) for _ in range(150)]
        ])

        self.out_layer = nn.Linear(dim*dim*64+1024, img_output_shape*img_output_shape*3, dtype=torch.bfloat16)

    # @torch.compile(mode="reduce-overhead")
    def forward(self, image_input, text_input, example_index=None):
        with torch.no_grad():
            text_input, = embedding_model.encode([text_input])

        text_input = torch.tensor(text_input).bfloat16().to(device).reshape(image_input.shape[0], -1)
        text_input = self.text_in[0](text_input)
        original = text_input.clone()
        for layer_index in range(1, len(self.text_in)-1):
            text_input = self.text_in[layer_index](text_input, original)
        text_input = self.text_in[-1](text_input)
        image_input = image_input.permute(0,3,1,2)
        image_input = self.image_in(image_input).reshape(image_input.shape[0], -1)
        src = torch.cat([image_input, text_input], dim=-1).reshape(image_input.shape[0], -1)
        original = src.clone()
        for layer in self.master_layers:
            src = layer(src) + original
            # original = original + src
        return self.out_layer(src).reshape(src.shape[0], img_output_shape, img_output_shape, 3)

    
if __name__ == "__main__":
    epoch = 0
    example_index = 0
    total_steps = 0

    model = Model().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001)

    loss_function_text = nn.CrossEntropyLoss()
    loss_function_image = nn.MSELoss()
    writer = SummaryWriter(log_dir=f"runs/{model_name}_128_.0001", flush_secs=10)

    dataset = ImageDatasetGenerative() # ImageDatasetInfill()
    # for i, (test_metadata, test_x, test_y) in enumerate(test_dataset, 0):
    #     test_x = test_x.bfloat16()
    #     test_y = test_y.bfloat16()
    #     # test_x = torch.cat([torch.full((test_x.shape[0], text_length), -1, dtype=torch.bfloat16), test_x], dim=-1)
    #     break
    total_steps = -1
    for i in range(5000):
        total_steps += 1
        metadata, x_all, y = dataset[0]
        x_img = x_all[0].bfloat16().to(device)
        x_text = x_all[1]
        y = y.bfloat16().to(device)

        print("example index", example_index, "total steps", total_steps)
        if total_steps % 10 == 0 and total_steps > 0:
            print("doing example")
            for item in glob.glob("test_out/*"):
                os.remove(item)
            image = x_img[-1].reshape(image_dim, image_dim, 3)
            img_array = image.cpu().detach().int().numpy().astype(np.uint8)
            Image.fromarray(img_array).save("test_out/main.png")
            for example_part_index in range(0, x_img.shape[0]):
                with torch.no_grad():
                    # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prediction = model(x_img[example_part_index:example_part_index+1].to(device),x_text)
                    prediction = prediction.reshape(img_output_shape,img_output_shape,3)
                    prediction = prediction.cpu().int().numpy().astype(np.uint8)
                    Image.fromarray(prediction).save(f"test_out/{example_part_index}.png")

                    target = y[example_part_index:example_part_index+1]
                    target = target.reshape(img_output_shape,img_output_shape,3)
                    target = target.cpu().int().numpy().astype(np.uint8)
                    Image.fromarray(target).save(f"test_out/{example_part_index}_target.png")


        loss_function = loss_function_image
        for batch_item_index in range(x_img.shape[0]):
            optimizer.zero_grad()
            start = time.time()
            prediction = model.forward(x_img[batch_item_index:batch_item_index+1].to(device), x_text)
            loss = loss_function(prediction.to(device), y[batch_item_index:batch_item_index+1].to(device))
            writer.add_scalar("main_loss", loss.float().item(), total_steps)
            writer.add_scalar("min", torch.min(prediction).float(), total_steps)
            writer.add_scalar("max", torch.max(prediction).float(), total_steps)
            if batch_item_index % 8 == 0:
                print(batch_item_index, x_img.shape, x_text)
                print("total steps", total_steps, "batch_item_index", batch_item_index, "loss", loss.item())
                print("output", prediction[0,0,0], prediction.shape)
                print("target", y[batch_item_index:batch_item_index+1,0,0])
            loss.backward()
            optimizer.step()
        
        # save every 100,000 examples, include epoch and example_index in save file
        if example_index % 100000 == 0 and example_index > 0 and False:
            print("not saving expoch")
            # model.save_checkpoint(f"{checkpoint_dir}/{total_steps}_epoch_{epoch}_example_{example_index}.pt", client_state={
            #     'epoch': epoch,
            #     'example_index': example_index,
            #     'total_steps': total_steps
            # })
            # torch.save({
            #     'epoch': epoch,
            #     'example_index': example_index,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'total_steps': total_steps
            # }, f"{checkpoint_dir}/epoch_{epoch}_example_{example_index}.pt")


