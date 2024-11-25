import torch
from img_preprocess_infill import ImageDatasetInfill
# from mamba_ssm import Mamba
# import pyarrow.parquet as pq
import os
import glob
from torch import nn
from data_preprocess import MultiDataset
from img_preprocess_generative_embedding_small import ImageDatasetGenerative, TrainMode, Metadata, OutputType
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
# from deepspeed.ops.adam import FusedAdam as DeepSpeedCPUAdam
# from deepspeed import DeepSpeedEngine
import time
from trainer_complicated import TrainLinear, TrainLayer, TrainModel
from text_preprocess import ArticleDataset
from get_embedding import embedding_model
import torch.nn.functional as F
from squarenet import SquareNet, SquareNetHighMemory
import math

# import deepspeed


device = torch.device("cuda:0")
model_name = "img_hans"
out_dir = f"{model_name}/samples"
checkpoint_dir = f"{model_name}/checkpoints"
learning_rate=.0001
image_dim=36
img_output_shape=8

if not os.path.exists(model_name):
    os.mkdir(model_name)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)


device = torch.device("cuda")

# class nn.Linear(nn.Linear):
#     def __init__(self, in_features, out_features, **kwargs):
#         super(nn.Linear, self).__init__(in_features, out_features, **kwargs)
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Kaiming initialization
#         nn.init.zeros_(self.bias)  # Initialize bias to 0

class Sine(nn.Module):
    def forward(self, src: torch.Tensor):
        return F.relu(src).sin()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_in_processing_dim = image_dim**2*3
        image_in = []
        for i in range(5):
            image_in.append(nn.Sequential(*[
                nn.Linear(self.image_in_processing_dim, self.image_in_processing_dim, dtype=torch.bfloat16, bias=False),
                Sine(),
            ]))
        self.image_in = nn.ModuleList(image_in)
        self.text_embedding_dim=1024
        self.master_dim = 1024
        master_layers = [nn.Sequential(*[
            nn.Linear(self.image_in_processing_dim+self.text_embedding_dim,self.image_in_processing_dim+self.text_embedding_dim, dtype=torch.bfloat16, bias=False),
            Sine(),
            nn.Linear(self.image_in_processing_dim+self.text_embedding_dim, self.master_dim, dtype=torch.bfloat16, bias=False),
            Sine(),
        ])]
        for i in range(50):
            master_layers.append(nn.Sequential(*[
                nn.Linear(self.master_dim, self.master_dim, dtype=torch.bfloat16, bias=False),
                Sine(),
            ]))
        
        master_layers.append(nn.Sequential(*[
            nn.Linear(self.master_dim, img_output_shape**2*3, dtype=torch.bfloat16, bias=False)
        ]))
        self.master_layers = nn.ModuleList(master_layers)

        # self.out_layer = nn.Linear(dim*dim*64+1024, img_output_shape*img_output_shape*3, dtype=torch.bfloat16)

    # @torch.compile(mode="reduce-overhead")
    def forward(self, image_input, text_input):
        with torch.no_grad():
            text_input, = embedding_model.encode([text_input])
        text_input = torch.from_numpy(text_input).bfloat16().to(device).reshape(1,-1)
        image_input = image_input.reshape(image_input.shape[0], -1)

        image_processed = image_input
        for layer in self.image_in:
            output = layer(image_processed)
            if output.shape == image_processed.shape:
                image_processed = output + image_processed
            else:
                image_processed = output

        src = torch.cat([image_processed, text_input], dim=-1)

        for layer in self.master_layers:
            output = layer(src)
            if output.shape == src.shape:
                src = output + src
            else:
                src = output
        return src.reshape(src.shape[0], img_output_shape, img_output_shape, 3)

torch.set_float32_matmul_precision('high')
if __name__ == "__main__":

    model = Model().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    checkpoints_sorted = glob.glob(f'{checkpoint_dir}/*.pt')
    if len(checkpoints_sorted) > 0:
        checkpoints_sorted.sort(key=os.path.getmtime)
        print("loading checkpoint", checkpoints_sorted[-1],)
        checkpoint = torch.load(checkpoints_sorted[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_steps = checkpoint['total_steps']
        print("loaded checkpoint", checkpoints_sorted[-1], "total steps", total_steps)

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = target_lr

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print("num trainable params is", params)

    loss_function_text = nn.CrossEntropyLoss()
    loss_function_image = nn.MSELoss()
    writer_dir = f"runs/{model_name}_128_.0001"
    writer = SummaryWriter(log_dir=writer_dir, flush_secs=10)

    dataset = ImageDatasetGenerative(image_dim, img_output_shape) # ImageDatasetInfill()
    # for i, (test_metadata, test_x, test_y) in enumerate(test_dataset, 0):
    #     test_x = test_x.bfloat16()
    #     test_y = test_y.bfloat16()
    #     # test_x = torch.cat([torch.full((test_x.shape[0], text_length), -1, dtype=torch.bfloat16), test_x], dim=-1)
    #     break
    computed_steps = 0
    # total_steps = 0
    for total_steps in range(5000000):
        sample_info = dataset[total_steps%1]
        if sample_info is None:
            continue
        metadata, x_all, y = sample_info
        x_img = x_all[0].bfloat16().to(device) / 255
        x_text = x_all[1]
        y = y.bfloat16().to(device) / 255
        print("max sample info", x_img.max())

        if total_steps % 25 == 0:
            print("doing example")
            for item in glob.glob("test_out/*"):
                os.remove(item)
            image = x_img[-1].reshape(image_dim, image_dim, 3)*255
            img_array = image.cpu().detach().int().numpy().astype(np.uint8)
            Image.fromarray(img_array).save("test_out/main.png")
            for example_part_index in range(0, 4): #x_img.shape[0]):
                with torch.no_grad():
                    # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    prediction = model(x_img[example_part_index:example_part_index+1].to(device),x_text)
                    prediction = prediction.reshape(img_output_shape,img_output_shape,3)*255
                    prediction = prediction.cpu().int().numpy().astype(np.uint8)
                    Image.fromarray(prediction).save(f"test_out/{example_part_index}.png")

                    target = y[example_part_index:example_part_index+1]
                    target = target.reshape(img_output_shape,img_output_shape,3)*255
                    target = target.cpu().int().numpy().astype(np.uint8)
                    Image.fromarray(target).save(f"test_out/{example_part_index}_target.png")


        loss_function = loss_function_image
        for batch_item_index in range(x_img.shape[0]):
            optimizer.zero_grad()
            start = time.time()
            # print(x_img[batch_item_index:batch_item_index+1])
            prediction = model.forward(x_img[batch_item_index:batch_item_index+1].to(device), x_text)
            loss = loss_function(prediction.to(device), y[batch_item_index:batch_item_index+1].to(device))
            writer.add_scalar("main_loss", loss.float().item(), total_steps)
            writer.add_scalar("min", torch.min(prediction).float(), total_steps)
            writer.add_scalar("max", torch.max(prediction).float(), total_steps)
            # torch.nn.utils.clip_grad_norm_()
            if batch_item_index % 8 == 0:
                print(batch_item_index, x_img.shape, x_text)
                print("total steps", total_steps, "batch_item_index", batch_item_index, "loss", loss.item())
                print("output", prediction[0,0,0], prediction.shape)
                print("target", y[batch_item_index:batch_item_index+1,0,0])
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        
        # save every 100,000 examples
        computed_steps += 1
        if computed_steps % 1000 == 0 and total_steps > 2:
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


