import torch
from img_preprocess_infill import ImageDatasetInfill
# from mamba_ssm import Mamba
# import pyarrow.parquet as pq
import os
import glob
from torch import nn
from data_preprocess import MultiDataset
from img_preprocess_generative_embedding_small import ImageDatasetGenerative, TrainMode, Metadata, OutputType, text_length
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

# import deepspeed


device = torch.device("cuda:0")
model_name = "imagen_yeshua_small"
out_dir = f"{model_name}/samples"
checkpoint_dir = f"{model_name}/checkpoints"
learning_rate=0.0001

if not os.path.exists(model_name):
    os.mkdir(model_name)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)


device = torch.device("cuda")

class LinearWithActivation(nn.Module):
    def __init__(self, in_features=None, out_features=None, dtype=torch.bfloat16):
        super().__init__()
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features=out_features, dtype=dtype)
        self.norm = nn.LayerNorm(out_features, dtype=dtype)
    
    def forward(self, src):
        return self.norm(F.relu(self.linear(src)))
    
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
        for i in range(2, len(self.layers)-2, 2):
            src = self.layers[i+1](F.relu(self.layers[i](src)))
        src = self.layers[-1](self.layers[-2](src))
        return src

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_in_processing_dim = image_dim**2*3
        self.image_in = ImagePreprocessLayerGroup(image_dim, 3, 64)
        
        self.text_embedding_dim=1024
        self.text_in = nn.ModuleList([
            LinearWithActivation(self.text_embedding_dim, self.text_embedding_dim*4),
            *[LinearWithActivation(self.text_embedding_dim*4, self.text_embedding_dim*4) for _ in range(5)],
            nn.Linear(self.text_embedding_dim*4, self.text_embedding_dim, dtype=torch.bfloat16),
        ])
        dim = image_dim // 2
        self.master_layers = nn.ModuleList([
            *[SquareNetHighMemory(dim*dim*64+1024, i, device=device) for i in range(5)],
            # *[SquareNet(dim*dim*64+1024, device=device) for i in range(125)]
        ])

        decoder_dims = (dim*dim*64+1024) / img_output_shape**2
        decoder_dims = (dim*dim*64+1024) // img_output_shape**2
        self.decoder = nn.Sequential(
            nn.Linear(decoder_dims, decoder_dims, dtype=torch.bfloat16, device=device),
            nn.ReLU(),
            nn.LayerNorm(decoder_dims, dtype=torch.bfloat16, device=device),
            nn.Linear(decoder_dims, decoder_dims, dtype=torch.bfloat16, device=device),
            nn.ReLU(),
            nn.LayerNorm(decoder_dims, dtype=torch.bfloat16, device=device),
            nn.Linear(decoder_dims, 3, dtype=torch.bfloat16, device=device)
        )

        # self.out_layer = nn.Linear(dim*dim*64+1024, img_output_shape*img_output_shape*3, dtype=torch.bfloat16)

    # @torch.compile(mode="reduce-overhead")
    def forward(self, image_input, text_input):
        with torch.no_grad():
            text_input, = embedding_model.encode([text_input])

        text_input = torch.tensor(text_input).bfloat16().to(device).reshape(image_input.shape[0], -1)
        text_input = self.text_in[0](text_input)
        for layer_index in range(1, len(self.text_in)-1):
            text_input = self.text_in[layer_index](text_input)
        text_input = self.text_in[-1](text_input)
        image_input = image_input.permute(0,3,1,2)
        image_input = self.image_in(image_input).reshape(image_input.shape[0], -1)
        src = torch.cat([image_input, text_input], dim=-1).reshape(image_input.shape[0], -1)
        for layer in self.master_layers:
            src = layer(src)
        src = src.reshape(src.shape[0], img_output_shape**2, -1)
        src = self.decoder(src)
        return src.reshape(src.shape[0], img_output_shape, img_output_shape, 3)

torch.set_float32_matmul_precision('high')
if __name__ == "__main__":

    model = Model().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

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
    loss_function_image = nn.L1Loss()
    writer = SummaryWriter(log_dir=f"runs/{model_name}_128_.0001", flush_secs=10)

    dataset = ImageDatasetGenerative() # ImageDatasetInfill()
    # for i, (test_metadata, test_x, test_y) in enumerate(test_dataset, 0):
    #     test_x = test_x.bfloat16()
    #     test_y = test_y.bfloat16()
    #     # test_x = torch.cat([torch.full((test_x.shape[0], text_length), -1, dtype=torch.bfloat16), test_x], dim=-1)
    #     break
    computed_steps = 0
    for total_steps in range(5000000):
        sample_info = dataset[total_steps]
        if sample_info is None:
            continue
        metadata, x_all, y = sample_info
        x_img = x_all[0].bfloat16().to(device) / 255
        x_text = x_all[1]
        y = y.bfloat16().to(device) / 255

        if total_steps % 25 == 0 and total_steps > 0:
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
        
        # save every 100,000 examples
        computed_steps += 1
        if computed_steps % 450 == 0 and total_steps > 2:
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


