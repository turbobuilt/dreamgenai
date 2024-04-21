import torch
from img_preprocess_infill import ImageDatasetInfill
from mamba_ssm import Mamba
import pyarrow.parquet as pq
import os
import glob
from torch import nn
from data_preprocess import MultiDataset
from img_preprocess_generative import TrainMode, Metadata, OutputType, text_length, img_output_shape, image_dim
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from deepspeed.ops.adam import FusedAdam as DeepSpeedCPUAdam
from deepspeed import DeepSpeedEngine
import time
from trainer_complicated import TrainLinear, TrainLayer, TrainModel

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

class Layer(TrainLayer):
    def __init__(self, parent_model, in_dim, out_dim, kernel_size=4, activation=nn.functional.gelu, add=False):
        super(Layer, self).__init__(parent_model)
        # self.conv = nn.Conv1d(1, out_channels=kernel_size, kernel_size=kernel_size, stride=kernel_size, device=device, dtype=torch.bfloat16, bias=False)

        self.linear_1 = nn.Linear(in_dim, in_dim, device=device, dtype=torch.bfloat16, bias=False)
        self.linear_1.weight.data.uniform_(-1, 1)  # Initializes weights with uniform random values between -1 and 1
        # self.linear_1.weight.data /= out_dim*2
        self.linear_2 = nn.Linear(in_dim, out_dim, device=device, dtype=torch.bfloat16, bias=False)
        self.linear_2.weight.data.uniform_(-1, 1)  # Initializes weights with uniform random values between -1 and 1
        # self.linear_2.weight.data /= out_dim*2

        self.activation = activation
        self.add = add

    # @torch.compile(mode="max-autotune")
    def forward(self, src):
        # original = src
        src = self.linear_1(src.reshape(src.shape[0], 1, src.shape[1]))
        if self.activation is not None:
            # src = self.activation(src)
            src = src.tanh()
            # src = nn.functional.gelu(src)
        src = self.linear_2(src.reshape(src.shape[0], -1))
        if self.activation is not None:
            # src = self.activation(src)
            src = src.tanh()
        # return src.clamp(-1,1)
        # return src.tanh()
        # if self.add:
        #     src = src + original
        # return src
        return src # src.clamp(-1,1)
    

class SigmoidOut(TrainLayer):
    def __init__(self, parent_model, in_dim, out_dim):
        super(SigmoidOut, self).__init__(parent_model)
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, device=device, dtype=torch.bfloat16, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    # @torch.compile(mode="max-autotune")
    def forward(self, src):
        src = self.linear(src)
        # src = self.sigmoid(src)
        return src.reshape(-1, img_output_shape, img_output_shape, 3)

class Model(TrainModel):
    def __init__(self):
        super().__init__()
        self.input_layers = [Layer(self, image_dim**2*3+text_length,image_dim**2*3+text_length,add=True) for _ in range(3)]
        self.input_layers_2 = [
            Layer(self, image_dim**2*3+text_length, image_dim**2+text_length),
            *[Layer(self, image_dim**2+text_length, image_dim**2+text_length, add=True) for _ in range(25)],
            Layer(self,image_dim**2+text_length, 1024)
            # nn.Linear(image_dim**2+text_length, 1024, device=device, dtype=torch.bfloat16, bias=False)
        ]
        # self.input_to_main = 
        self.main_layers = [Layer(self, 1024, 1024,add=True) for _ in range(5)] #250
        # self.image_in = [*[Layer(self, image_dim**2*3,image_dim**2*3) for _ in range(25)])
        self.image_out = [
            Layer(self, 1024, img_output_shape**2*3*32),
            *[Layer(self, img_output_shape**2*3*32,img_output_shape**2*3*32,add=True) for _ in range(5)], #25
            Layer(self, img_output_shape**2*3*32,img_output_shape**2*3),
            SigmoidOut(self, img_output_shape**2*3,img_output_shape**2*3)
        ]

        text_out_shape=4
        # self.text_in = [*[Layer(self, text_length, text_length) for _ in range(25)])
        self.text_out = [
            Layer(self, 1024,text_out_shape*32),
            *[Layer(self, text_out_shape*32,text_out_shape*32) for _ in range(25)], 
            Layer(self, text_out_shape*32,text_out_shape*128)
        ]
    # @torch.compile(mode="reduce-overhead")
    def forward(self, output_type: OutputType, src, example_index=None): # text_input=None, image_input=None):
        # if text_input is not None:
        #     text_input = self.text_in(src)
        # else:
        #     text_input = torch.full((1,text_length), -1)
        
        # if image_input is not None:
        #     image_input = self.image_in(image_input)
        # else:
        #     image_input = torch.full((1,image_dim**2*3), -1)

        # src = torch.cat([image_input, text_input], dim=-1)
        for layer in self.input_layers:
            src = layer.go(src)
            print("after input layers", src)
        for layer in self.input_layers_2:
            src = layer.go(src)
            print("after input layers 2", src[0,0])
        # src = self.input_to_main(src)

        for layer in self.main_layers:
            src = layer.go(src)
            print("after conver tto main", src[0,0])

        if output_type == OutputType.text_only:
            src = self.text_out.go(src).reshape(-1, 4, 128)
            return src
        elif output_type == OutputType.image_only:
            for layer in self.image_out:
                src = layer.go(src)
                print("after convert to output layer", src[0,0])
            return src
        else:
            raise "not implemented - no output type t (text) or i (image)"

    
if __name__ == "__main__":
    epoch = 0
    example_index = 0
    total_steps = 0

    model = Model()
    # print(">>>>>>>>>>>>>>>>>>>>>>DONE")
    # # model, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0, foreach=False)
    # lr = 0
    # target_lr = .000001
    # # optimizer_dict = {p: torch.optim.Adam([p], foreach=False) for p in model.parameters()}
    # # def optimizer_hook(parameter) -> None:
    # #     optimizer_dict[parameter].step()
    # #     optimizer_dict[parameter].zero_grad()
    # # for p in model.parameters():
    # #     p.register_post_accumulate_grad_hook(optimizer_hook)

    # def set_lr(lr):
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     # for key in optimizer_dict:
    #     #     for param_group in optimizer_dict[key].param_groups:
    #     #         param_group['lr'] = lr

    #     # Register the hook onto every parameter


    loss_function_text = nn.CrossEntropyLoss()
    loss_function_image = nn.MSELoss()
    writer = SummaryWriter(log_dir=f"runs/{model_name}_128_.0001", flush_secs=10)

    test_dataset = ImageDatasetInfill()
    for i, (test_metadata, test_x, test_y) in enumerate(test_dataset, 0):
        test_x = test_x.bfloat16()
        test_y = test_y.bfloat16()
        test_x = torch.cat([torch.full((test_x.shape[0], text_length), -1, dtype=torch.bfloat16), test_x], dim=-1)
        break

    # # read from save_files and load most recent
    # list_of_files = glob.glob(f"{checkpoint_dir}/*")
    # if len(list_of_files) > 0:
    #     latest_file = max(list_of_files, key=os.path.getctime)
    #     print(f"loading from {latest_file}")
    #     load_path, client_state = model.load_checkpoint(latest_file)
    #     epoch = client_state['epoch']
    #     example_index = client_state['example_index']
    #     total_steps = client_state['total_steps']
    #     print(f"loaded epoch: {epoch} example_index: {example_index} total_steps: {total_steps}")

    for epoch in range(epoch, 40):
        dataset = ImageDatasetInfill() # MultiDataset()
        # for example_index, (metadata, x, y) in enumerate(dataset, example_index):
        for i in range(5000):
            total_steps += 1
            x = test_x.clone()
            y = test_y.clone()
            metadata = test_metadata

            # new_lr = target_lr * .95**(total_steps // 100000)
            # if lr != new_lr:
            #     print("setting lr", new_lr)
            #     lr = new_lr
            #     set_lr(new_lr)
            #     # for param_group in optimizer.param_groups:
            #     #     param_group['lr'] = lr



            # if x.shape[1] == text_length:
            #     with torch.no_grad():
            #         x = torch.cat([x, torch.full((x.shape[0], image_dim**2*3), -1, dtype=torch.float)], dim=-1)
            # else:
            #     with torch.no_grad():
            #         x = torch.cat([torch.full((x.shape[0], text_length), -1, dtype=torch.float), x], dim=-1)
            x = x.bfloat16().to(device)
            y = y.bfloat16().to(device)
            # optimizer.zero_grad()


            if (example_index % 10 == 0 or example_index % 10 == 1) and example_index > 0:
                if metadata.output_type == OutputType.image_only:
                    image = test_x[-1,512:].reshape(image_dim, image_dim, 3)
                    img_array = image.int().numpy().astype(np.uint8)
                    Image.fromarray(img_array).save("test_out/main.png")
                    for example_part_index in range(test_x.shape[0]):
                        with torch.no_grad():
                            # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            prediction = model.forward(OutputType.image_only, test_x[example_part_index:example_part_index+1].to(device), example_index=example_part_index)
                            prediction = prediction.reshape(img_output_shape,img_output_shape,3)*255
                            prediction = prediction.cpu().int().numpy().astype(np.uint8)
                            Image.fromarray(prediction).save(f"test_out/{example_part_index}.png")

                            target= test_y[example_part_index:example_part_index+1]
                            target = target.reshape(img_output_shape,img_output_shape,3)*255
                            target = target.cpu().int().numpy().astype(np.uint8)
                            Image.fromarray(target).save(f"test_out/{example_part_index}_target.png")
                # exit()

            loss_function = loss_function_image
            if len(y.shape) == 3:
                loss_function = loss_function_text

            for batch_item_index in range(x.shape[0]):
                # optimizer.zero_grad()
                start = time.time()
                print("doing forward")
                prediction = model.forward(metadata.output_type, x[batch_item_index:batch_item_index+1].to(device))
                print("predicted", prediction.shape)
                # exit()

                # for param_group in optimizer.param_groups:
                #     # param_group['lr'] = lr
                #     print(param_group['lr'])
                # print("Prediction", prediction[0,0])
                loss = loss_function(prediction.to(device), y[batch_item_index:batch_item_index+1].to(device))
                if batch_item_index % 100 == 0:
                    print("total steps", total_steps, "batch_item_index", batch_item_index, "loss", loss.item())
                    print("output", prediction[0,0,0], prediction.shape)
                    print("target", y[batch_item_index:batch_item_index+1,0,0])
                    writer.add_scalar("main_loss", loss.float().item(), total_steps)
                    writer.add_scalar("min", torch.min(prediction).float(), total_steps)
                    writer.add_scalar("max", torch.max(prediction).float(), total_steps)
                model.backward(y[batch_item_index:batch_item_index+1], example_index == 0)
                # loss.backward()
                # optimizer.step()
                # print(time.time() - start)

            if example_index % 500 == 0 and False:
                print(f"epoch: {example_index} Loss: {loss.item()}")
                optimizer.zero_grad()
                test_data = torch.zeros(text_length, dtype=torch.bfloat16).to(device)
                for test_index in range(0, text_length, 4):
                    with torch.no_grad():
                        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        prediction = model(test_data)
                        predicted_chars = torch.argmax(prediction, dim=-1)
                        test_data[test_index:test_index+4] = predicted_chars

                out = "".join([chr(int(x)) for x in test_data])
                # replace all new lines with a special emoji return character "↩"
                out = out.replace("\n", "↩")
                print(out)
            
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


