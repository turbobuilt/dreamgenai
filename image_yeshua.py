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
    def __init__(self, in_dim, out_dim, kernel_size=4):
        super(Layer, self).__init__()
        self.conv = nn.Conv1d(1, out_channels=kernel_size, kernel_size=kernel_size, stride=kernel_size, device=device, dtype=torch.bfloat16)
        self.linear = nn.Linear(in_dim, out_dim, device=device, dtype=torch.bfloat16)

    def forward(self, src):
        src = self.conv(src.reshape(src.shape[0], 1, src.shape[1]))
        src = nn.functional.relu(src)
        src = self.linear(src.reshape(src.shape[0], -1))
        src = nn.functional.relu(src)
        return src

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_layers = nn.Sequential(*[Layer(image_dim**2*3+text_length,image_dim**2*3+text_length) for _ in range(25)])
        self.input_to_main = nn.Linear(image_dim**2*3+text_length, 1024, device=device, dtype=torch.bfloat16)
        self.main_layers = nn.Sequential(*[Layer(1024, 1024) for _ in range(250)])
        # self.image_in = nn.Sequential(*[Layer(image_dim**2*3,image_dim**2*3) for _ in range(25)])
        self.image_out = nn.Sequential(
            Layer(1024, img_output_shape**2*3*32),
            *[Layer(img_output_shape**2*3*32,img_output_shape**2*3*32) for _ in range(25)], 
            Layer(img_output_shape**2*3*32,img_output_shape**2*3),
            nn.Linear(img_output_shape**2*3,img_output_shape**2*3, device=device, dtype=torch.bfloat16),
            nn.Sigmoid()
        )

        text_out_shape=4
        # self.text_in = nn.Sequential(*[Layer(text_length, text_length) for _ in range(25)])
        self.text_out = nn.Sequential(
            Layer(1024,text_out_shape*32),
            *[Layer(text_out_shape*32,text_out_shape*32) for _ in range(25)], 
            Layer(text_out_shape*32,text_out_shape*128)
        )
    
    def forward(self, output_type: OutputType, src): # text_input=None, image_input=None):
        # if text_input is not None:
        #     text_input = self.text_in(src)
        # else:
        #     text_input = torch.full((1,text_length), -1)
        
        # if image_input is not None:
        #     image_input = self.image_in(image_input)
        # else:
        #     image_input = torch.full((1,image_dim**2*3), -1)

        # src = torch.cat([image_input, text_input], dim=-1)
        src = self.input_layers(src)
        src = self.input_to_main(src)

        src = self.main_layers(src)

        if output_type == OutputType.text_only:
            src = self.text_out(src).reshape(-1, 4, 128)
            return src
        elif output_type == OutputType.image_only:
            src = self.image_out(src).reshape(-1, img_output_shape, img_output_shape, 3)
            return src
        else:
            raise "not implemented - no output type t (text) or i (image)"

    
if __name__ == "__main__":
    epoch = 0
    example_index = 0
    total_steps = 0

    model = Model().to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    loss_function_text = nn.CrossEntropyLoss()
    loss_function_image = nn.MSELoss()
    writer = SummaryWriter(log_dir=f"runs/{model_name}_128_.0001", flush_secs=10)

    test_dataset = ImageDatasetInfill()
    for i, (test_metadata, test_x, test_y) in enumerate(test_dataset, 0):
        test_x = test_x.bfloat16()
        test_y = test_y.bfloat16()
        test_x = torch.cat([torch.full((test_x.shape[0], text_length), -1, dtype=torch.bfloat16), test_x], dim=-1)
        break



    # read from save_files and load most recent
    list_of_files = glob.glob(f"{checkpoint_dir}/*")
    if len(list_of_files) > 0:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"loading from {latest_file}")
        checkpoint = torch.load(latest_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        example_index = checkpoint['example_index']
        loss = checkpoint['loss']
        total_steps = checkpoint['total_steps']
        print(f"loaded epoch: {epoch} example_index: {example_index} loss: {loss}")

    for epoch in range(epoch, 40):
        dataset = ImageDatasetInfill() # MultiDataset()
        # for example_index, (metadata, x, y) in enumerate(dataset, example_index):
        for i in range(5000):
            total_steps += 1
            x = test_x.clone()
            y = test_y.clone()
            metadata = test_metadata


            # if x.shape[1] == text_length:
            #     with torch.no_grad():
            #         x = torch.cat([x, torch.full((x.shape[0], image_dim**2*3), -1, dtype=torch.float)], dim=-1)
            # else:
            #     with torch.no_grad():
            #         x = torch.cat([torch.full((x.shape[0], text_length), -1, dtype=torch.float), x], dim=-1)
            x = x.bfloat16().to(device)
            y = y.bfloat16().to(device)

            optimizer.zero_grad()


            # if example_index % 500 == 0 or example_index % 500 == 1:
            #     if metadata.output_type == OutputType.image_only:
            #         print("outputting")
            #         image = test_x[-1,512:].reshape(image_dim, image_dim, 3)
            #         img_array = image.int().numpy().astype(np.uint8)
            #         Image.fromarray(img_array).save("test_out/main.png")
            #         for example_part_index in range(test_x.shape[0]):
            #             with torch.no_grad():
            #                 # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            #                 prediction = model(OutputType.image_only, test_x[example_part_index:example_part_index+1].to(device))
            #                 prediction = prediction.reshape(img_output_shape,img_output_shape,3)*255
            #                 prediction = prediction.cpu().int().numpy().astype(np.uint8)
            #                 Image.fromarray(prediction).save(f"test_out/{example_part_index}.png")

            #                 target= test_y[example_part_index:example_part_index+1]
            #                 target = target.reshape(img_output_shape,img_output_shape,3)*255
            #                 target = target.cpu().int().numpy().astype(np.uint8)
            #                 Image.fromarray(target).save(f"test_out/{example_part_index}_target.png")
            # exit()

            loss_function = loss_function_image
            if len(y.shape) == 3:
                loss_function = loss_function_text

            for batch_item_index in range(x.shape[0]):
                optimizer.zero_grad()
                prediction = model(metadata.output_type, x[batch_item_index:batch_item_index+1])
                # print(prediction.shape)
                # exit()
                loss = loss_function(prediction, y[batch_item_index:batch_item_index+1])
                if batch_item_index % 30 == 0:
                    print("batch_item_index", batch_item_index, loss.item())
                    print("output", prediction[0,0], prediction.shape,"target", y[batch_item_index:batch_item_index+1,0])
                loss.backward()
                optimizer.step()

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
                torch.save({
                    'epoch': epoch,
                    'example_index': example_index,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'total_steps': total_steps
                }, f"{checkpoint_dir}/epoch_{epoch}_example_{example_index}.pt")


