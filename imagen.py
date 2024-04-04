import torch
from img_preprocess_infill import ImageDatasetInfill
from mamba_ssm import Mamba
import pyarrow.parquet as pq
import os
import glob
from torch import nn
from data_preprocess import MultiDataset

device = torch.device("cuda:0")
model_name = "imagen"
out_dir = f"{model_name}/samples"
checkpoint_dir = f"{model_name}/checkpoints"



# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=512, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
device = torch.device("cuda")

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mid_layers = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=.1, batch_first=True, norm_first=True, device=device, dtype=torch.bfloat16) for i in range(10)])


        self.image_in_transformer = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=.1, batch_first=True, norm_first=True, device=device, dtype=torch.bfloat16) for i in range(10)])
        self.image_out_transformer = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=.1, batch_first=True, norm_first=True, device=device, dtype=torch.bfloat16) for i in range(10)])
        self.image_linear = nn.Linear(512, 4*128)


        self.text_in_transformer = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=.1, batch_first=True, norm_first=True, device=device, dtype=torch.bfloat16) for i in range(10)])
        self.text_out_transformer = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=.1, batch_first=True, norm_first=True, device=device, dtype=torch.bfloat16) for i in range(10)])
        self.text_linear = nn.Linear(512, 4*128)

    
    def forward(self, text_input=None, image_input=None):
        if text_input is not None:
            src = self.text_in_transformer(src)
        elif output_type == "i":
            src = self.image_in_transformer(src)

        src = self.mid_layers(src)
        if output_type == "t":
            src = self.text_out_transformer(src)
            src = self.text_linear(src[:,-1]).reshape(out.shape[0], -1, 128)
            return src
        elif output_type == "i":
            src = self.image_out_transformer(src)
            src = self.image_linear(src[:,-1]).reshape(out.shape[0], -1, 128)
            return src
        else:
            raise "not implemented - no output type t (text) or i (image)"

    



# model = Model()
# dataset = ImageDatasetInfill()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=.0001)
# loss_function = nn.MSELoss()
# for index, example in enumerate(dataset):
#     x, y = example
#     x = x.to("cuda")
#     y = y.to("cuda")

#     optimizer.zero_grad()
#     with torch.cuda.amp.autocast_mode.autocast(dtype=torch.bfloat16):
#         out = model(x)
#         loss = loss_function(out, y)
#         loss.backward()

#     if index % 1 == 0:
#         print(index, loss.item())
#         print(x.shape, y.shape)
    
#     optimizer.step()







# exit()
epoch = 0
example_index = 0
total_steps = 0


model = Imagen().to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss()


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
    dataset = MultiDataset()
    for example_index, (x, y) in enumerate(dataset, example_index):  
        if input_data is None:
            continue
        total_steps += 1

        input_data = input_data.to(device)  
        output_data = output_data.to(device)

        optimizer.zero_grad()

        prediction = model(input_data)
        loss = loss_function(prediction, output_data)
        loss.backward()
        optimizer.step()

        if example_index % 500 == 0:
            print(f"epoch: {example_index} Loss: {loss.item()}")
            optimizer.zero_grad()
            test_data = torch.zeros(article_length, dtype=torch.float32).to(device)
            for test_index in range(article_length):
                with torch.no_grad():
                    prediction = model(test_data)
                    predicted_char = torch.argmax(prediction)
                    test_data[test_index] = predicted_char

            out = "".join([chr(int(x)) for x in test_data])
            # replace all new lines with a special emoji return character "↩"
            out = out.replace("\n", "↩")
            print(out)

        # save every 100,000 examples, include epoch and example_index in save file
        if example_index % 100000 == 0 and example_index > 0:
            torch.save({
                'epoch': epoch,
                'example_index': example_index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'total_steps': total_steps
            }, f"{save_files}/epoch_{epoch}_example_{example_index}.pt")


