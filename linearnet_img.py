# iterator that reads images from images/*.  This contains a bunch of dirs with image.jpg, and info.json
# read the image and info.json, and return the image and the json as a dict
from PIL import Image
import os
import json
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import enum
import subprocess
import io

class TrainMode(enum.Enum):
    create_image=0
    image_only=1
    text_only=2

class OutputType(enum.Enum):
    text_only=1
    image_only=2

image_dim = 64
img_output_shape=16
text_length=1024

def encode_image_description(data, max_length=text_length):
    data = data.encode("ascii", "ignore").decode("ascii")[:max_length]
    return data
    data = torch.tensor([ord(c) for c in data], dtype=torch.long)
    data = torch.nn.functional.pad(data, (0, max_length - data.size(0)), "constant", 0)
    return data.reshape(1, text_length)

class Metadata():
    def __init__(self, mode: TrainMode, output_type: OutputType, img_width=-1, img_height=-1, inset_image_width=-1, inset_image_height=-1):
        self.mode = mode
        self.output_type = output_type
        self.img_width = img_width
        self.img_height = img_height
        self.inset_image_width = inset_image_width
        self.inset_image_height = inset_image_height

    def encode(self):
        metadata = torch.zeros(text_length*3)
        metadata[0] = self.mode.value
        metadata[1] = self.output_type
        metadata[2] = self.img_width
        metadata[3] = self.img_height
        metadata[4] = self.inset_image_width
        metadata[5] = self.inset_image_height

        return metadata


# def encode_metadata(mode: TrainMode, img_width=-1, img_height=-1, inset_image_width=-1, inset_image_height=-1):
#     metadata = torch.zeros(d_model*3)
#     metadata[0] = mode.value
#     metadata[1] = img_width
#     metadata[2] = img_height
#     metadata[3] = inset_image_width
#     metadata[4] = inset_image_height

#     metadata = metadata.reshape(-1,d_model)
#     return metadata


class ImageDatasetGenerative(Dataset):
    def __init__(self):
        self.image_dirs = glob.glob('images/*')
        self.image_dirs = [x for x in self.image_dirs if os.path.exists(f"{x}/info.json") and os.path.exists(f"{x}/image.jpg")]
        
    def __len__(self):
        return len(self.image_dirs)
    
    def __getitem__(self, idx):
        try:
            image_dir = self.image_dirs[idx % (len(self.image_dirs)-1)]
            with open(f"{image_dir}/info.json", "r") as f:
                info = json.load(f)
            image = Image.open(f"{image_dir}/image.jpg").resize((image_dim,image_dim))
            img_numpy = np.array(image).astype(np.int32) / 255
            description = encode_image_description(info["TEXT"])
            return description, img_numpy
        except Exception as e:
            # self.image_dirs.remove(image_dir)
            # os.remove(f"{image_dir}")
            # import traceback
            # print(traceback.format_exc())
            # print("error getting image", idx, f"{image_dir}/image.jpg",  e)
            
            return None
        
def print_image(img):
    img_array = img.int().numpy().astype(np.uint8)
    img = Image.fromarray(img_array)
    return img

if __name__ == "__main__":
    dataset = ImageDatasetGenerative()
    # print first 2
    for i in range(1):
        metadata, x, y = dataset[i]
        files = glob.glob("training_data/*")
        for file in files:
            os.remove(file)
        for step in range(image_dim // 16 * (image_dim // 16)):
            images, description = x
            print(images.shape, y.shape)
            img = x[0][step].reshape(image_dim,image_dim,3)
            print_image(img).save(f"training_data/{step}_in.png")
            print_image(y[step]).save(f"training_data/{step}_out.png")