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
from img_preprocess_generative import OutputType, TrainMode, Metadata, text_length, img_output_shape, image_dim


class ImageDatasetInfill(Dataset):
    def __init__(self):
        self.image_dirs = glob.glob('images/*')
        self.image_dirs = [x for x in self.image_dirs if os.path.exists(f"{x}/info.json") and os.path.exists(f"{x}/image.jpg")]
        self.index = 0
        
    def __len__(self):
        return len(self.image_dirs)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == len(self.image_dirs) - 1:
            self.index = 0
            raise StopIteration
        return self.__getitem__(self.index)
    
    def __getitem__(self, idx):
        try:
            image_dir = self.image_dirs[idx]
            with open(f"{image_dir}/info.json", "r") as f:
                info = json.load(f)
            image = Image.open(f"{image_dir}/image.jpg").resize((image_dim, image_dim))
            img_numpy = np.array(image).astype(np.int32)

            odds = img_numpy.copy()
            evens = img_numpy.copy()
            evens_in = []
            odds_in = []
            odds_out = []
            evens_out = []

            img_width = info["content_width"]
            img_height = info["content_height"]
            metadata = Metadata(
                TrainMode.image_only, 
                OutputType.image_only,
                img_width=image_dim, 
                img_height=image_dim, 
                inset_image_width=img_width,
                inset_image_height=img_height
            )

            for y in range(img_numpy.shape[0]-img_output_shape*2, -1, -img_output_shape*2):
                for x in range(img_numpy.shape[1]-img_output_shape*2, -1, -img_output_shape*2):
                    evens_out.append(evens[y:y+img_output_shape, x:x+img_output_shape].copy())
                    evens[y:y+img_output_shape, x:x+img_output_shape] = -1
                    evens_in.append(evens.copy())

                    odds_out.append(odds[y+img_output_shape:y+img_output_shape*2, x+img_output_shape:x+img_output_shape*2].copy())
                    odds[y+img_output_shape:y+img_output_shape*2, x+img_output_shape:x+img_output_shape*2] = -1
                    odds_in.append(odds.copy())

            all_in = evens_in + odds_in
            all_out = evens_out + odds_out

            # for i in range(len(all_in)):
            #     # print(all_in[i].shape)
            #     all_in[i] = np.concatenate([metadata, np.full((1, d_model), -1), all_in[i].reshape(-1, d_model)])
            #     # print(all_in[i].shape)
            

            return metadata, torch.tensor(np.stack(all_in), dtype=torch.float32).reshape(len(all_in), -1), torch.tensor(np.stack(all_out), dtype=torch.float32)/255
        except Exception as e:
            raise e
            print(e)
            return self.__getitem__(idx + 1)

        
def print_image(img):
    img_array = img.int().numpy().astype(np.uint8)
    img = Image.fromarray(img_array)
    return img

if __name__ == "__main__":
    dataset = ImageDatasetInfill()
    # print first 2
    for i in range(1):
        metadata, x, y = dataset[i]
        files = glob.glob("training_data/*")
        for file in files:
            os.remove(file)
        for step in range(x.shape[0]):
            # print("shape", x[step,4:].shape)
            img = x[step,:].reshape(image_dim,image_dim,3)
            print_image(img).save(f"training_data/{step}_in.png")
            print_image(y[step].reshape(img_output_shape, img_output_shape, 3)).save(f"training_data/{step}_out.png")