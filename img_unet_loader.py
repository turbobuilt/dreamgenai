from glob import glob
from PIL import Image
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
class ImgUnetLoader():
    def __init__(self, img_dim=3**4, output_patch_dim=9):
        self.img_dim = img_dim
        self.folders = glob('images/*')
        self.output_patch_dim = output_patch_dim

    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, idx):
        folder = self.folders[idx]
        image = Image.open(f"{folder}/image.jpg")
        image = image.resize((self.img_dim, self.img_dim))
        image = np.array(image)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        text = open(f"{folder}/info.json").read()
        text = json.loads(text)
        text = text["TEXT"]
        embeddings = model.encode([text])

        text_embedding = torch.from_numpy(embeddings[0])

        random_image = -torch.rand((3, self.img_dim, self.img_dim))

        inputs = []
        outputs = []
        for i in range(0, self.img_dim, self.output_patch_dim):
            for j in range(0, self.img_dim, self.output_patch_dim):
                inputs.append(random_image.clone())
                random_image[:, i:i+self.output_patch_dim, j:j+self.output_patch_dim] = image[:, i:i+self.output_patch_dim, j:j+self.output_patch_dim]
                outputs.append(random_image.clone())
        inputs = torch.stack(inputs)
        outputs = torch.stack(outputs)
        return inputs, outputs, text_embedding, text

if __name__ == "__main__":
    loader = ImgUnetLoader()
    for i in range(1):
        inputs, outputs, text_embedding = loader[i]
        print(inputs.shape, outputs.shape, text_embedding.shape)
        # clear test folder
        import os
        os.system("rm -rf input_test")
        os.system("mkdir input_test")
        # write all to input_test
        for i in range(inputs.shape[0]):
            img = inputs[i]
            img = (img * 255).int().numpy().astype(np.uint8)
            img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(f"input_test/{i}_in.png")
            # write all to output_test
            img = outputs[i]
            img = (img * 255).int().numpy().astype(np.uint8)
            img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(f"input_test/{i}_out.png")


        break