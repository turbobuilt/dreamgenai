import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel

class ImageTextDataset(Dataset):
    def __init__(self, root_dir, image_dim, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.root_dir = root_dir
        self.image_dim = image_dim
        self.folders = [
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.to_tensor = transforms.ToTensor()

    def letterbox_image(self, img, size):
        w, h = img.size
        scale = min(size / w, size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        background = Image.new("RGB", (size, size), (0, 0, 0))
        background.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
        return background

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx, attempt=0):
        if attempt > 10:
            raise Exception("Too many failed attempts to get item")
        try:
            folder = self.folders[idx]
            image_path = os.path.join(folder, "image.jpg")
            with open(os.path.join(folder, "info.json"), "r", encoding="utf-8") as f:
                info = json.load(f)

            text = info["TEXT"]
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                text_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)

            img = Image.open(image_path).convert("RGB")
            img = self.letterbox_image(img, self.image_dim)
            img_tensor = self.to_tensor(img)
            return img_tensor, text_embedding, text
        except Exception as e:
            # try again with random index
            rand_idx = torch.randint(0, len(self.folders), (1,)).item()
            # print(f"Error with index {idx}, trying again with random index", rand_idx)
            return self.__getitem__(rand_idx, attempt=attempt+1)


def create_loader(image_dim, root_dir="images", batch_size=2, shuffle=False, num_workers=4):
    dataset = ImageTextDataset(root_dir, image_dim)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == "__main__":
    data_loader = create_loader(64)
    for idx, (img, text_emb, text) in enumerate(data_loader):
        print(f"Batch {idx}: {img.shape}, img_max: {img.max()}, img_min {img.min()} {text_emb.shape}, {text}")
        if idx == 0:
            break