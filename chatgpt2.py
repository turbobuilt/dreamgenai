import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class ConvTextToImage(nn.Module):
    def __init__(self, input_dim=1024, output_channels=3, num_layers=5, base_channels=64, image_size=64):
        super(ConvTextToImage, self).__init__()
        
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.num_layers = num_layers
        self.base_channels = base_channels
        self.image_size = image_size
        
        # Initial fully connected layer to expand the input embedding
        self.fc = nn.Linear(input_dim, base_channels * (image_size // 4) * (image_size // 4))
        
        # Convolutional layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = base_channels if i == 0 else base_channels * (2 ** (i - 1))
            out_channels = base_channels * (2 ** i)
            self.convs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            self.convs.append(nn.BatchNorm2d(out_channels))
            self.convs.append(nn.ReLU())
        
        # Final layer to output the desired number of channels
        self.final_conv = nn.Conv2d(out_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.base_channels, self.image_size // 4, self.image_size // 4)
        
        for layer in self.convs:
            x = layer(x)
        
        x = self.final_conv(x)
        return torch.tanh(x)


class ImageTextDataset(Dataset):
    def __init__(self, root_dir, transform=None, patch_size=64):
        self.root_dir = root_dir
        self.transform = transform
        self.patch_size = patch_size
        self.image_paths = []
        self.texts = []

        # Load image paths and texts from JSON files
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                image_path = os.path.join(folder_path, 'image.jpg')
                json_path = os.path.join(folder_path, 'info.json')
                
                if os.path.exists(image_path) and os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            info = json.load(f)
                            self.image_paths.append(image_path)
                            self.texts.append(info['TEXT'])
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error loading JSON from {json_path}: {e}")
                    except Exception as e:
                        print(f"Unexpected error loading {json_path}: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            text = self.texts[idx]

            # Resize the image
            if self.transform:
                image = self.transform(image)

            # Cut the image into patches
            patches = self.cut_into_patches(image)

            # Create a composite image with the first patch filled in
            composite_image = self.create_composite_image(patches)

            return composite_image, patches, text
        except (FileNotFoundError, OSError) as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
        except Exception as e:
            print(f"Unexpected error loading image {self.image_paths[idx]}: {e}")

        # Return a default value in case of error
        return None, None, None

    def cut_into_patches(self, image):
        patches = []
        width, height = image.size
        
        for i in range(0, height, self.patch_size):
            for j in range(0, width, self.patch_size):
                box = (j, i, j + self.patch_size, i + self.patch_size)
                patch = image.crop(box)
                if patch.size == (self.patch_size, self.patch_size):
                    patches.append(patch)

        return patches

    def create_composite_image(self, patches):
        if not patches:
            return None

        # Create a black canvas
        composite_image = Image.new('RGB', (len(patches[0]) * self.patch_size, len(patches) * self.patch_size), (0, 0, 0))
        
        # Paste the patches into the composite image
        for idx, patch in enumerate(patches):
            x = (idx % (composite_image.width // self.patch_size)) * self.patch_size
            y = (idx // (composite_image.width // self.patch_size)) * self.patch_size
            composite_image.paste(patch, (x, y))

        return composite_image
    
    
# DataLoader configuration
def get_dataloader(root_dir, batch_size=8, patch_size=64):
    transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageTextDataset(root_dir=root_dir, transform=transform, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def save_progress_images(input_image, target_patch, generated_patch, step, output_dir='ai_img_out'):
    """Save the input image, target patch, and generated patch."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure to display the images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image
    axs[0].imshow(input_image.permute(1, 2, 0).numpy())
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    
    # Target patch
    axs[1].imshow(target_patch.permute(1, 2, 0).numpy())
    axs[1].set_title('Target Patch')
    axs[1].axis('off')
    
    # Generated patch
    axs[2].imshow(generated_patch.permute(1, 2, 0).numpy())
    axs[2].set_title('Generated Patch')
    axs[2].axis('off')
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'progress_{step}.png'))
    plt.close(fig)

# Example usage
if __name__ == "__main__":
    # Define the root directory for images
    root_dir = './images'
    
    # Create DataLoader
    dataloader = get_dataloader(root_dir, batch_size=8, patch_size=64)

    # Create the model
    model = ConvTextToImage(input_dim=1024, output_channels=3, num_layers=5, base_channels=64, image_size=64)
    model.train()  # Set the model to training mode

    # Example training loop
    num_epochs = 5
    save_interval = 10  # Save progress every 10 iterations
    step = 0

    for epoch in range(num_epochs):
        for composite_images, patches in dataloader:
            # Simulate a forward pass (replace with actual input embeddings)
            generated_patches = model(composite_images)  # Forward pass
            
            # For demonstration, we will just take the first patch
            target_patch = patches[0][0]  # First patch of the first image
            generated_patch = generated_patches[0]  # Generated patch
            
            # Save progress images at specified intervals
            if step % save_interval == 0:
                save_progress_images(composite_images[0], target_patch, generated_patch, step)

            step += 1