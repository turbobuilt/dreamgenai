from typing import Tuple
from text_preprocess import ArticleDataset
from img_preprocess_infill import ImageDatasetInfill
import torch
from img_preprocess_generative import TrainMode, Metadata, text_length, img_output_shape, image_dim


class MultiDataset():
    def __init__(self):
        self.text_dataset = ArticleDataset()
        self.image_dataset = ImageDatasetInfill()
        self.text_dataset_done = False
        self.image_dataset_done = False
        
    def __iter__(self):
        self.text_dataset_iter = iter(self.text_dataset)
        self.image_dataset_iter = iter(self.image_dataset)
        self.current_iterator = self.text_dataset_iter
        self.text_dataset_done = False
        self.image_dataset_done = False
        return self
    
    def __next__(self) -> Tuple[Metadata, torch.tensor, torch.tensor]:
        try:
            if self.current_iterator == self.text_dataset_iter:
                print("getting text")
                try:
                    metadata, x, y = self.text_dataset.__next__()
                except StopIteration:
                    if self.text_dataset_done and self.image_dataset_done:
                        raise StopIteration
                    self.text_dataset_iter = iter(self.text_dataset)
                self.current_iterator = self.image_dataset_iter
            else:
                print("getting image")
                try:
                    metadata, x, y = self.image_dataset.__next__()
                except StopIteration:
                    if self.text_dataset_done and self.image_dataset_done:
                        raise StopIteration
                    self.image_dataset_iter = iter(self.image_dataset)
                self.current_iterator = self.text_dataset_iter
            if x is None:
                print("got none")
                return self.__next__()
                
            return metadata, x, y
        
        except StopIteration:
            self.iterator = iter(self.train)
            raise StopIteration
        

if __name__ == "__main__":
    dataset = MultiDataset()
    i = 0
    for x,y in dataset:
        i += 1
        if i > 50:
            break
        print(x.shape, y.shape)