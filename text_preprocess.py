from datasets import load_dataset
import torch
from torch import nn
import multiprocessing as mp
import os
from img_preprocess_generative import OutputType, TrainMode, Metadata, text_length, img_output_shape, image_dim


device = torch.device("cuda:0")
article_length=512
output_length=4

def preprocess_file(example):
    data = example["title"] + "\n\n" + example["text"]
    # limit to article_length
    data = data[:article_length]
    # filter to make sure only ascii characters using regexp
    data = data.encode("ascii", "ignore").decode("ascii")
    if len(data) < article_length:
        return None, None
    # convert to ascii

    all_in = []
    all_out = []
    for index in range(0, article_length, output_length):
        input_data = []
        for char_index, char in enumerate(data):
            if char_index < index:
                input_data.append(ord(char))
            else:
                input_data.append(0)
        torch_input_data = torch.tensor(input_data, dtype=torch.float32)
        output_data = [ord(data[index + x]) for x in range(output_length)]
        torch_output_data = torch.tensor(output_data, dtype=torch.long)
        all_in.append(torch_input_data)
        all_out.append(torch_output_data)
    x = torch.stack(all_in)
    # x = x.flip(-1)
    y = torch.stack(all_out)
    # y = y.flip(-1)
    y = torch.nn.functional.one_hot(y, 128)
    return x.reshape(x.shape[0],x.shape[1]).float(), y

# iterator that will go through the dataset and yield the next example
class ArticleDataset():
    def __init__(self):
        self.index = 0
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        dataset = load_dataset("wikicorpus", "raw_en", data_dir="~/.cache/huggingface/datasets", streaming=False)
        self.train = dataset["train"]
        self.queue = mp.Queue(150)
        self.index = 0
        self.iterator = iter(self.train)
        self.try_count = 0
        
    def __iter__(self):
        self.iterator = iter(self.train)
        return self
    
    def __next__(self):
        try:
            example = next(self.iterator)
            x, y = preprocess_file(example)

            metadata = Metadata(TrainMode.text_only, OutputType.text_only)
            if x is None:
                self.try_count += 1
                return self.__next__()
            self.try_count = 0
            return metadata, x, y
        except StopIteration:
            raise StopIteration
        
    def reset(self):
        self.iterator = iter(self.train)
        
        

if __name__ == "__main__":
    dataset = ArticleDataset()
    i = 0
    for metadata, x, y in dataset:
        i += 1
        print("X is")
        for j in range(5):
            print("j:", j, "".join([chr(val) for val in x[j].int()]))
        print("Y is")
        for j in range(5):
            print("j:", j, "".join([chr(val) for val in y[j].argmax(-1).int()]))
        print(x.shape, y.shape)
        break