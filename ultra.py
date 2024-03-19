from datasets import load_dataset
import torch
from torch import nn

device = torch.device("cuda:0")

#X X X X X
#X X X X X X X X X X X
#X X X 

article_length=64

class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()

        self.brain = nn.Sequential(
            nn.Linear(article_length, article_length*15),
            nn.ReLU(),
            nn.LayerNorm(article_length*15),
            nn.Linear(article_length*15, article_length*15),
            nn.ReLU(),
            nn.LayerNorm(article_length*15),
            nn.Linear(article_length*15, article_length*15),
            nn.ReLU(),
            nn.LayerNorm(article_length*15),
            nn.Linear(article_length*15, article_length*15),
            nn.ReLU(),
            nn.LayerNorm(article_length*15),
            nn.Linear(article_length*15, article_length*15),
            nn.ReLU(),
            nn.LayerNorm(article_length*15),
            nn.Linear(article_length*15, article_length*15),
            nn.ReLU(),
            nn.LayerNorm(article_length*15),
            nn.Linear(article_length*15, article_length*15),
            nn.ReLU(),
            nn.LayerNorm(article_length*15),
            nn.Linear(article_length*15, article_length*15),
            nn.ReLU(),
            nn.LayerNorm(article_length*15),
            nn.Linear(article_length*15, article_length*15),
            nn.ReLU(),
            nn.LayerNorm(article_length*15),
            nn.Linear(article_length*15, article_length*15),
            nn.ReLU(),
            nn.LayerNorm(article_length*15),
            nn.Linear(article_length*15, 128),
        )

    def forward(self, x):
        return self.brain(x)


# convert to tensor from ascii
def preprocess_file(example):
    data = example["coding_problem"] + "\n\n" + example["coding_solution"]
    # limit to article_length
    data = data[:article_length]
    # filter to make sure only ascii characters using regexp
    data = data.encode("ascii", "ignore").decode("ascii")
    if len(data) < article_length:
        return None, None
    # convert to ascii

    all_in = []
    all_out = []
    for index in range(0, article_length):
        input_data = []
        for char_index, char in enumerate(data):
            if char_index < index:
                input_data.append(ord(char))
            else:
                input_data.append(0)
        torch_input_data = torch.tensor(input_data, dtype=torch.float32)

        output_data = ord(data[index])
        torch_output_data = torch.tensor(output_data, dtype=torch.long)
        all_in.append(torch_input_data)
        all_out.append(torch_output_data)
    
    return torch.stack(all_in), torch.stack(all_out)


model = LanguageModel().to(device)
# print number of parameters
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
exit()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss()


dataset = load_dataset("wikicorpus", "raw_en")
split = dataset["train"].train_test_split(test_size=0.1)
train = split["train"]

save_files = "language_model_saves"
# create directory if not exists
import os
if not os.path.exists(save_files):
    os.makedirs(save_files)

# read from save_files and load most recent
import glob
list_of_files = glob.glob(f"{save_files}/*")
epoch = 0
example_index = 0
total_steps = 0
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
    for example_index, example in enumerate(train, example_index):  
        input_data, output_data = preprocess_file(example)

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


# for example in train:
#     title = example["title"]
#     text = example["text"]
#     break