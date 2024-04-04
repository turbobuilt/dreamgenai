from torch import nn
import torch
from typing import List

device = torch.device("cuda")
cpu = torch.device("cpu")

class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
    
    def forward(self, src):
        return torch.sin(src)


class LayerInput():
    def __init__(self, level: int, data):
        self.level = level
        self.data = data

class TrainModel():
    def __init__(self):
        self.reset_layer_history()

    def reset_layer_history(self):
        self.layer_history = []

    def backward(self, target):
        self.reset_layer_history()
        self.layer_history[-1].output.copy_(target)
        for layer_index in range(len(self.layer_history)-1, -1, -1):
            self.layer_history[layer_index].learn()


class TrainLayer(nn.Module):
    def __init__(self, parent_model, learning_rate=0.001): #.015
        super(TrainLayer, self).__init__()
        self.parent_model = parent_model
        self.input_data: List[torch.Tensor] = []
        self.layer_levels = None
        self.output: torch.Tensor = None
        self.registered = False
        self.layers: nn.Sequential
        self.learning_rate = learning_rate

    def go(self, *inputs):
        self.input_data = []
        max_in_layer_index = 0
        inputs_temp = []
        for input in inputs:
            if isinstance(input, torch.Tensor):
                if not hasattr(input, "layer_index"):
                    input.layer_index = 0
                max_in_layer_index = max(max_in_layer_index, input.layer_index)
                inputs_temp.append(input.to(device))
            else:
                inputs_temp.append(input)
            self.input_data.append(input)
            
        self.to(device)
        with torch.no_grad():
            self.output = self.forward(*inputs_temp).to(cpu)
        self.to(cpu)
        
        self.output.layer_index = max_in_layer_index+1
        if self.parent_model.layer_history is None:
            self.parent_model.layer_history = []
        self.parent_model.layer_history.append(self)
        return self.output
    
    def learn(self):
        self.to(device)
        params = []
        inputs_temp: List[torch.Tensor] = []
        start_lr = self.learning_rate
        for input_item in self.input_data:
            if isinstance(input_item, torch.Tensor):
                if not hasattr(input_item, "layer_index"):
                    print("No layer index!")
                    exit()
                inputs_temp.append(input_item.detach().to(device))
                inputs_temp[-1].detach_()
                inputs_temp[-1].requires_grad_(True)
                params.append({ "params": inputs_temp[-1], "lr": start_lr*input_item.layer_index })
            else:
                inputs_temp.append(input_item)
        params.append({ 'params': self.parameters(), 'lr': start_lr })
        optimizer = torch.optim.AdamW(params)
        criterion = nn.L1Loss()
        # print("target", self.output)
        # print("inputs before", self.input_data)
        temp_output: torch.Tensor = self.output.to(device)
        current_lr = start_lr
        def multiply_lr(value):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= value
            return value*current_lr
        previous_loss = 100
        for i in range(1000):
            optimizer.zero_grad()
            
            out = self.forward(*inputs_temp)
            loss = criterion(out, temp_output)
            if i % 10 == 0:
                print("loss is", loss.item(), "current lr", current_lr) #, "temp output", temp_output, "out", out, "in ", *inputs_temp)
            loss.backward()
            optimizer.step()
            if loss.item() < previous_loss:
                current_lr = multiply_lr(1.01)
                print("increasing lr")
            else:
                current_lr = multiply_lr(.95)
                if current_lr < start_lr*.1:
                    print("reducing lr", )
            previous_loss = loss.item()

        print(loss.item())

        optimizer.zero_grad()

        for index in range(len(self.input_data)):
            if isinstance(self.input_data[index], torch.Tensor):
                self.input_data[index].copy_(inputs_temp[index])
                inputs_temp[index].detach_()

        temp_output.detach_()
            # if i % 100 == 0:
            #     print("loss", loss.item())
        # print("inputs after", self.input_data)
        # exit()
    

class TrainLinear(TrainLayer):
    def __init__(self, parent_model, num_inputs, num_outputs, activation=True):
        super(TrainLinear, self).__init__(parent_model)
        layers = []
        if activation:
            layers.append(Sine())
        layers.append(nn.Linear(num_inputs, num_outputs))
        self.layers = nn.Sequential(*layers)

    # @torch.compile(mode="reduce-overhead")
    def forward(self, src):
        out = self.layers(src)
        return out


class MyModel(TrainModel):
    def __init__(self):
        super(TrainModel, self).__init__()
        self.layers = [
            TrainLinear(self, 10,10, activation=False),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10),
            TrainLinear(self, 10,10)
        ]

    def forward(self, src):
        for layer in self.layers:
            src = layer.go(src)
        return src




if __name__ == "__main__":
    model = MyModel()

    x = torch.arange(0,20, dtype=torch.float).reshape(2,10)
    y = torch.arange(0,20, dtype=torch.float).flip(dims=[-1]).reshape(2,10)
    import time
    for i in range(20):
        start = time.time()
        out = model.forward(x)

        model.backward(y)
        out = model.forward(x)
        print("out is", out)
        print("target", y)
        print(time.time() - start)
