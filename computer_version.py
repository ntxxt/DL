from asyncore import file_dispatcher
from pyexpat import model
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#torchvision,
#training
train_data = datasets.FashionMNIST(
    root="/Users/xiaotongxu/data", # where to download data to?
    train=True, # get training data
    download=False, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)
#test
test_data = datasets.FashionMNIST(
    root="/Users/xiaotongxu/data",
    train=False, # get test data
    download=False,
    transform=ToTensor()
)


#load data in batches, computation efficient
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32)
"""
class Basemodel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)

model0 = Basemodel(input_shape=784, hidden_units=10, output_shape=10)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model0.parameters(), lr=0.1)

epochs = 3
for epoch in range(epochs):
    train_loss = 0
    for batch, (X,y) in enumerate(train_dataloader):
        model0.train()
        y_pred = model0(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(train_loss)

    test_loss = 0
    model0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            test_prob = model0(X)
            test_loss += loss_fn(test_prob, y)
        test_loss /=  len(test_dataloader)
    print(test_loss)

"""


class CNNmodel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shapes):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shapes))
    
    def forward(self, x):
        x = self.block_1(x)
        #print(x.shape)
        x = self.block_2(x)
        x = self.classifier(x)
        #print(x.shape)
        return x
    

#break layers into pieces
image = torch.rand(size=(32,3,64,64))
#print(image.shape)
test_image = image[0]
#print(test_image.shape)
#kernel=filter stribe=move amount at a time
conv_layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0) #color channel (3)
cov_output = conv_layer(test_image)
#print(cov_output.shape)
#max pool layer
maxpool = nn.MaxPool2d(kernel_size=2)
maxpool_output = maxpool(cov_output) 
#print(maxpool_output.shape)

#number of channel in images, black and white dataset this is one 
model2 = CNNmodel(input_shape=1, hidden_units=10, output_shapes=10)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model2.parameters(), lr=0.1)


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer
               ):
    train_loss = 0
    for batch, (X, y) in enumerate(data_loader):
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    print(train_loss)

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module):
    test_loss =0
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # 1. Forward pass
            test_pred = model(X)          
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
         # Go from logits -> pred labels
        
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        print(test_loss)

epochs = 3
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model2, 
        loss_fn=loss_fn,
        optimizer=optimizer
    )
    test_step(data_loader=test_dataloader,
        model=model2,
        loss_fn=loss_fn)

