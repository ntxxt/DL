from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt

weigiht = 0.7
bias = 0.3

#create data sets
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weigiht * X + bias
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#linear regression class
class Linearregression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.parameter.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.parameter.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    
    #forward method
    def forward(self, x: torch.Tensor):
        return self.weights * x + self.bias

torch.manual_seed(42)
model1 = Linearregression()
with torch.inference_mode():
    y_preds = model1(X_test)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model1.parameters(), lr=0.01) #learning rate

#training
epochs = 200 #pass data through model 
for epoch in range(epochs):
    #set to training mode 
    model1.train()
    #1. forward pass
    y_preds = model1(X_train)
    #2. cal loss
    loss =loss_fn(y_preds, y_train)
    #3. optimize
    optimizer.zero_grad()
    #4. backprop
    loss.backward()
    #5. step the optimizer
    optimizer.step()

    #testing
    model1.eval()
    with torch.inference_mode(): #trun off gradient tracking
        #forward pass
        test_pred = model1(X_test)
        #loss
        test_loss = loss_fn(test_pred, y_test)
        #print(test_loss)

#save/load modell 
PATH = Path("models")
PATH.mkdir(parents=True, exist_ok=True)
name = "model1.pth"
save_path = PATH / name 
#torch.save(obj=model1.state_dict(), f=save_path)
model1.load_state_dict(torch.load(f=save_path))

#setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#without manual parameters
class Linearregressionv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x:torch.Tensor):
        return self.linear_layer(x)

torch.manual_seed(43)
model2 = Linearregressionv2()

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model2.parameters(), lr=0.01) #learning rate

for epoch in range(epochs):
    #set to training mode 
    model2.train()
    #1. forward pass
    y_preds = model2(X_train)
    #2. cal loss
    loss =loss_fn(y_preds, y_train)
    #3. optimize
    optimizer.zero_grad()
    #4. backprop
    loss.backward()
    #5. step the optimizer
    optimizer.step()
    
    #testing
    model2.eval()
    with torch.inference_mode(): #trun off gradient tracking
        #forward pass
        test_pred = model2(X_test)
        #loss
        test_loss = loss_fn(test_pred, y_test)
        print(test_loss)


