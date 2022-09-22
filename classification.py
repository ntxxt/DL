from os import openpty
from select import select
from types import MethodWrapperType
import torch 
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class CirculemModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


model = CirculemModel()
model0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1))

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

model.eval()
with torch.inference_mode():
    y_logits = model(X_test)
y_pred_prob = torch.round(torch.sigmoid(y_logits)) #pass raw prediction outputs to sigmoid functions


epochs = 1000
for epoch in range(epochs):
    model.train()
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) #logdits -> prediction probablities -> lables
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)


from sklearn.datasets import make_blobs
num_class = 4
num_features  = 2
X_blob, y_blob = make_blobs(n_samples=1000, n_features=num_features, centers=num_class)
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, test_size=0.2)

#multi-class classification 
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )
    def forward(self,x):
        return self.linear_layer_stack(x)
    
model2 = BlobModel(input_features=2, output_features=4, hidden_units=8)

loss_fn2 = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model2.parameters(), lr=0.1)

#logits -> prob -> lable
y_logits = model2(X_blob_test)
y_pred_prob = torch.softmax(y_logits, dim=1)
y_lable = torch.argmax(y_pred_prob)

epochs = 100
for epoch in range(epochs):
    model2.train()
    y_logits = model2(X_blob_train)
    y_pred= torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn2(y_logits, y_blob_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model2.eval()
    with torch.inference_mode():
        test_logits = model2(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn2(test_logits, y_blob_test)
        
    if epoch %10 == 0:
        print(loss)


