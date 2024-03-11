import torch

# cross entropy loss
l = torch.nn.CrossEntropyLoss()

y_pred = torch.tensor([0.1, 0.1])
y = torch.tensor(0)

loss = l(y_pred, y)

# verift the loss
# loss_ = -torch.log(torch.exp(0.1)/(torch.exp(0.1) + torch.exp(0.2)))
loss_ = -torch.log(torch.exp(y_pred[0])/(torch.exp(y_pred[0]) + torch.exp(y_pred[1])))
loss_ = (torch.exp(y_pred[0])/(torch.exp(y_pred[0]) + torch.exp(y_pred[1])))
print(loss)
print(loss_)