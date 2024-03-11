'''
TEMP: {225: 0, 250: 1, 275: 2, 300: 3}  
TIME: {180: 0, 360: 1, 540: 2, 720: 3, 900: 4}
'''
from cmath import exp
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
# import joblib

from util_funcs import super_print
from util_funcs import validate
from network import ANN_model
from get_train_test import train_test_loader

torch.use_deterministic_algorithms(True)
seed_val = 0
torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)

learning_rate = 5e-4
time_wt = 1000
mse_wt = 100000
num_epochs = 5000
print = super_print(f'logs/w_{time_wt}-{mse_wt}_r_{learning_rate}_e_{num_epochs}.txt')(print)
exp_name = "YellowPoplar"
train_loader, test_loader, [sc], [temp_classes, time_classes] = train_test_loader(exp_name, batch_size=128)
model = ANN_model()

# softmax for classification
l = nn.CrossEntropyLoss()
mse_l = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr =learning_rate )

# mse_l , rmse_l, mape_l, r2_l = [], [], [], []
temp_cel, time_cel, temp_acc, time_acc = [], [], [], []
# all losses for temp and time


for epoch in range(num_epochs):
    model.train()
    for i, (x, y) in enumerate(train_loader):
        # conver x to float tensor
        x = x.float()
        y_pred = model(x.float())

        loss_temp = l(y_pred[0], y[:, 0].long())
        loss_time = l(y_pred[1], y[:, 1].long())
        # add mse loss for time
        # pdb.set_trace()
        time_preds = torch.argmax(y_pred[1], dim=1)
        mse_l_time = mse_l(time_preds, y[:, 1])
        loss = loss_temp + time_wt * loss_time + mse_wt * mse_l_time
        optimizer.zero_grad()
        # pdb.set_trace()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch: {epoch+1}, train loss: {loss.item()}, temp_loss: {loss_temp.item()}, time_loss: {loss_time.item()}, mse_loss: {mse_l_time.item()}')
    # train_loss, train_gt, train_pred = validate(model, train_loader)
    # all_loss, gt, pred = validate(model, test_loader)
    train_metrics = validate(model, train_loader)
    test_metrics = validate(model, test_loader)
    # print(f"TRAIN: mse_l: {train_loss[0].item()}, rmse_l: {train_loss[1].item()}, mape_l: {train_loss[2].item()}, r2_l: {train_loss[3].item()}")
    # print(f"TEST : mse_l: {all_loss[0].item()}, rmse_l: {all_loss[1].item()}, mape_l: {all_loss[2].item()}, r2_l: {all_loss[3].item()}")
    # pdb.set_trace()
    if epoch % 100 == 0:
        print(f'TRAIN: temp_loss: {train_metrics["temp_cel"]}, time_loss: {train_metrics["time_cel"]}, temp_acc: {train_metrics["temp_acc"]}, time_acc: {train_metrics["time_acc"]}')
        print(f'TEST : temp_loss: {test_metrics["temp_cel"]}, time_loss: {test_metrics["time_cel"]}, temp_acc: {test_metrics["temp_acc"]}, time_acc: {test_metrics["time_acc"]}')
    # mse_l.append(all_loss[0])
    # rmse_l.append(all_loss[1])
    # mape_l.append(all_loss[2])
    # r2_l.append(all_loss[3])
    temp_cel.append(test_metrics["temp_cel"])
    time_cel.append(test_metrics["time_cel"])
    temp_acc.append(test_metrics["temp_acc"])
    time_acc.append(test_metrics["time_acc"])

    # save best model based on time accuracy
    if epoch == 0:
        best_time_acc = test_metrics["time_acc"]
        torch.save(model.state_dict(), f"models/{exp_name}_model.pt")
    elif test_metrics["time_acc"] > best_time_acc:
        best_time_acc = test_metrics["time_acc"]
        print(f"Best time accuracy: {best_time_acc}")
        temp_acc_best = test_metrics["temp_acc"]
        torch.save(model.state_dict(), f"models/{exp_name}_model.pt")
    
torch.save(model.state_dict(), f"models/{exp_name}_model.pt")

fig, ax = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
ax[0, 0].plot(temp_cel, label='temp_cel')
ax[0, 0].set_title('temp_cel')
ax[0, 1].plot(time_cel, label='time_cel')
ax[0, 1].set_title(f'time_cel')
ax[1, 0].plot(temp_acc, label='temp_acc')
ax[1, 0].set_title('temp_acc')
ax[1, 1].plot(time_acc, label='time_acc')
ax[1, 1].set_title(f'time_acc | best_time_acc: {best_time_acc}')


plt.savefig(f'plots/w_{time_wt}-{mse_wt}_r_{learning_rate}_e_{num_epochs}.png')
print(f"Best time accuracy: {best_time_acc}")
print(f"Temp accuracy with best time accuracy: {temp_acc_best}")
# plt.show()

