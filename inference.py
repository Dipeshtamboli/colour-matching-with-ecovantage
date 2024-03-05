from cProfile import label
import random
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import ANN_model
from util_funcs import validate
from get_train_test import train_test_loader
from scipy import stats
import matplotlib.pyplot as plt

torch.use_deterministic_algorithms(True)
seed_val = 0
torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)


exp_name = "YellowPoplar"
data_split = 'test' #"train" #or 'test'

train_loader, test_loader, [sc, sct] = train_test_loader(exp_name)

model = ANN_model()
model.load_state_dict(torch.load(f"models/{exp_name}_model.pt"))
model.eval()

if data_split == "train":
    losses, gt, pred = validate(model, train_loader, sct)
elif data_split == "test":
    losses, gt, pred = validate(model, test_loader, sct)

mse_l=losses[0].item()
rmse_l=losses[1].item()
mape_l=losses[2].item()
r2=losses[3].item()


slope, intercept, r_value, p_value, std_err = stats.linregress(gt,pred)

# plt plot a line with slpe and intercept   
x = np.linspace(0, 60, 10000)
y = slope*x + intercept

fig = plt.figure(figsize=(6,6))

plt.plot(x, y, 'r', label='Regression Line',color=[255/255, 128/255.0, 0/255.0])
# plt.plot(x, y, 'r', label='Regression Line',color=[214.0/255, 164/255.0, 62/255.0])
plt.plot(x, x, 'black', linestyle='dashed', label='Prediction = GroundTruth')
plt.scatter(gt, pred, label='data', facecolors='none', edgecolors='black')
plt.xlabel('Ground Truth')
plt.ylabel(f'Prediction = {slope:.2f}*GroundTruth + {intercept:.2f}')
plt.title(f'{data_split} data split | R2: {r2:.4f}')
plt.legend(edgecolor='black', framealpha=1)
plt.savefig(f'plots/{exp_name}_{data_split}_regression.png')   
