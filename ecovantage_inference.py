# temp: 250, time: 6, 9, 15

TEMP = {225: 0, 250: 1, 275: 2, 300: 3}  
TIME = {180: 0, 360: 1, 540: 2, 720: 3, 900: 4}

# create reverse dictionary
TEMP_class = {str(v): k for k, v in TEMP.items()}
TIME_class = {str(v): k for k, v in TIME.items()}

import torch
import pdb
import numpy as np
import random
import glob
import os
from util_funcs import super_print
from util_funcs import validate
from network import ANN_model
from get_train_test import train_test_loader, get_ecovantage_final_lab




torch.use_deterministic_algorithms(True)
seed_val = 745
# seed_val = np.random.randint(0, 1000)
torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)



exp_name = "YellowPoplar"
train_loader, test_loader, [sc], [temp_classes, time_classes] = train_test_loader(exp_name, batch_size=128)
model = ANN_model()
all_models = glob.glob("models/*.pt")
# path = f"models/best_YellowPoplar_model_0.92_0.952.pt"

# path = f"path: models/best_YellowPoplar_model_0.856_0.912.pt"
# path: models/best_YellowPoplar_model_0.84_0.888.pt
# path: models/best_YellowPoplar_model_0.832_0.872.pt
# path: models/best_YellowPoplar_model_0.848_0.912.pt
for path in ["models/best_YellowPoplar_model_0.856_0.912.pt",
             "models/best_YellowPoplar_model_0.84_0.888.pt",
             "models/best_YellowPoplar_model_0.832_0.872.pt",
             "models/best_YellowPoplar_model_0.848_0.912.pt"]:
# for path in all_models:
    # path =f"models/best_YellowPoplar_model_0.848_0.896.pt"
    time_acc = path.split('_')[-2]
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"path: {path}")


    # print = super_print(f'logs/test_{time_acc}_{seed_val}_.txt')(print)

    eco_temps = [190, 200, 212]
    # eco_temps = [212]
    for eco_temp in eco_temps:

        eco_points = get_ecovantage_final_lab(eco_temp, sc)
        eco_points = eco_points.float()
        temp, time = model(eco_points)

        temp_pred = torch.argmax(temp, dim=1)
        time_pred = torch.argmax(time, dim=1)

        # return the temp and time from the dictionary
        pred_temp_list = [TEMP_class[str(temp.item())] for temp in temp_pred]
        pred_time_list = [TIME_class[str(time.item())] for time in time_pred]

        mean_temp = np.mean(pred_temp_list)
        mean_time = np.mean(pred_time_list)
        meadian_temp = np.median(pred_temp_list)
        median_time = np.median(pred_time_list)
        std_temp = np.std(pred_temp_list)
        std_time = np.std(pred_time_list)

        # pdb.set_trace()
        # count the number of times each temp and time is predicted
        temp_dict = {}
        time_dict = {}
        for temp in pred_temp_list:
            if temp in temp_dict:
                temp_dict[temp] += 1
            else:
                temp_dict[temp] = 1
        for time in pred_time_list:
            if time in time_dict:
                time_dict[time] += 1
            else:
                time_dict[time] = 1
        # sort the dictionary
        temp_dict = dict(sorted(temp_dict.items()))
        time_dict = dict(sorted(time_dict.items()))
        high_freq_temp = max(temp_dict, key=temp_dict.get)
        high_freq_time = max(time_dict, key=time_dict.get)
        print(f'For Eco Temp: {eco_temp}')

        print(f"freq of temp: {temp_dict}")
        print(f"freq of time: {time_dict}")
        print(f'High freq Temp: {high_freq_temp} | High freq Time: {high_freq_time}')

        print(f'Median Temp: {meadian_temp} | Median Time: {median_time}')
        print(f'Mean Temp: {mean_temp} | Mean Time: {mean_time}')
        print(f'Std Temp: {std_temp} | Std Time: {std_time}')



    print(f"##############################################")