import pdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

exp_name = "YellowPoplar"
file_errors_location = f'Data4Dipesh/{exp_name}/{exp_name}_Front.xlsx'
df_init = pd.read_excel(file_errors_location)
temps = [225, 250, 275, 300, 325]
times = [10, 20, 30, 40, 50]

# times = [10]
# temps = [300]
clean_data = pd.DataFrame(columns=df_init.columns)

for temp in temps:
    for time in times:
        file_name = f'{temp}-{time}'
        print(file_name)
        df = pd.read_excel(file_errors_location, sheet_name=file_name)

        # df = df[df['Point_No']>4 and df['Point_No']<= 20]
        df = df[df['Point_No']<= 20]
        df = df[df['Point_No']>4]
        
        L_after = df[df.columns[7]]
        L_before = df[df.columns[4]]
        a_after = df[df.columns[8]]
        a_before = df[df.columns[5]]
        b_after = df[df.columns[9]]
        b_before = df[df.columns[6]]

        color_diff = np.sqrt((L_after-L_before)**2 + (a_after-a_before)**2 + (b_after-b_before)**2)
        mean = np.mean(color_diff)
        std = np.std(color_diff)

        # print(f"before shape: {df.shape}")
        df = df[color_diff < mean+2*std]
        df = df[color_diff > mean-2*std]
        # print(f"after shape: {df.shape}")
        clean_data = pd.concat([clean_data, df], axis=0)

clean_data.to_csv(f'Data4Dipesh/{exp_name}/{exp_name}_Front_Clean.csv')

# first_few = 72

# fig, ax = plt.subplots(3, 1, figsize=(10, 10))
# ax[0].plot(df[df2.columns[4]][:first_few])
# ax[0].plot(mean_L)
# ax[1].plot(L_diff[:first_few])
# ax[2].plot(k[:first_few])
# plt.savefig('plots/Ash_Back.png')

