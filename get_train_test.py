import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data_utils
import torch 
import pdb


def train_test_loader(exp_name, batch_size=32):

    data_file = f'Data/Yellow_Poplar_0305.xlsx'
    df = pd.read_excel(data_file, sheet_name="225-10")
    # df = pd.read_csv(data_file)
    # df = df.drop(columns=['Unnamed: 0'])

    # ['Temperature/℃', 'Time/s', 'L*Before', 'a*Before', 'b*Before', 'L*After', 'a*After', 'b*After']

    temp = df[df.columns[0]]
    time = df[df.columns[1]]
    
    L_before = df[df.columns[2]]
    a_before = df[df.columns[3]]
    b_before = df[df.columns[4]]

    L_after = df[df.columns[5]]
    a_after = df[df.columns[6]]
    b_after = df[df.columns[7]]


    # pdb.set_trace()
    X = df[df.columns[2:8]]
    # X = X.drop(columns=['Point_No'])

    Y = df[df.columns[0:2]]

    # Y = np.sqrt((L_after-L_before)**2 + (a_after-a_before)**2 + (b_after-b_before)**2)

    x_train, x_test, y_train, y_test = train_test_split(X, Y,  test_size=0.2)

    # convert to numpy
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values

    sc = MinMaxScaler()
    sct = MinMaxScaler()

    x_train=sc.fit_transform(x_train)
    y_train =sct.fit_transform(y_train)
    x_test = sc.transform(x_test)
    y_test = sct.transform(y_test)

    train_set = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = data_utils.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, [sc, sct]

if "__main__" == __name__:
    exp_name = "Ash"
    train_loader, test_loader, [sc, sct] = train_test_loader(exp_name)
    for x, y in train_loader:
        print(x)
        print(y)
        pdb.set_trace()
        break