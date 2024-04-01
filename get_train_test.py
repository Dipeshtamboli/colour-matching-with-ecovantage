import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data_utils
import torch 
import pdb


def train_test_loader(exp_name, batch_size=64):

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

    X = X.values
    Y = Y.values
    # convert y values to class
    temp_classes = sorted(list(set(Y[:,0])))
    time_classes = sorted(list(set(Y[:,1])))

    temp_classes = {temp_classes[i]: int(i) for i in range(len(temp_classes))}
    time_classes = {time_classes[i]: int(i) for i in range(len(time_classes))}

    Y_class = np.zeros((Y.shape[0], 2))

    for i in range(Y.shape[0]):
        Y_class[i, 0] = temp_classes[Y[i, 0]]
        Y_class[i, 1] = time_classes[Y[i, 1]]
    

    x_train, x_test, y_train, y_test = train_test_split(X, Y_class,  test_size=0.2)


    sc = MinMaxScaler()
    # sct = MinMaxScaler()

    x_train=sc.fit_transform(x_train)
    # y_train =sct.fit_transform(y_train)
    x_test = sc.transform(x_test)
    # y_test = sct.transform(y_test)

    train_set = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = data_utils.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, [sc], [temp_classes, time_classes]

def get_ecovantage_final_lab(temp, sc):
    data_file = f'Data/Yellow_Poplar_0305.xlsx'
    df_prev = pd.read_excel(data_file, sheet_name="225-10")
    X = df_prev[df_prev.columns[2:5]]    
    x_prev = X.values
    # sample 10 random data points from x_prev
    # x_prev = x_prev[np.random.choice(x_prev.shape[0], 10, replace=False)]

    data_file = f'Data/{temp}.xlsx'
    df = pd.read_excel(data_file)
    # convert df to torch tensor
    x_after = df[df.columns[1:4]].values

    repeated_rows = []
    for row in x_prev:
        row = np.tile(row, [x_after.shape[0], 1])
        repeated_rows.append(row)
    
    x_prev_repeated = np.concatenate(repeated_rows, axis=0)

    x_after_repeated = np.tile(x_after, [x_prev.shape[0], 1])
    # print(f" x_prev shape: {x_prev.shape}")
    # print(f" x_after shape: {x_after.shape}")
    # print(f" x_prev_repeated shape: {x_prev_repeated.shape}")
    # print(f" x_after_repeated shape: {x_after_repeated.shape}")

    x = np.concatenate((x_prev_repeated, x_after_repeated), axis=1)
    x = sc.transform(x)
    x = torch.from_numpy(x)

    return x

if "__main__" == __name__:
    exp_name = "Ash"
    train_loader, test_loader, [sc], [temp_classes, time_classes] = train_test_loader(exp_name, batch_size=2)
    # for x, y in train_loader:
    #     print(x)
    #     print(y)
    #     pdb.set_trace()
    #     break
    # sc = MinMaxScaler()
    points = get_ecovantage_final_lab(190,sc)
