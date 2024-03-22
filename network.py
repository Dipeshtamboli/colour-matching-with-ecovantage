import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN_model(torch.nn.Module):
    def __init__(self):
        super(ANN_model, self).__init__()
        self.linear1 = torch.nn.Linear(6, 24)  
        self.linear2 = torch.nn.Linear(24, 48)  

        self.linear_temp = torch.nn.Linear(48, 4)
        self.linear_time = torch.nn.Linear(48, 5)

        # batch normalization
        self.bn1 = nn.BatchNorm1d(24)
        self.bn2 = nn.BatchNorm1d(48)


    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        temp = self.linear_temp(x)
        time = self.linear_time(x)
        # temp_n_time = torch.cat((temp, time), 1)
        return temp, time




        return x
        
if "__main__" == __name__:
    model = ANN_model()
    print(model)
    input = torch.FloatTensor(2, 6)
    # model.load_state_dict(torch.load("ash_model.pt"))
    model.eval()
    output = model(input)
    print(output)