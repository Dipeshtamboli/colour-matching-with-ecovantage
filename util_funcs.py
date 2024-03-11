import torch
import itertools
import pdb
import numpy as np
# import scikit-learn library to calculate accuracy, precision, recall, f1-score, and confusion matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def mse_loss_func(y_pred, y):
    return torch.mean((y_pred-y)**2)

# root mean squared error
def rmse_loss_func(y_pred, y):
    return torch.sqrt(torch.mean((y_pred-y)**2))

# mean absolute percentage error
def mape_loss_func(y_pred, y):
    return torch.mean(torch.abs(y_pred-y)/y)*100

# r^2 score
def r2_loss_func(y_pred, y):
    return 1 - torch.mean((y_pred-y)**2)/torch.var(y)

def super_print(filename):
    '''filename is the file where output will be written'''
    def wrap(func):
        '''func is the function you are "overriding", i.e. wrapping'''
        def wrapped_func(*args,**kwargs):
            '''*args and **kwargs are the arguments supplied 
            to the overridden function'''
            #use with statement to open, write to, and close the file safely
            with open(filename,'a') as outputfile:
                outputfile.write(*args,**kwargs)
                # add a newline character to the end of the string
                outputfile.write('\n')
            #now original function executed with its arguments as normal
            return func(*args,**kwargs)
        return wrapped_func
    return wrap

def validate(model, test_loader):
    gt_temp, pred_temp = [], []
    gt_time, pred_time = [], []
    mse_l , rmse_l, mape_l, r2_l = 0, 0,0,0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.float()
            y_pred = model(x.float())
            # loss += l(y_pred, y.to(torch.float32))
            y_pred_temp = y_pred[0].detach().numpy()
            y_pred_time = y_pred[1].detach().numpy()
            y = (y.detach().numpy())
            # y_pred = sct.inverse_transform(y_pred)
            # y = sct.inverse_transform(y)            
            # pdb.set_trace()
            y_temp = y[:, 0]
            y_time = y[:, 1]
            # gt.append(y)
            # pred.append(y_pred)
            gt_temp.append(y_temp)
            pred_temp.append(y_pred_temp)
            gt_time.append(y_time)
            pred_time.append(y_pred_time)

            #calculate accuracy
             

    # pdb.set_trace()
    gt_temp = gt_temp[0]
    pred_temp = pred_temp[0]
    gt_time = gt_time[0]
    pred_time = pred_time[0]

    # crossentropy loss for temp and time
    cel = torch.nn.CrossEntropyLoss()
    # numpy cross entropy loss
    
    # pdb.set_trace()
    temp_cel = cel(torch.tensor(pred_temp), torch.tensor(gt_temp).type(torch.LongTensor))
    time_cel = cel(torch.tensor(pred_time), torch.tensor(gt_time).type(torch.LongTensor))


    pred_temp = np.argmax(pred_temp, axis=1)
    pred_time = np.argmax(pred_time, axis=1)

    temp_acc = accuracy_score(gt_temp, pred_temp)
    time_acc = accuracy_score(gt_time, pred_time)
    temp_prec = precision_score(gt_temp, pred_temp, average='weighted')
    time_prec = precision_score(gt_time, pred_time, average='weighted')
    temp_recall = recall_score(gt_temp, pred_temp, average='weighted')
    time_recall = recall_score(gt_time, pred_time, average='weighted')
    temp_f1 = f1_score(gt_temp, pred_temp, average='weighted')
    time_f1 = f1_score(gt_time, pred_time, average='weighted')
    temp_conf = confusion_matrix(gt_temp, pred_temp)
    time_conf = confusion_matrix(gt_time, pred_time)

    metrics = {
        'temp_cel': temp_cel.item(),
        'time_cel': time_cel.item(),
        'temp_acc': temp_acc,
        'time_acc': time_acc,
        'temp_prec': temp_prec,
        'time_prec': time_prec,
        'temp_recall': temp_recall,
        'time_recall': time_recall,
        'temp_f1': temp_f1,
        'time_f1': time_f1,
        'temp_conf': temp_conf,
        'time_conf': time_conf
    }

    return metrics



