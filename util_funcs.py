import torch
import itertools
# testing

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



def validate(model, test_loader, sct):
    gt, pred = [], []
    mse_l , rmse_l, mape_l, r2_l = 0, 0,0,0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.float()
            y_pred = model(x.float())
            # loss += l(y_pred, y.to(torch.float32))
            y_pred = sct.inverse_transform(y_pred.detach().numpy())
            y = sct.inverse_transform(y.detach().numpy())
            # y_pred = sct.inverse_transform(y_pred)
            # y = sct.inverse_transform(y)            
            gt.append(y)
            pred.append(y_pred)
            mse_l += mse_loss_func(torch.tensor(y_pred), torch.tensor(y).to(torch.float32))
            rmse_l += rmse_loss_func(torch.tensor(y_pred), torch.tensor(y).to(torch.float32))
            mape_l += mape_loss_func(torch.tensor(y_pred), torch.tensor(y).to(torch.float32))
            r2_l += r2_loss_func(torch.tensor(y_pred), torch.tensor(y).to(torch.float32))

    
    gt_temp = [i[0].item() for i in list(itertools.chain(*gt))]
    gt_time = [i[1].item() for i in list(itertools.chain(*gt))]
    pred_temp = [i[0].item() for i in list(itertools.chain(*pred))]
    pred_time = [i[1].item() for i in list(itertools.chain(*pred))]
    all_losses = [mse_l/len(test_loader), rmse_l/len(test_loader), mape_l/len(test_loader), r2_l/len(test_loader)]
    return all_losses, gt, pred
