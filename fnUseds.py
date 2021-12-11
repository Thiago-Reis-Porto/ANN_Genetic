import datetime as dt
import json
import os
import random
from itertools import chain, combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

#----------------------------------------------------------------------------------------------------------------
#  Create directores to keeps NN data and plots  ----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def dirStart():
    tsNow = dt.datetime.now()
    dirName = str(tsNow.year) + '-' + str(tsNow.month) + '-' + str(tsNow.day) + \
        '-' + str(tsNow.hour - 3) + '-' + str(tsNow.minute) + '-' + str(tsNow.second)
    os.mkdir(dirName)
    os.mkdir(dirName+'/'+'plots')
    return dirName
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#  Set Seed  ----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def set_reproducible_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def transformDl(inputs, targets):
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)

    data2 = TensorDataset(inputs, targets)
    batch_size = 100
    train_dl = DataLoader(data2, batch_size, shuffle=True)
    return train_dl, inputs, targets
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#  Set correct thetas format and target in the data set ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def set_data_thetas_targets(df, ins):
    data = pd.DataFrame(df[ins])
    if 'tetas' in df.columns: 
        data['tetas'] = df['tetas']
        data['umidade'] = df['umidade']
    else: 
        columns = df.columns[:-8]
        data[columns] = df[columns]
    return data    
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
# Correct train_test_split call----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def train_split_handler(data):
    columns = data.columns[-8:][:-1]
    if 'tetas' in data.columns: 
        return train_test_split(data, stratify=data['tetas'], test_size=0.1)
    else: return train_test_split(data, stratify=data[columns], test_size=0.1)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
# Set correct data shape for training ---------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def prepare_data_shape(train_split, val):
    data_train = train_split.values
    inputs_data = data_train[0:, :-1].astype(np.float32)
    
    data_val = val.values
    val_inputs_data = data_val[0:, :-1].astype(np.float32)
    
    umidade = data_train[0:, -1].astype(np.float32)
    umidade_val = data_val[0:, -1].astype(np.float32)
    
    umidade = umidade.reshape(-1, 1)
    umidade_val = umidade_val.reshape(-1, 1)
    
    return inputs_data, umidade, umidade_val, val_inputs_data
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
# Split data for training ---------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def split_data(df, ins):
    data = set_data_thetas_targets(df, ins)
    train_split, val = train_split_handler(data)

    data_shapes = prepare_data_shape(train_split, val)
    inputs_data, umidade, umidade_val, val_inputs_data = data_shapes

    train_dl, inputs_data, umidade = transformDl(inputs_data, umidade)
    val_dl, val_inputs_data, umidade_val = transformDl(val_inputs_data, umidade_val)
    
    result = train_dl, val_dl, inputs_data.shape[1], inputs_data, umidade, val_inputs_data, umidade_val
    
    return result
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#  Prepare data for test, return X, y ---------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
def prepare_test_data(df, ins):
    data = set_data_thetas_targets(df, ins)
    values = data.values
    umidade = values[0:, -1].astype(np.float32)
    umidade = umidade.reshape(-1, 1)
    #test_dl, ins, outs = transformDl(inputs_data, umidade)
    inputs = data.drop(columns='umidade')
    targets = torch.from_numpy(umidade)

    return inputs, targets
#----------------------------------------------------------------------------------------------------------------

def plot_test(metrics, predicts, targets, path, name):
    plt.plot(targets)
    plt.plot(predicts)
    plt.ylabel('Umidade')
    plt.xlabel('Index')
    plt.legend(['Targets', 'Predicts'], loc='upper left')
    plt.xticks([0, 50, 100, 150, 200, 250, 300])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.text(230, 0.95, 'R² =' + str(round(metrics['r2'], 2)), fontsize=12)
    plt.text(230, 0.90, 'MAE =' + str(round(metrics['mae'], 2)), fontsize=12)
    plt.text(230, 0.85, 'RMSE =' + str(round(metrics['rmse'], 2)), fontsize=12)
    plt.savefig(path + '/' + name + '.png')
    plt.clf()


def plot_train_no_val(metrics, predicts, targets, path, name):
    plt.plot(targets)
    plt.plot(predicts)
    plt.ylabel('Umidade')
    plt.xlabel('Index')
    plt.legend(['Targets', 'Predicts'], loc='upper left')
    plt.xticks([0, 50, 100, 150, 200, 250, 300])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.text(230, 0.95, 'R² =' + str(round(metrics['r2'], 2)), fontsize=12)
    plt.text(230, 0.90, 'MAE =' + str(round(metrics['mae'], 2)), fontsize=12)
    plt.text(230, 0.85, 'RMSE =' + str(round(metrics['rmse'], 2)), fontsize=12)
    plt.show()
    #plt.savefig(path + '/' + name + '.png')
    # plt.clf()


def train(ins, path, config, epochs, model, loss_fn, opt, data, val_data, scheduler, check, paciencia=9):
    
    epochs_losses = []
    contador = 0
    val_batches_losses = []
    val_epochs_losses = []
    val_best_loss = 9999999999999999

    model.train()
    
    for epoch in range(epochs):
        batches_losses = []
        
        batch_interaction(model, loss_fn, opt, data, batches_losses)
        epoch_loss, current_lr = get_losses_and_lr(opt, scheduler, epochs_losses, batches_losses)
        print_epoch_status(check, epoch, epoch_loss, current_lr)

        with torch.no_grad():
            model.eval()
            preds = []
            targets = []
            
            result = validation_batch_interation(model, loss_fn, val_data, val_batches_losses, preds, targets)
            preds, targets, val_yb, val_pred = result

            val_epoch_loss = np.array(val_batches_losses).mean()
            val_epochs_losses.append(val_epoch_loss)
           
            # if its best so far, save the model
            if(val_best_loss > val_epoch_loss):
                saved = save_model(ins, path, config, model, preds, targets, val_yb, val_pred, val_epoch_loss)
                val_best_loss, best_model, metrics, metrics_val = saved
                contador = 0
            
            # if not improving stop
            elif(contador >= paciencia):
                print('{} \n {}'.format(val_best_loss, val_epoch_loss))
                break
            
            else:
                contador += 1

    return epoch_loss, np.array(epochs_losses).mean(), epochs_losses, val_epochs_losses, val_best_loss, metrics, best_model, metrics_val

def validation_batch_interation(model, loss_fn, val_data, val_batches_losses, preds, targets):
    for val_xb, val_yb in val_data:
        val_xb, val_yb = val_xb.to('cuda'), val_yb.to('cuda')
        val_pred = model(val_xb)
        val_loss = loss_fn(val_pred, val_yb)
        val_batches_losses.append(val_loss.detach().item())
        preds = preds + val_pred.to('cpu').detach().numpy().tolist()
        targets = targets + val_yb.to('cpu').detach().numpy().tolist()
    return preds,targets,val_yb,val_pred

def save_model(ins, path, config, model, preds, targets, val_yb, val_pred, epoch_loss, val=True):
    best_loss = epoch_loss
    best_model = model
    path = path+config+'/'+ins
    
    if val:
        torch.save(model.state_dict(), os.path.join(path+'/best_model.pth'))
        metrics, metrics_val = export_best_metrics(ins, path, config, preds, targets, val_yb, val_pred)
        with open(os.path.join(path+'/', 'args.json'), mode='w') as file:
            json.dump({'config': config, 'Columns': ins}, file)
        return best_loss, best_model, metrics, metrics_val
    
    torch.save(model.state_dict(), os.path.join(path + '/noVal/best_model_no_val.pth'))
    metrics = export_best_metrics(ins, path, config, preds, targets, None, None, val=False) 
    with open(os.path.join(path + '/noVal/', 'args_no_val.json'), mode='w') as file:
                    json.dump({'config': config, 'Columns': ins}, file)

    return best_loss, best_model, metrics
                   
def export_best_metrics(ins, path, config, preds, targets, val_yb, val_pred, val=True):
    metrics = create_metrics_dict(preds, targets)
    l = [metrics]
   
    if not val: 
        df2 = pd.DataFrame(metrics)
        df2.to_csv(os.path.join(path + '/noVal/', 'best_metrics_no_val.csv'), index=False)
        return metrics

    metrics_val = create_metrics_dict(val_pred, val_yb, val=True)
    
    df2 = pd.DataFrame([metrics, metrics_val])
    df2.to_csv(os.path.join(path + '/', 'best_metrics.csv'), index=False)
    
    return metrics, metrics_val
    


def create_metrics_dict(preds, targets, val=False):
    r2, mae, rmse = error_metrics(preds, targets)
    metrics_names = ['r2', 'mae', 'rmse']
    if val: metrics_names = list(map(lambda x : x + '_val', metrics_names))
    metrics = {metrics_names[0]: r2, 
               metrics_names[1]: mae, 
               metrics_names[2]: rmse}              

    return metrics

def print_epoch_status(check, epoch, epoch_loss, current_lr):
    if(epoch % check == 0):
        print('In epoch {}, loss:{}, lr:{}'.format(
                epoch, epoch_loss, current_lr))

def get_losses_and_lr(opt, scheduler, epochs_losses, batches_losses):
    epoch_loss = np.array(batches_losses).mean()
    epochs_losses.append(epoch_loss)
    scheduler.step(epoch_loss)
    current_lr = opt.param_groups[0]['lr']
    return epoch_loss,current_lr

def batch_interaction(model, loss_fn, opt, data, batches_losses, no_val=False, preds=None, targets=None):
    for xb, yb in data:
        xb, yb = xb.to('cuda'), yb.to('cuda')
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        batches_losses.append(loss.detach().item())
        if no_val:
            preds += pred.to('cpu').detach().numpy().tolist()
            targets += yb.to('cpu').detach().numpy().tolist()

def error_metrics(predicts, targets):
    r2 = metrics.r2_score(targets, predicts)
    mae = metrics.mean_absolute_error(targets, predicts)
    rmse = metrics.mean_squared_error(targets, predicts, squared=False)

    return r2, mae, rmse


def mkPlots(net, losses, inputs, targets, metrics, path, metrics_val=None, inputs_val=None, targets_val=None):
    name = 'Train_TargetsXOutputs'
    plt.plot(targets)
    plt.plot(net(inputs).detach().numpy())
    plt.ylabel('Umidade')
    plt.xlabel('Index')
    plt.legend(['Targets', 'Predicts'], loc='upper left')
    plt.xticks([0, 200, 400, 600, 800, 1000, 120])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.text(800, 0.95, 'R² =' + str(round(metrics['r2'], 2)), fontsize=12)
    plt.text(800, 0.90, 'MAE =' + str(round(metrics['mae'], 2)), fontsize=12)
    plt.text(800, 0.85, 'RMSE =' + str(round(metrics['rmse'], 2)), fontsize=12)
    plt.savefig(path+'/'+name+'.png')
    plt.clf()
    lossesPlots(losses, name, path)
    if(metrics_val != None):
        plotsVal(name, net, inputs_val, targets_val, metrics_val, path)
    scatter_plot(net(inputs).detach().numpy(), targets, name, path)

def plotsVal(name, net, inputs, targets, metrics, path):
    plt.plot(targets)
    plt.plot(net(inputs).detach().numpy())
    plt.ylabel('Umidade')
    plt.xlabel('Index')
    plt.legend(['Targets', 'Predicts'], loc='upper left')
    plt.title('Predições de validação', loc='center')
    plt.xticks([0, 20, 40, 60, 80, 100, 120])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    plt.text(95, 0.75, 'R² =' + str(round(metrics['r2_val'], 2)))
    plt.text(95, 0.72, 'MAE =' + str(round(metrics['mae_val'], 2)))
    plt.text(95, 0.69, 'RMSE =' + str(round(metrics['rmse_val'], 2)))
    plt.savefig(path+'/'+name+'(Validação).png')
    plt.clf()


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def lossesPlots(losses, name, path):
    plt.plot(losses)
    plt.ylabel('Loss')
    plt.title('Variação de loss', loc='center')
    plt.savefig(path+'/'+name+'(Losses).png')
    # plt.show()
    plt.clf()

def scatter_plot(predicted, target,name, path, test=False):
    plt.scatter(target, predicted)
    plt.ylabel('Predicted')
    plt.xlabel('Target')
    if test: title = 'Target x Predicted - Test' 
    else: title ='Target x Predicted - Treino'
    plt.title(title, loc='center')
    plt.savefig(path+'/'+name+'Scatter.png')
    # plt.show()
    plt.clf()

def train_no_val(ins, path, config, epochs, model, loss_fn, opt, data, scheduler, check, paciencia=9):
    epochs_losses = []
    best_loss = 9999999999999
    contador = 0
    model.train()
    
    for epoch in range(epochs):
        batches_losses = []
        preds = []
        targets = []
        
        batch_interaction(model, loss_fn, opt, data, batches_losses,
                        no_val=True, preds=preds, targets=targets)
        
        epoch_loss, current_lr = get_losses_and_lr(opt, scheduler, epochs_losses, batches_losses)

        print_epoch_status(check, epoch, epoch_loss, current_lr)
        
        with torch.no_grad():
            model.eval()
            
            # if best do far, save the model
            if(best_loss > epoch_loss):
                saved = save_model(ins, path, config, model, preds, 
                        targets, None, None, epoch_loss, val=False)
                
                best_loss, best_model, metrics, = saved
                contador = 0
            
            # if not improving stop
            elif (contador >= paciencia):
                print('{} \n {}'.format(best_loss, epoch_loss))
                break
            
            else:
                contador += 1

    return epoch_loss, np.array(epochs_losses).mean(), epochs_losses, metrics, best_model


def split_data_no_val(df, ins):
    data = set_data_thetas_targets(df, ins)
    data_train = data.values
    inputs_data = data_train[0:, :-1].astype(np.float32)
    umidade = data_train[0:, -1].astype(np.float32)
    umidade = umidade.reshape(-1, 1)

    train_dl, inputs_data, umidade = transformDl(inputs_data, umidade)
    return train_dl, inputs_data.shape[1], inputs_data, umidade
