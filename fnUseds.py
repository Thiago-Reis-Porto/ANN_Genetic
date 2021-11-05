import numpy as np
import torch
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import json
import datetime as dt
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import chain, combinations

def dirStart():
  tsNow = dt.datetime.now()
  dirName = str(tsNow.year) + '-' + str(tsNow.month) + '-' + str(tsNow.day) + '-' + str(tsNow.hour - 3) + '-' + str(tsNow.minute) + '-' + str(tsNow.second)
  os.mkdir(dirName)
  os.mkdir(dirName+'/'+'plots')
  return dirName


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

def transformDl( inputs, targets ):
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)

    data2 = TensorDataset(inputs, targets)
    batch_size = 100
    train_dl = DataLoader(data2, batch_size, shuffle=True)
    return train_dl, inputs, targets

def split_data(df, ins):
    umidade = df[['umidade']]
    tetas = df[['tetas']]
    data = pd.DataFrame(df[ins])
    data['tetas'] = tetas
    data['umidade'] = umidade
    train_split, val = train_test_split(data, stratify=data.tetas, test_size=0.1)
    data_train = train_split.values
    inputs_data = data_train[0:, :-1].astype(np.float32)
    data_val = val.values
    val_inputs_data = data_val[0:, :-1].astype(np.float32)
    umidade = data_train[0:, -1].astype(np.float32)
    umidade_val = data_val[0:, -1].astype(np.float32)
    umidade = umidade.reshape(-1,1)
    umidade_val = umidade_val.reshape(-1, 1)

    train_dl, inputs_data, umidade = transformDl(inputs_data, umidade)
    val_dl, val_inputs_data, umidade_val = transformDl(val_inputs_data, umidade_val)
    return train_dl, val_dl, inputs_data.shape[1], inputs_data, umidade, val_inputs_data, umidade_val

def prepare_test_data(df, ins):
    umidade = df[['umidade']]
    tetas = df[['tetas']]
    data = pd.DataFrame(df[ins])
    data['tetas'] = tetas
    data['umidade'] = umidade
    data = data.values
    inputs_data = data[0:, :-1].astype(np.float32)
    umidade = data[0:,-1].astype(np.float32)
    umidade = umidade.reshape(-1,1)
    #test_dl, ins, outs = transformDl(inputs_data, umidade)
    inputs = torch.from_numpy(inputs_data)
    targets = torch.from_numpy(umidade)

    return inputs, targets


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
    #plt.clf()


def stringnize(list):
    string = ''
    for word in list:
        string += word + '_'
    return string[:-1]

def train(ins, path, config, epochs, model, loss_fn, opt, data, val_data, scheduler, check, paciencia=9):
  epochs_losses = []
  best_loss = 9999999999999
  contador = 0
  val_batches_losses = []
  val_epochs_losses = []
  val_best_loss = 9999999999999999
  model.train()
  for epoch in range(epochs):
    batches_losses = []
    for xb, yb in data:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        batches_losses.append(loss.detach().item())
    epoch_loss = np.array(batches_losses).mean()
    epochs_losses.append(epoch_loss)
    scheduler.step(epoch_loss)
    current_lr = opt.param_groups[0]['lr']
    if(epoch % check == 0):
        print('In epoch {}, loss:{}, lr:{}'.format(epoch, epoch_loss, current_lr))
    with torch.no_grad():
        model.eval()
        preds = []
        targets = []
        for val_xb ,val_yb in val_data:
            val_pred = model(val_xb)
            val_loss = loss_fn(val_pred, val_yb)
            val_batches_losses.append(val_loss.detach().item())
            preds = preds + val_pred.detach().numpy().tolist()
            targets = targets + val_yb.detach().numpy().tolist()
        val_epoch_loss = np.array(val_batches_losses).mean()
        val_epochs_losses.append(val_epoch_loss)
        if(val_best_loss > val_epoch_loss):
            val_best_loss = val_epoch_loss
            best_model = model
            torch.save(model.state_dict(), os.path.join(path+config+'/'+ins+'/', 'best_model.pth'))

            r2, mae, rmse = error_metrics(preds, targets)
            metrics = {'r2': r2, 'mae': mae, 'rmse': rmse}
            r2_val, mae_val, rmse_val = error_metrics(val_pred, val_yb)
            metrics_val = {'r2_val': r2_val, 'mae_val': mae_val, 'rmse_val': rmse_val}
            df2 = pd.DataFrame([metrics, metrics_val])
            df2.to_csv(os.path.join(path+config+'/'+ins+'/', 'best_metrics.csv'), index= False)

            with open(os.path.join(path+config+'/'+ins+'/', 'args.json'), mode= 'w') as file:
                json.dump({'config': config, 'Columns': ins }, file)

            contador = 0
        elif(contador >= paciencia):
            print('{} \n {}'.format(val_best_loss, val_epoch_loss))
            break
        else:
            contador += 1
  return epoch_loss, np.array(epochs_losses).mean(), epochs_losses, val_epochs_losses, val_best_loss, metrics, best_model, metrics_val

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
  plt.text(800,0.95, 'R² ='+ str(round(metrics['r2'],2)), fontsize=12)
  plt.text(800,0.90, 'MAE ='+ str(round(metrics['mae'],2)), fontsize=12)
  plt.text(800,0.85, 'RMSE ='+ str(round(metrics['rmse'],2)), fontsize=12)
  plt.savefig(path+'/'+name+'.png')
  plt.clf()
  lossesPlots(losses, name, path)
  if(metrics_val != None):
    plotsVal(name, net, inputs_val, targets_val, metrics_val, path)


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
  #plt.show()
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
        for xb, yb in data:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            batches_losses.append(loss.detach().item())
            preds += pred.detach().numpy().tolist()
            targets += yb.detach().numpy().tolist()
        epoch_loss = np.array(batches_losses).mean()
        epochs_losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        current_lr = opt.param_groups[0]['lr']
        if (epoch % check == 0):
            print('In epoch {}, loss:{}, lr:{}'.format(epoch, epoch_loss, current_lr))
        with torch.no_grad():
            model.eval()
            if(best_loss > epoch_loss):
                best_loss = epoch_loss
                best_model = model
                torch.save(model.state_dict(), os.path.join(path + config + '/' + ins + '/noVal/best_model_no_val.pth'))

                r2, mae, rmse = error_metrics(preds, targets)
                metrics = {'r2': r2, 'mae': mae, 'rmse': rmse}
                df2 = pd.DataFrame([metrics])
                df2.to_csv(os.path.join(path + config + '/' + ins + '/noVal/', 'best_metrics_no_val.csv'), index=False)
                with open(os.path.join(path + config + '/' + ins + '/noVal/', 'args_no_val.json'), mode='w') as file:
                    json.dump({'config': config, 'Columns': ins}, file)
                contador = 0
            elif (contador >= paciencia):
                print('{} \n {}'.format(best_loss, epoch_loss))
                break
            else:
                contador += 1
    return epoch_loss, np.array(
        epochs_losses).mean(), epochs_losses, metrics, best_model

def split_data_no_val(df, ins):
    umidade = df[['umidade']]
    tetas = df[['tetas']]
    data = pd.DataFrame(df[ins])
    data['tetas'] = tetas
    data['umidade'] = umidade
    data_train = data.values
    inputs_data = data_train[0:, :-1].astype(np.float32)
    umidade = data_train[0:, -1].astype(np.float32)
    umidade = umidade.reshape(-1, 1)

    train_dl, inputs_data, umidade = transformDl(inputs_data, umidade)
    return train_dl, inputs_data.shape[1], inputs_data, umidade