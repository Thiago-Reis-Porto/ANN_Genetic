from sklearn.base import BaseEstimator, RegressorMixin
import torch.nn as nn
import torch
import torch.optim as optim
import fnUseds
import pandas as pd
import pathlib
import os
from sklearn.model_selection import KFold
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
import sklearn as sk
import numpy as np

device = 'cpu'

class Net(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.linear1 = nn.Linear(inputs, 20).to(device)
        self.bn1 = nn.BatchNorm1d(20).to(device)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(20, 10).to(device)
        self.bn2 = nn.BatchNorm1d(10).to(device)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(10, 1).to(device)
        self.bn3 = nn.BatchNorm1d(1).to(device)
        self.act3 = torch.sigmoid

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.act3(x)

        return x


class Model(BaseEstimator, RegressorMixin):
    def __init__(self, path, net=Net, ins=[''], lr=0.1, momentum=0.1, seed=1, eps=1000):
        self.path = path
        #self.data = data
        self.net = net
        self.ins = ins
        self.lr = lr
        self.momentum = momentum
        self.seed = seed
        self.eps = eps


    def fit(self, x, y=None, **kwargs):

        fnUseds.set_reproducible_seed(self.seed)
        ins2 = list(self.ins)
        self.ins2 = list(self.ins)
        self.netTrained_ = self.net(len(ins2) + 1)
        ins = fnUseds.stringnize(ins2)

        opt = optim.SGD(self.netTrained_.parameters(), lr=self.lr, momentum=self.momentum)
        lr_plateau = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5, min_lr=1e-12)
        criterion = torch.nn.functional.mse_loss
        train_data, val_data, shape, x, y, val_x, val_y = fnUseds.split_data(x, ins2)
        self.config_ = f'seed_{self.seed}_lr_{self.lr}_momentum_{self.momentum}_epochs_{self.eps}'


        os.makedirs(self.path + self.config_, exist_ok=True)
        os.makedirs(self.path + self.config_ + '/' + ins, exist_ok=True)
        lo, er, losses, val_losses, valbest, self.error_, best_model, error_val = fnUseds.train(ins, self.path, self.config_, self.eps, self.netTrained_, criterion, opt, train_data, val_data, lr_plateau, 200, paciencia=9)
        fnUseds.mkPlots(best_model, losses, x, y, self.error_,
                self.path + self.config_+'/'+ins+'/', error_val, val_x, val_y)
        return self

    def predict(self, x=None, y=None):
        data = pd.DataFrame(x[self.ins2])
        data['tetas'] = x['tetas']
        data_np = data.values
        inputs_data = data_np.astype(np.float32)
        inputs = torch.from_numpy(inputs_data)


        return self.netTrained_(inputs)

    def score(self, x, y=None):
        predicts = self.predict(x)
        umidade = y.values
        umidade = umidade.astype(np.float32)
        umidade = torch.from_numpy(umidade)
        errors = fnUseds.error_metrics(predicts.detach().numpy().tolist(), umidade.detach().numpy().tolist())
        return errors[0]

    def test(self, x, y=None):
        ins2 = list(self.ins)
        ins = fnUseds.stringnize(ins2)
        x_test, y_test = fnUseds.prepare_test_data(x, ins2)

        with torch.no_grad():
            self.netTrained_.eval()
            predicts = self.predict(x_test)
            r2, mae, rmse= fnUseds.error_metrics(predicts, y_test)
            metrics = {'r2': r2, 'mae': mae, 'rmse': rmse}
            fnUseds.plot_test(metrics, predicts, y_test, self.path+self.config_+'/'+ins+'/', 'Test_TargetsXPredicts')
            df2 = pd.DataFrame([metrics])
            df2.to_csv(os.path.join(self.path + self.config_ + '/' + ins + '/', 'test_metrics.csv'), index=False)

    def train_no_val(self, x, y=None):
        fnUseds.set_reproducible_seed(self.seed)
        ins2 = list(self.ins)
        self.netTrained_ = self.net(len(ins2) + 1)
        ins = fnUseds.stringnize(ins2)

        opt = optim.SGD(self.netTrained_.parameters(), lr=self.lr, momentum=self.momentum)
        lr_plateau = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5, min_lr=1e-12)
        criterion = torch.nn.functional.mse_loss
        data, shape, x, y = fnUseds.split_data_no_val(x, ins2)
        self.config_ = f'seed_{self.seed}_lr_{self.lr}_momentum_{self.momentum}_epochs_{self.eps}'
        os.makedirs(self.path + self.config_, exist_ok=True)
        os.makedirs(self.path + self.config_ + '/' + ins, exist_ok=True)
        os.makedirs(self.path + self.config_ + '/' + ins + '/noVal', exist_ok=True)

        lo, er, losses, self.error_, best_model = fnUseds.train_no_val(ins, self.path,
                                                                            self.config_, self.eps,
                                                                            self.netTrained_,
                                                                            criterion, opt,
                                                                            data,
                                                                            lr_plateau, 200,
                                                                            paciencia=9)
        fnUseds.mkPlots(best_model, losses, x, y, self.error_, self.path + self.config_+'/'+ins+'/noVal/')
        return self



def scorer(model, x, y):
    return model.score(x, y)


def main():

    path = pathlib.Path(__file__).parent.absolute()
    os.chdir(path)
    name = fnUseds.dirStart()
    path = str(path) + '/' + name + '/plots/'
    df = pd.read_csv('tabela-7-tensoes-treinamento.csv', sep=';')
    #df_test = pd.read_csv('tabela-7-tensoes-teste.csv', sep=';')
    labels = df.columns.values
    ins = fnUseds.powerset(labels[:13])

    #tetas = [0, 1, 6, 10, 33, 100, 1500]
    #df_with_teste = []
    #for teta in tetas:
    #    df_with_teste.append(df_test.loc[df_test['tetas'] == teta])


    #seeds = [22738, 26915, 30093, 41938, 42786, 48417, 68097, 81705, 8581, 878, 98186]


    #for seed in seeds:
    #    net = Model(path, Net, ['argila', 'ds', 'mi', 'c', 'curvature', 'plancurve'], 0.2, 0.2, seed, 5000)
    #    net.train_no_val(df)
    #    print('Seed atual: {}'.format(seed))
    #    predicts_all = []
    #    targets_all = []
    #    for df_teste in df_with_teste:
    #        x_test, y_test = fnUseds.prepare_test_data(df_teste, ['argila', 'ds', 'mi', 'c', 'curvature', 'plancurve'])
    #        predicts = net.predict(x_test)
    #        r2, mae, rmse = fnUseds.error_metrics(predicts.detach().numpy(), y_test)
    #        predicts_all += predicts.detach().numpy().tolist()
    #        targets_all += y_test
    #        teta = df_teste.iloc[0]['tetas']
    #        print('   R² da MLP com teta {}: {}'.format(teta, r2))
    #    r2, mae, rmse = fnUseds.error_metrics(predicts_all, targets_all)
    #    print('   R² da MLP Total: {}'.format(r2))

        #net.test(df_test)



    #for seed in seeds:
    #    net = Model(path, Net, ['argila', 'ds', 'ma', 'soilslope', 'plancurve'], 0.2, 0.1, seed, 1000)
    #    net.fit(df)


    #seeds = np.random.randint(1,1000, size=20)
    #lrs = (0.1, 0.2, 0.01, 0.02, 0.001, 0.002, 0.025, 0.0025)
    #momentums = (0.1, 0.2, 0.01, 0.02, 0.001, 0.002, 0.025, 0.0025)

    #params = {'lr': lrs, 'ins': (list(ins)), 'seed': tuple(seeds), 'momentum': momentums, 'eps': (1000, 10000, 5000) }
    #teste = sk.model_selection.RandomizedSearchCV(net, params, n_iter=2, refit=True, cv=10 )
    #final = teste.fit(df)
    #print('-----------Teste Finalizado-------------')
    #print('Melhores parametros:',final.best_params_)
    #print('Melhores scores:', final.best_score_)
    param_grid = {'ins': Categorical(choices=list(ins)), 'seed': Integer(lower=98186, upper=98186)}

    cv = KFold(n_splits=5, shuffle=True)

    model = Model(path, Net, ['argila', 'ds', 'ma', 'soilslope', 'plancurve'], 0.2, 0.1, 98186, 1000)

    evolved_estimator = GASearchCV(estimator=model,
                                   cv=cv,
                                   scoring=scorer,
                                   population_size=20,
                                   generations=10,
                                   tournament_size=3,
                                   elitism=True,
                                   crossover_probability=0.8,
                                   mutation_probability=0.1,
                                   param_grid=param_grid,
                                   criteria='max',
                                   algorithm='eaMuPlusLambda',
                                   verbose=True)

    final = evolved_estimator.fit(df, df['umidade'])
    print(final.best_params_)
    print(final.best_estimator_)



main()
