import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_line_numpy(line, start=11, end=91, step=4):
    '''
    输入一行，返回处理后的np数组
    默认start是11，end是91，step是4
    start=10 end=69 step=3
    '''

    np_line = np.zeros([20])
    index = 0
    for symbol, num in zip(line[start:end:step], line[start + 1:end:step]):
        if symbol == '-':
            statu = -1
        else:
            statu = 1
        np_line[index] = statu * int(num)
        index = index + 1
    return np_line


def getPSSM(filepath):
    with open(filepath, 'r') as f:
        for i in range(2):  # 前2行无用
            f.readline()
        line = f.readline()  # 第三行判断
        if line[12] == 'A':
            line = f.readline()
            pssm_np = get_line_numpy(line)
            line = f.readline()
            while line != '\n':
                pssm_np = np.vstack([pssm_np, get_line_numpy(line)])
                line = f.readline()
        else:
            line = f.readline()
            pssm_np = get_line_numpy(line, 10, 69, 3)
            line = f.readline()
            while line != '\n':
                pssm_np = np.vstack([pssm_np, get_line_numpy(line, 10, 69, 3)])
                line = f.readline()
        f.close()
    return pssm_np


def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


train = pd.read_csv('Data/train.csv')
traindir = train['pssm'].values

test = pd.read_csv('Data/test.csv')
testdir = test['pssm'].values

print("data done")

path1 = 'Data/PSSM/trainPSSM/'
path2 = 'Data/PSSM/testPSSM/'
num = 0

X_train = []
for i in range(len(traindir)):
    temp = np.zeros((50, 20))
    datapath = path1 + traindir[i]
    pssm_np = getPSSM(datapath)
    idx = pssm_np.shape[0]
    temp[0:idx, :] = pssm_np
    X_train.append(temp)

X_test = []
for i in testdir:
    temp = np.zeros((50, 20))
    datapath = path2 + i
    pssm_np = getPSSM(datapath)
    idx = pssm_np.shape[0]
    temp[0:idx, :] = pssm_np
    X_test.append(temp)

X_train.extend(X_test)
all_pssm = X_train
print(len(all_pssm))

tensor_pssm = torch.Tensor(all_pssm)

"""Load preprocessed data."""
dir_input = ('Data/rdkit/radius2/')
compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)

labels = load_tensor(dir_input + 'labels', torch.LongTensor)
fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
n_fingerprint = len(fingerprint_dict)

"""Create a dataset and split it into train/dev/test."""
dataset = list(zip(compounds, adjacencies,tensor_pssm, labels))
dataset = shuffle_dataset(dataset, 1234)
dataset_train, dataset_test = split_dataset(dataset, 0.85)


class AttentionPrediction(nn.Module):
    def __init__(self):
        super(AttentionPrediction, self).__init__()
        self.embed_fingerprint = torch.nn.Embedding(n_fingerprint, dim)
        self.W_gnn = torch.nn.ModuleList([torch.nn.Linear(dim, dim)
                                          for _ in range(layer_gnn)])

        self.final_h = nn.Parameter(torch.randn(1, 75 * 4 + dim))
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=36,
                                     kernel_size=(1, 20),
                                     stride=(1, 1),
                                     padding=(0, 0),
                                     )
        self.bn= torch.nn.BatchNorm2d()

        self.rnn = torch.nn.GRU(input_size=36 * 50 * 1,
                                hidden_size=75,
                                num_layers=2,
                                bidirectional=True,
                                dropout=0.5
                                )

        self.W_s1 = nn.Linear(75 * 4 + dim, da)
        self.W_s2 = nn.Linear(da, 1)

        self.fc1 = torch.nn.Linear(75 * 4 + dim, units)

        self.fc2 = torch.nn.Linear(units, 2)

    def gnn(self, xs, A, layer):
        gnn_median = []
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
            temp = torch.mean(xs, 0)
            temp = temp.squeeze(0)
            temp = temp.unsqueeze(0)
            gnn_median.append(temp)
        return gnn_median

    def cnn(self, x):
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.bn(x)
        x = torch.nn.ReLU()(x)
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(0)
        output, hidden = self.rnn(x)
        return torch.cat([hidden[-1], hidden[-2], hidden[-3], hidden[-4]], dim=1)


    def selfattention(self, cat_vector):

        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(cat_vector)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=1)

        return attn_weight_matrix

    def forward(self, gnn_peptide, gnn_adjacencies, cnn_pssm):

        """Peptide vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(gnn_peptide)
        gnn_vectors = self.gnn(fingerprint_vectors, gnn_adjacencies, layer_gnn)
        self.feature1 = gnn_vectors

        """Peptide vector with CNN based on PSSM."""
        cnn_vectors = self.cnn(cnn_pssm)
        self.feature2 = cnn_vectors

        """Concatenate the above three vectors and output the prediction."""
        vector = []
        for i in range(layer_gnn):
            vector.append(torch.cat([gnn_vectors[i], cnn_vectors], dim=1))
        all_vector = vector[0]
        for i in range(1, layer_gnn):
            all_vector = torch.cat((all_vector, vector[i]), 0)


        all_vector = all_vector.unsqueeze(0)

        attn_weight_matrix = self.selfattention(all_vector)
        hidden_matrix = torch.bmm(attn_weight_matrix, all_vector)

        x = torch.nn.functional.relu(self.fc1(hidden_matrix.view(1, -1)))
        label = self.fc2(x)

        prediction = torch.nn.functional.softmax(label)
        return prediction

dim = 50
layer_gnn = 4
units = 840
da = 160

model =  AttentionPrediction().to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

acc_list = []
train_loss = []
test_loss = []
min_test_acc = 0.5
for epoch in range(100):
    print('epochs:', epoch)
    total_loss = 0
    num = 0
    for i,(train_peptide,train_adjacencies,pssm_train,y) in enumerate(dataset_train,1):
        train_peptide = train_peptide.to(device)
        train_adjacencies = train_adjacencies.to(device)
        pssm_train = pssm_train.to(device)
        y = torch.Tensor([y]).long().to(device)
        output = model(train_peptide,train_adjacencies,pssm_train)
        loss = loss_function(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 500 == 0:
            print(f'[ Epoch {epoch} ',end='')
            print(f'[{i}/{len(dataset_train)}] ',end='')
            print(f'loss={total_loss / i}')

    correct =0
    total =len(dataset_test)
    print("evaluating trained model ...")
    y_pred = []
    with torch.no_grad():
        for test_peptide,test_adjacencies,pssm_test,y in dataset_test:
            test_peptide = test_peptide.to(device)
            test_adjacencies = test_adjacencies.to(device)
            pssm_test = pssm_test.to(device)
            y = torch.Tensor([y]).long().to(device)
            output = model(test_peptide,test_adjacencies,pssm_test)
#             print(output.detach().numpy())
            loss = loss_function(output,y)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(y).item()

        percent = '%.4f'%(100*correct/total)
        print (f'Test set: Accuray {correct}/{total} {percent}%')

    # acc_list.append(percent)
    # train_loss.append(total_loss)
    # model_name = 'self-attention/radius2/layer_gnn_' +str(layer_gnn) + '_dim_' + str(dim) + '_units_'+str(units) + '_da_'+str(da) + '.model'
    # if (float(acc_list[-1]) > min_test_acc):
    #     torch.save({'epoch': epoch, 'model': model.state_dict(), 'train_loss': train_loss,
    #                 'test_best_acc': percent,'test_acc':acc_list},
    #                 model_name) # 保存字典对象，里面'model'的value是模型
    #     min_test_acc = float(acc_list[-1])
