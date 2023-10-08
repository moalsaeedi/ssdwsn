# import asyncio
# from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchmetrics import Accuracy
from sklearn.preprocessing import LabelBinarizer, Binarizer, MultiLabelBinarizer, StandardScaler, MinMaxScaler, normalize
from collections import deque
from torch.distributions import Normal
from math import sqrt, exp, pi, log
import time
import re
# np1 = torch.empty(2,150,4).reshape(-1, 4)
# print(np1.shape)
# np2 = torch.empty(2,150,1).reshape(-1, 1)
# print(np2.shape)

# sumnps = torch.hstack((np1, np2))
# print(sumnps.shape)

'''
nds = Normal(0, 1)
sampval = nds.rsample(torch.Size((1,1)))
print(sampval)
# print((1/1(sqrt(2*pi)))*exp(-0.5*(((sampval-0)/1))**2))
print(nds.log_prob(sampval))

prop = [[0.3, 0.5, 0.2],
        [0.2, 0.6, 0.2]]
prop = torch.tensor(prop)
pro = torch.sigmoid(prop)
print(pro)
mlb = Binarizer(threshold=0.6)
print(mlb.fit_transform(pro.numpy()))

print(torch.tensor(0).float())
'''
obs_cols = ['port', 'intftypeval', 'datatypeval', 'distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', 'engcons', \
'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val'] 
cal_cols = ['ts', 'id']
cat_cols = ['id']
con_cols = ['id', 'port', 'intftypeval', 'datatypeval', 'distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', 'engcons', \
'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val']
observation_space = np.empty((0, len(cal_cols)+len(obs_cols))) # +1 is the energy variance (sigma**2) from the mean
ts = int(round(time.time(), 0))
obs_time = 5
state = np.array([[1691164979, 6661,0,3,1,8,8,0,719,516,0,49108,59.6219205856323,16.4905060075481,7.35539678275575,549,22233,518,20718,4,0,0,0,0],
[1691164980, 6662,0,3,1,7,7,0,781,581,0,49134,47.0059394836426,21.8822568656532,7.14754985054603,363,14948,332,13505,0,0,0,0,0],
[1691164981, 6663,0,3,1,6,6,0,786,519,0,49164,46.3039398193359,24.8692776871433,6.89856062618292,142,5699,111,4328,0,0,0,0,0],
[1691164982, 6664,0,3,1,7,7,0,667,568,0,49095,65.9999227523804,11.7182791052277,7.46523509973759,627,25947,596,24519,5,0,0,0,0],
[1691164983, 6664,0,3,1,7,7,0,667,568,0,49095,65.9999227523804,11.7182791052277,7.46523509973759,627,25947,596,24519,5,0,0,0,0],
[1691164984, 6665,0,3,1,8,8,0,716,612,0,49107,62.843918800354,15.794278360644,7.36356380186106,556,22335,525,20835,3,0,0,0,0],
])
nodes = np.array([['1.0.1'],
['1.0.2'],
['1.0.3'],
['1.0.4'],
['1.0.4'],
['1.0.5'],
])
observation_space = np.hstack((nodes, state))
obs = pd.concat([pd.DataFrame(observation_space, columns=cat_cols+con_cols)], axis=1)
obs = obs.iloc[:,1:].astype(float)
vs = torch.ones((200, 20))
print(vs.shape)

'''
attn_scores = torch.matmul(vs, vs.transpose(-2, -1)) / sqrt(5)
attn_probs = torch.softmax(attn_scores, dim=-1)
output = torch.matmul(attn_probs, vs)
print(output.shape)
splitvs = output.view(output.shape[0], 4, 5).transpose(0,1)
print(splitvs.shape)
joindsplitvs = splitvs.transpose(0, 1).contiguous().view(200, 20)
print(joindsplitvs.shape)
jj = output + joindsplitvs
print(jj.shape)
'''

pe = torch.zeros(200, 4)
position = torch.arange(0, 200, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, 4, 2).float() * -(log(10000.0) /4))

pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

x = torch.randn((200, 4))
output = x + pe
print(x.size(1))
print(output)
'''
R = np.repeat(np.mean((obs['throughput'] - obs['delay'] - obs['drpackets_val']).to_numpy().reshape(-1,1)), obs.shape[0], axis=0).reshape(-1,1)
ls = [1,2,3,4]
print(ls[-1])
print(R)
action = torch.tensor(np.array([[3, 4, 0],
                  [2, 5, 7]]), dtype=float)
softmax = nn.LogSoftmax(-1)
logprob = softmax(action)
print(f'logpob: {logprob}')
x = torch.exp(logprob).numpy()
print(x)
print(f'action: {np.argmax(x, axis=-1).reshape(-1,1)+1}')



print(obs['delay'].mean())

nodes = np.array([[500],
[200],
[250],
[352],
[457],
[2000],
])
print(np.mod(nodes, 200))
R = obs['throughput'].sub(obs['delay']).sub(obs['engcons']).sub(obs['drpackets_val']).to_numpy().reshape(-1,1) - \
            np.sqrt(np.square(obs['engcons'].to_numpy().reshape(-1,1) - np.mean(obs['engcons']))) + \
            np.sqrt(np.square(obs['batt'].to_numpy().reshape(-1,1) - np.mean(obs['batt']))) + \
            (np.mean(obs['batt']) - obs['batt'].to_numpy().reshape(-1,1))*(obs['distance'].to_numpy().reshape(-1,1) - np.mean(obs['distance']))
R = np.nan_to_num(R, nan=0)
print(f'R:\n{R}')
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)
    
emb = Embedder(24, 50)
print(emb(torch.from_numpy(np.array([['Hallo']]))))
print(torch.zeros(10,3))
loc = torch.tensor([[1, 2],
                    [2, 3],
                    [3, 4]]).float()
softmax = nn.Softmax(dim=-1)
scores = torch.mm(loc, loc.transpose(0,1))
atten = softmax(scores)
print(atten)
print(torch.mm(atten, loc))
# print(torch.mm(atten, loc).shape)
print(torch.tensor(np.ones_like(loc))*0.5)
print(1/(1+np.exp(-4555656555.5665554544545)))
data = pd.DataFrame(state, columns=con_cols)
print(data.to_numpy().mean(axis=-1, keepdims=True))
nodes = np.array([['1.0.1'],
['1.0.2'],
['1.0.3'],
['1.0.4'],
['1.0.4'],
['1.0.5'],
])
logsig = nn.LogSigmoid()
arr = np.array([[-4545484784]], dtype=float)
action = torch.tanh(torch.from_numpy(arr))
print(action)
log_prob = logsig(action)
print(log_prob)
print(torch.exp(log_prob))
# log_prob = torch.log(action).sum(dim=-1, keepdim=True)
# print(log_prob)

print(1 / (1 + np.exp(np.log(np.sqrt(np.square(1+3000))))))
observation_space = np.hstack((nodes, state))
data = pd.concat([pd.DataFrame(observation_space, columns=cat_cols+con_cols)], axis=1)
print(data.get(['port', 'intftypeval']).to_numpy().max())
data = data.iloc[:,1:].astype(float)
print(data.get(['batt', 'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val']).max().to_numpy())
sig = nn.LogSigmoid()
x = sig(torch.from_numpy(data.to_numpy()).float())
prob = torch.exp(x)
print(prob)
action = action = MinMaxScaler((0,1)).fit_transform(prob)
print(action)
R = (data/data.max()) * (data/data.min()) * \
            (data/data.max()) * (data/data.min())
print(f'max: {np.nan_to_num(data.to_numpy(), nan=0, posinf=0, neginf=0)}')
print(np.sum(R.to_numpy(), axis=1))
R = data['throughput']-data['delay']-data['engcons']-data['drpackets_val']-\
    np.sqrt(np.square(data['engcons']-np.mean(data['engcons'])))
print(np.nan_to_num(R, nan=0).reshape((-1, 1)))
tss = np.unique(state[:,0]).flatten().tolist()
print(state[:,0].reshape(-1,1))
idxs = np.where((state[:,0] == tss[0]))
print(idxs[0])
print(state[idxs,1:])

print(np.where(np.isin(state, tss)))
print(np.unique(state[:,1]).tolist())
print(state[:,0].astype(int).flatten().tolist())

print(f'data:\n{data}')
print(f'data to nmpy:\n{data.to_numpy()}')
print(data['ts'].unique().flatten().tolist())
print(data.shape[0])
print(f'data ts reshape:\n{data["ts"].to_numpy().reshape(-1,1)}')
# data.to_csv('outputs/dataset.csv', mode='w+', sep='\t', index=False)
for cat in cat_cols:
    data[cat] = data[cat].astype('category')
for con in con_cols:
    data[con] = data[con].astype('float')
cat_szs = [len(data[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]

print(cat_szs)
print(emb_szs)
print(data.shape)
print(data[1:].to_numpy())
print(data.dtypes)
embeds = nn.ModuleList(nn.Embedding(ni,nf) for ni,nf in emb_szs)
print(embeds)
embedings = []
x_cat = data['id'].to_numpy().reshape(-1,1)
print(x_cat.dtype)
print(x_cat)
for i,e in enumerate(embeds):
    lookup_tensor = torch.tensor(x_cat[:,i], dtype=torch.long)

    embedings.append(e(lookup_tensor))

x0 = np.empty((0, 6))
x1 = np.array([0.003, -0.233, 0.34, 0.60, 0.03, -0.001])
x2 = np.array([0.003, -0.233, 0.34, 0.60, 0.03, -0.001])
print(np.vstack((x0,x1,x2)))
R = np.array([0.003, -0.233, 0.34, 0.60, 0.03, -0.001])
R2 = np.array([0.003, -0.233, 0.34, 0.60, 0.03, -0.001, -0.9, 0.89])
D = MinMaxScaler((0,1)).fit_transform(R.reshape(-1,1))
print(D)
R = 1 / (1 + np.exp(-R))
R2 = 1 / (1 + np.exp(-R2))
print(R)
print(R2)
xx1 = torch.tensor([[-1.4531e+02, -2.8265e-12, -1.7752e-12],
        [-1.4340e+02, -1.7526e-13, -3.5154e-12],
        [-1.4477e+02, -1.3767e-12, -1.9775e-12],
        [-1.4308e+02, -1.2177e-13, -4.3009e-12],
        [-1.4305e+02, -1.1760e-13, -4.3847e-12],
        [-1.4447e+02, -1.0614e-12, -1.9004e-12],
        [-1.4301e+02, -1.1433e-13, -4.4121e-12],
        [-1.4304e+02, -1.1778e-13, -4.4571e-12],
        [-1.4318e+02, -1.3795e-13, -4.0858e-12],
        [-1.4309e+02, -1.2282e-13, -4.3116e-12]])
print(torch.exp(xx1).float())
x1 = np.array([[500, 2, 10, 50], 
               [25, 20, 35, 4], 
               [98, 33, 4, 85], 
               [50, 80, 40, 65], 
               [23, 96, 47, 78]])
print(f'max: {x1[:,0].max()}')
x2 = np.array([['0.1', '0.2', '0.3', '0.5', '0.6']])

scalerdict = { k:v for (k,v) in zip(x2.flatten().tolist(), x1[:,-2].flatten().tolist())}
print(scalerdict)
x1 = torch.tensor(x1).float()
trans = nn.LogSigmoid()
x1 = trans(x1)
print(x1)
x1 = torch.exp(x1)
# x1 = MinMaxScaler((0,1)).fit_transform(x1)
print(x1)
print(f'sum for x=0: {x1.sum(0)}')
print(f'argmax: {torch.argmax(x1, dim=-1)}')
x1 = torch.tensor([0.455, 0.55])
x1 = x1.reshape(-1,1).unsqueeze(-1)
print(x1)
print(x1.shape)
x1 = np.array([[400], 
               [1000,], 
               [200], 
               [50]])
x2 = np.array([[50], 
               [4], 
               [500], 
               [600]])
x3 = np.array([[2], 
               [20], 
               [33], 
               [80]])
print(np.hstack((x1, x2, x3)))
print(np.hstack((x1, x2, x3)).mean())
print(np.nan_to_num(np.hstack((x1, x2, x3)).sum(-1), nan=0).reshape(-1,1))
print(np.min(np.hstack((x1, x2, x3)), axis=0, keepdims=True))
print(np.min(np.hstack((x1, x2, x3)), axis=0, keepdims=True).sum(-1))
print(np.repeat(np.min(np.hstack((x1, x2, x3)), axis=0, keepdims=True).sum(-1), 4, axis=0).reshape(-1,1))
_nodes = np.array([['1.0.1'], ['1.0.2'], ['1.0.3']])
_nodes = _nodes.view(np.uint8)
nodes = torch.tensor(_nodes)
for i in range(nodes.shape[0]):
    print(i)
    print(nodes[i])
    print(''.join(c for c in nodes[i].numpy().view()))
node = [49,  0,  0,  0, 46,  0,  0,  0, 48,  0,  0,  0, 46,  0,  0,  0, 49,  0,  0,  0]
print(''.join(c for c in nodes[i].char()))
for row in range(4 - 1, -1, -1):
    print(row)
prop = np.array([[0.3, 0.5, 0.2],
        [0.2, 0.6, 0.2]])
print(np.mean(prop))
print(np.sqrt(np.square(prop - np.mean(prop))))
print(prop.std())
nsamples = deque(maxlen=2)
nsamples.append(5)
nsamples.append(6)
result = map(sum, zip(nsamples))
print(sum(nsamples))

npar = np.array([[False, True],
                 [True, True]])
print(npar)
print(npar[0].astype(int))
loc = [[-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000, -0.9913],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000],
        [-1.0000,  1.0000]]
loc = torch.tensor(loc)
print(loc.sum())
scale = [[0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010],
        [0.0010, 0.0010]]
scale = torch.tensor(scale)
dist = Normal(loc, scale)
action = dist.rsample()
print(action)
print(dist.log_prob(action).sum(-1, keepdim=True))
print(torch.sigmoid(action))
print('here.............')
print(dist.log_prob(action))

a = np.array([[0.1, 0.3], [0.5, 0.4]])
b = np.array([[0.2], [0.4], [0.1]])

c = np.array([[2, 3]])
d = np.array([[4], [5]])
print(d.mean(0))

buffer = [(a, b), (c, d)]
reshape_fn = lambda x: x.reshape(-1, 2)
batch = map(np.vstack, zip(*buffer))
a_b, c_d = map(reshape_fn, batch)
print(a_b)
print(c_d)
qe = deque(maxlen=4)
qe.extend([1,2])
qe.extend([3,5])
qe.extend([6,7])
print(list(qe)[:])
print(qe)
a = np.array([[1, 5], 
             [7, 3]])
amax = np.max(a, axis=0)
amin = np.min(a, axis=0)
res = (((a.mean() - (-1))/(1 - (-1))) * (amax - amin) + amin)
print(res)
dc = {'1.0':0, '1.1':1, '1.2':2}
print(list(dc.keys())[0])
ls = ['1.0', '1.1', '1.2']
for nd in ls:
    print(nd)
a = np.array([[0.1], [0.5]])
b = np.array([[0.2], [0.4], [0.1]])
idxs = np.where((b < 0.4))
print(idxs)
new_b = b[idxs[0],:].reshape(-1,1)
print(new_b)
res = np.where(np.isin(new_b, a))
print(res)
print(new_b[res[0],:])
print('here')
a = np.array([[400.], [403.], [406.]])
b = np.array([[0.2,0.55,0.6], [0.4,0.66,0.7], [0.1,0.55,0.5]])
print(a)
print(b)

print({A: B for A, B in zip(a.flatten(), b)})

val = int(True)
# val = ( val << 16 ) & 0xffff
scaler = MinMaxScaler((-1,1))
x1 = np.array([[400, 2, 10, 50], 
               [1000, 20, 35, 4], 
               [200, 33, 4, 500], 
               [50, 80, 40, 600]])
print(scaler.fit_transform(x1))
sig = nn.Sigmoid()
probabilities = torch.distributions.Normal(50, 60)
print(probabilities.rsample())
t1 = np.array([[[0.8986, -0.7279,  1.1745,  0.2611]]])
t1 = torch.tensor(t1).float()
print(sig(t1))
print(torch.tanh(t1)*torch.tensor(1))
val = int.to_bytes(4, 1, 'big', signed=False)+int.to_bytes(1, 1, 'big', signed=False)
print(bytearray(val))
print(' '.join(format(i, '08b') for i in val)) 
a = np.array([[0, 1, 2],
               [0, 2, 4],
               [0, 3, 6]])
print(np.where(a < 4, 1, -1))
t1 = np.array([[[2004.078], [2014.0343], [2025.34]]])
t1 = torch.tensor(t1).float()
print(t1.shape)
print(t1)
t1 =torch.tanh(t1)
print(t1.shape)
print(t1)
aggr = [bytearray(b'\x06'), bytearray(b'\x03'), bytearray(b'\x04')]
aggrPayload = bytearray()
for pl in aggr:
    aggrPayload += int.to_bytes(len(pl), length=2, byteorder='big')+pl
    print(aggrPayload)
l1 = [2,3,4]
print(sum(l1))
arr1 = torch.tensor(np.array([[0.2],[-.04],[0.92],[-5], [4], [-50]]))
res = torch.tanh(arr1)
# res = torch.argmax(res, dim=0)
print(res)
import re

class RegexDict(dict):

    def get_matching(self, event):
        return (self[key] for key in self if re.match(key, event))

rd = RegexDict()
rd['.*1.0.3'] = 'item3'
rd['.*1.0.2'] = 'item2'
# rd['1.0.0-1.0.2'] = 'item1'
# print(rd)
print(list(rd.get_matching('1.0.0-1.0.2')))
print(list(rd.get_matching('1.0.3')))
path = ['1.0.5','1.0.2','1.0.0']
print(path[:-1])
a = np.array([[1,2,3],[2,1,3],[2,3,5],[5,6,4]])
print(a.shape)
t = torch.from_numpy(a).unsqueeze(0)
print(t.shape)
act_nds = {'1.0.0': -1, '1.0.1': 0.45, '1.0.2':0.62, '1.0.4':0.2, '1.0.5':0.1}
routes = [['1.0.5','1.0.4','1.0.2','1.0.0'], ['1.0.5','1.0.2','1.0.0']]
# rt_weights = [[act_nds[nd] if act_nds.get(nd) else -1 for nd in rt] for rt in routes]
print(f'routes: {routes}')
# print(f'rt_weights: {rt_weights}')
weights = [sum([(((val - min(rtv)) / (max(rtv) - min(rtv))) * (1 - (-1)) + (-1))  for val in rtv])/len(rtv) for rtv in [[act_nds[nd] if act_nds.get(nd) else -1 for nd in rt] for rt in routes]]
sel_route = routes[weights.index(min(weights))]
print(f'actions: {[[act_nds[nd] if act_nds.get(nd) else -1 for nd in rt] for rt in routes]}')
print(f'weights: {weights}')
print(f'sel_route: {sel_route}')
'''
'''
n = np.array([[5000, 50, 4, 5, 23, 101, 4], [4000, 232, 4, 5, 23, 11, 4], [3000, 232, 4, 5, 23, 101, 4]])
n1 = torch.tensor([[[5000, 5000, 4, 5, 23, 101, 4], [4000, -3500, -4, 0, 23, 11, 4], [1000, 5000, 4, 5, 23, 50, 4]]])
n2 = torch.tensor([[[5000, 4250, -4, 5, 23], [4000, 232, 4, 5, 23], [3000, 232, 4, 5, 23]]])

sig = nn.Sigmoid()
# target = torch.tensor([[[1]]]).flatten()

# preds = torch.tensor([[[0.4]]]).flatten()
# accuracy = Accuracy()
# acc = accuracy(preds, target)
# print(acc)
# scaler = StandardScaler()
# sn1 = np.hstack((scaler.fit_transform(n[:,0].reshape(-1,1)), scaler.fit_transform(n[:,1].reshape(-1,1)), scaler.fit_transform(n[:,2].reshape(-1,1)), scaler.fit_transform(n[:,3].reshape(-1,1)), scaler.fit_transform(n[:,4].reshape(-1,1)), scaler.fit_transform(n[:,5].reshape(-1,1)), scaler.fit_transform(n[:,6].reshape(-1,1))))
# sn2 = scaler.fit_transform(n)
sn = sig(n1.float())
# print(sn1)
print(sn.to(int))
# maxn = n1.max(dim=1)
# minn = n1.min(dim=1)
# print(n1.shape)
# print(n2.shape)
# result = torch.cat((n1, n2), 2)
# print(result)
# print(result.shape)
# softmax = nn.Softmax(dim=1)
# print(n1)
# print(n1.shape)
# print(softmax(n1.float()))
# min_n = n.min(axis=0, keepdims=True)
# mean_n = n.mean(axis=1, keepdims=True)
# print(f'min:{min_n}')
# print(f'mean:{mean_n}')

# print(n1)
# m = torch.tanh(n1.float())
# print(m.numpy())
# print(minn.values)
# m = (m.numpy() + minn.values.numpy())*maxn.values.numpy()
# print(m)
# def run_loop_in_process(i):
#     async def subprocess_async_work():
#         print(i)
#         await asyncio.sleep(5) #whatever async code you need
#     asyncio.run(subprocess_async_work())

# async def main():
#     loop = asyncio.get_running_loop()
#     pool = ProcessPoolExecutor()
#     tasks = [loop.run_in_executor(pool, run_loop_in_process, i) for i in range(5)]
#     await asyncio.gather(*tasks)

# if __name__ == "__main__":
#     asyncio.run(main())

'''