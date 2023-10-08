import asyncio
import nest_asyncio
nest_asyncio.apply()
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict
from math import floor, sqrt, log, cos, sin
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal
from os import cpu_count, path
from ssdwsn.util.utils import quietRun, CustomFormatter
# ssdwsn libraries

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

num_gpus = T.cuda.device_count()
device = f'cuda:{num_gpus-1}' if T.cuda.is_available() else 'cpu'

class NxtHop_Model(nn.Module):
    """ANN model to predict the next-hop route"""
    def __init__(self, emb_szs, n_cont, output_dim, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList(nn.Embedding(ni,nf) for ni,nf in emb_szs)
        self.em_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)        
        
        layerlist = []
        n_emb = sum([nf for ni,nf in emb_szs])
        n_in = n_emb + n_cont
        
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        
        layerlist.append(nn.Linear(layers[-1], output_dim))
        # for _ in output_dim:
        #     layerlist.append(nn.Linear(layers[-1], 1))
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        embedings = []
        for i,e in enumerate(self.embeds):
            embedings.append(e(x_cat[:,i]))

        x = T.cat(embedings,1)
        x = self.em_drop(x)
        
        x_cont = self.bn_cont(x_cont)
        x = T.cat([x,x_cont], 1)
        x = self.layers(x)
        return x 
    
class LSTM(nn.Module):
    """LSTM Flow-rules Setup Prediction Model - Regression"""        
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super().__init__()
        
        # Number of input dimensions
        self.input_dim = input_dim
        # Number of output dimensions
        self.output_dim = output_dim
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(input_size=input_dim , hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x):  
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
        # Initialize hidden state with zeros
        h_0 = Variable(T.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(device))
        
        c_0 = Variable(T.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(device))
            
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_dim)   
        out = self.fc(h_out)
        # self.lstm.flatten_parameters()
        # _, (hidden, ) = self.lstm(x)
        # out = hidden[-1]
        return out    
    
class _LSTM_Encoder(nn.Module):
    """LSTM Time Series Encoder - classification"""        
    def __init__(self, input_dim, hidden_dim, num_layers=1, p=0.5):
        super().__init__()
        
        # Number of input dimensions
        self.input_dim = input_dim
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(input_size=input_dim , hidden_size=hidden_dim, num_layers=num_layers)
    
    def forward(self, x):                
        lstm_out, self.hidden = self.lstm(x.view(x.shape[0], x.shape[1], self.input_dim))
        return lstm_out, self.hidden
    
    def init_hidden(self, batch_size):
        return (T.zeros(self.num_layers, batch_size, self.hidden_dim), 
                T.zeros(self.num_layers, batch_size, self.hidden_dim))
    
class _LSTM_Decoder(nn.Module):
    """LSTM Time Series Decoder - classification"""        
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, p=0.5):
        super().__init__()
        
        # Number of input dimensions
        self.input_dim = input_dim
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of output dimensions
        self.output_dim = output_dim
        # Number of hidden layers
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(input_size=input_dim , hidden_size=hidden_dim, num_layers=num_layers)

        # Linear
        self.linear = nn.Linear(hidden_dim, output_dim)
        
        # Softmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, encoder_hs):  
        print(x)              
        print(encoder_hs)
        lstm_out, self.hidden = self.lstm(x, encoder_hs)
        out = self.softmax(self.linear(lstm_out[0]))        
        return out, self.hidden
    
class LSTM_Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, p=0.5):
        super().__init__()
        
        self.encoder = _LSTM_Encoder(input_dim, hidden_dim, num_layers, p)
        self.decoder = _LSTM_Decoder(input_dim, hidden_dim, output_dim, num_layers, p)

class Transformer(nn.Module):
  # d_model : number of features
    def __init__(self,feature_size, output_size, num_headers=2, num_layers=3, dropout=0):
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_headers, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size, output_size)
        # self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (T.triu(T.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):      
        # if isinstance(src, np.ndarray):
        #     src = T.from_numpy(src).to(device)
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src,mask)
        # output = self.softmax(self.decoder(output))
        output = self.decoder(output)
        return output
  
class SAC_DQN(nn.Module):
    """Reinforcement Deep-Q Learning Network (to predict the reward of (observation+action))
    Args:
        nn (_type_): Neural Network
    """
    def __init__(self, obs_size, hidden_size, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),           
            nn.Linear(hidden_size, 1)
        ).to(device)
        self.softmax = nn.Softmax(dim=0)
    def forward(self, obs, action):
        if isinstance(obs, np.ndarray):
            obs = T.from_numpy(obs).to(device)
        # obs = T.tensor(obs).float().to(device)
        if isinstance(action, np.ndarray):
            action = T.from_numpy(action).to(device)
        # action = T.tensor(action).float().to(device)        
        # in_vector = T.hstack((obs.squeeze(), action.squeeze()))
        in_vector = T.cat((obs, action), dim=-1).float()
        return self.net(in_vector) #ERROR TODO return T._C._nn.linear(input, weight, bias) RuntimeError: mat1 dim 1 must match mat2 dim 0
    
class SAC_GradientPolicy(nn.Module):
    """Gradient Policy Network (to predict an action following a normal distribution (infinit possible of actions) of an observation)
    Args:
        nn (_type_): neural network
    """
    def __init__(self, obs_size, hidden_size, action_dim):
        super().__init__()

        # self.max = T.from_numpy(max).to(device)

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(device)
        # get Normal distribution elements (mu and std) of possible actions
        self.linear_mu = nn.Linear(hidden_size, action_dim).to(device)
        self.linear_std = nn.Linear(hidden_size, action_dim).to(device)
        # self.linear_log_std = nn.Linear(hidden_size, action_dim)    

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
            # x = T.tensor(x).float().to(device)        
        x = self.net(x.float())

        loc = self.linear_mu(x)  
        loc = T.tanh(loc)      
        scale = self.linear_std(x)
        scale = F.softplus(scale) + 1e-3

        
        # action = T.sigmoid(action)
        return loc, scale

class A2C_DQN(nn.Module):
    """Reinforcement Deep-Q Learning Network (to predict the reward of (observation+action))
    Args:
        nn (_type_): Neural Network
    """
    def __init__(self, hidden_size, obs_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device)
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
        # x = T.tensor(x).float().to(device)
        return self.net(x.float())

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='outputs/tmp/sac'):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = T.from_numpy(state).to(device)

        if isinstance(state, np.ndarray):
            action = T.from_numpy(action).to(device)
        print(f'state:\n{state}\n action:\n{action}\n merge:\n{T.cat([state.float(), action.float()], dim=-1)}\n')
        action_value = self.fc1(T.cat([state.float(), action.float()], dim=-1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value.float())
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
    
class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='outputs/tmp/sac'):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = T.from_numpy(state).to(device)
        state_value = self.fc1(state.float())
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value.float())
        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
    
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=2, name='actor', chkpt_dir='outputs/tmp/sac'):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = path.join(self.checkpoint_dir, name+'_sac')
        self.max_aciton = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = T.from_numpy(state).to(device)
        prob = self.fc1(state.float())
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        action = T.tanh(actions)*T.tensor(self.max_aciton).to(device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(-1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class REINFORCE_GradientPolicy(nn.Module):
    """Gradient Policy Network (to predict an action following a normal distribution (infinit possible of actions) of an observation)
    Args:
        nn (_type_): neural network
    """
    def __init__(self, obs_dim, hidden_size, action_dim):
        super().__init__()

        # self.max = T.from_numpy(max).to(device)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(device)
        # get Normal distribution elements (mu and std) of possible actions
        self.linear_mu = nn.Linear(hidden_size, action_dim).to(device)
        self.linear_std = nn.Linear(hidden_size, action_dim).to(device)
        # self.linear_log_std = nn.Linear(hidden_size, action_dim)    
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
            # x = T.tensor(x).float().to(device)        
        x = self.net(x.float())

        loc = self.linear_mu(x)
        loc = T.tanh(loc) * 1
        scale = self.linear_std(x)
        scale = F.softplus(scale) + 1e-3

        return loc, scale
        
class TD3_DQN(nn.Module):

    def __init__(self, hidden_size, obs_size, out_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + out_dims, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = T.from_numpy(state).to(device)
        if isinstance(action, np.ndarray):
            action = T.from_numpy(action).to(device)
        # in_vector = T.hstack((state, action))
        in_vector = T.cat((state, action), dim=2)
        return self.net(in_vector.float())

class TD3_GredientPolicy(nn.Module):

    def __init__(self, hidden_size, obs_size, out_dims, min, max):
        super().__init__()
        self.min = T.from_numpy(min).to(device)
        self.max = T.from_numpy(max).to(device)
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dims),
            nn.Tanh()
        )

    def mu(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
        return self.net(x.float()) * self.max

    def forward(self, x, epsilon=0.0, noise_clip=None):
        mu = self.mu(x)
        noise = T.normal(0, epsilon, mu.size(), device=device)
        if noise_clip is not None:
            noise = T.clamp(noise, -noise_clip, noise_clip)

        mu = mu + noise
        action = T.max(T.min(mu, self.max), self.min)
        action = action.detach().cpu().numpy()
        return action

class PPO_GradientPolicy(nn.Module):
    """Gradient Policy Network (to predict an action following a normal distribution (infinit possible of actions) of an observation)
    Args:
        nn (_type_): neural network
    """
    def __init__(self, obs_dim, hidden_size, action_dim):
        super().__init__()

        # self.max = T.from_numpy(max).to(device)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(device)
        # get Normal distribution elements (mu and std) of possible actions
        self.linear_mu = nn.Linear(hidden_size, action_dim).to(device)
        self.linear_std = nn.Linear(hidden_size, action_dim).to(device)   
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
            # x = T.tensor(x).float().to(device)        
        x = self.net(x.float())

        loc = self.linear_mu(x)
        loc = T.tanh(loc) * 1
        scale = self.linear_std(x)
        scale = F.softplus(scale) + 1e-3

        return loc, scale

class PPO_ValueNet(nn.Module):
    """Reinforcement Value Network to estimate the expected return
    Args:
        nn (_type_): Neural Network
    """
    def __init__(self, input_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(input_dim).to(device)
        self.ff = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),        
            nn.Linear(hidden_size, 1)
        ).to(device)
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
        x = self.ff(self.norm(x.float()))
        return x
    
class PPO_Policy(nn.Module):
    """Gradient Policy Network (to predict an action following a normal distribution (infinit possible of actions) of an observation)
    Args:
        nn (_type_): neural network
    """
    def __init__(self, input_dim, hidden_size, action_dim, dropout=0.1):
        super().__init__()

        self.norm = LayerNorm(input_dim).to(device)
        self.self_attn = SelfAttention(input_dim).to(device)
        # self.self_attn = MultiHeadAttention(input_dim, 4).to(device)
        self.norm1 = LayerNorm(input_dim).to(device)
        self.ff = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim)
        ).to(device)
        self.norm2 = LayerNorm(input_dim).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

        self.linear_mu = nn.Linear(input_dim, action_dim).to(device)
        self.linear_std = nn.Linear(input_dim, action_dim).to(device)

        self.fc = nn.Linear(input_dim, action_dim).to(device)
        self.logsigmoid = nn.LogSigmoid().to(device)
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
            # x = T.tensor(x).float().to(device)    
        
        x = self.norm(x.float())
        '''
        # position encoding
        pe = T.zeros_like(x).to(device)
        position = T.arange(0, pe.shape[0], dtype=T.float).unsqueeze(1)
        div_term = T.exp(T.arange(0, pe.shape[1], 2).float() * -(log(10000.0) / pe.shape[1]))
        
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        x = x + pe
        '''
        # self attention
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        '''
        log_prob = self.logsigmoid(self.fc(x))
        action = T.exp(log_prob)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # log_prob -= (2* (np.log(2) - action - F.softplus(-2*action))).sum(dim=-1, keepdim=True)
        '''
        # '''
        loc = self.linear_mu(x)
        loc = T.tanh(loc) * 1
        scale = self.linear_std(x)
        scale = F.softplus(scale) + 1e-3
        # scale = T.clip(F.softplus(scale), 1e-6, 1e-3)
        
        dist = Normal(loc, scale)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # log_prob -= (2* (np.log(2) - action - F.softplus(-2*action))).sum(dim=-1, keepdim=True)
        # entropy = dist.entropy().sum(dim=-1, keepdim=True)
        # '''
        return log_prob, action

class PPO_Att_ValueNet(nn.Module):
    """Reinforcement Value Network to estimate the expected return
    Args:
        nn (_type_): Neural Network
    """
    def __init__(self, input_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(input_dim).to(device)
        self.ff = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),        
            nn.Linear(hidden_size, 1)
        ).to(device)
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
        x = self.ff(self.norm(x.float()))
        return x
   
class PPO_Att_Policy(nn.Module):
    """Gradient Self-Attention Policy Network (to predict an action following a normal distribution (infinit possible of actions) of an observation)
    Args:
        nn (_type_): neural network
    """
    def __init__(self, input_dim, hidden_size, action_dim, dropout=0.1):
        super().__init__()

        self.norm = LayerNorm(input_dim).to(device)
        self.self_attn = SelfAttention(input_dim).to(device)
        # self.self_attn = MultiHeadAttention(input_dim, 4).to(device)
        self.norm1 = LayerNorm(input_dim).to(device)
        self.ff = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim)
        ).to(device)
        self.norm2 = LayerNorm(input_dim).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

        self.linear_mu = nn.Linear(input_dim, action_dim).to(device)
        self.linear_std = nn.Linear(input_dim, action_dim).to(device)

        self.fc = nn.Linear(input_dim, action_dim).to(device)
        self.logsigmoid = nn.LogSigmoid().to(device)
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
            # x = T.tensor(x).float().to(device)    
        
        x = self.norm(x.float())
        '''
        # position encoding
        pe = T.zeros_like(x).to(device)
        position = T.arange(0, pe.shape[0], dtype=T.float).unsqueeze(1)
        div_term = T.exp(T.arange(0, pe.shape[1], 2).float() * -(log(10000.0) / pe.shape[1]))
        
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        x = x + pe
        '''
        # self attention
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        '''
        log_prob = self.logsigmoid(self.fc(x))
        action = T.exp(log_prob)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= (2* (np.log(2) - action - F.softplus(-2*action))).sum(dim=-1, keepdim=True)
        '''
        # '''
        loc = self.linear_mu(x)
        loc = T.tanh(loc) * 1
        scale = self.linear_std(x)
        # scale = F.softplus(scale) + 1e-3
        scale = T.clip(F.softplus(scale), 1e-3, 1e-1)
        
        dist = Normal(loc, scale)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= (2* (np.log(2) - action - F.softplus(-2*action))).sum(dim=-1, keepdim=True)
        # entropy = dist.entropy().sum(dim=-1, keepdim=True)
        # '''
        return log_prob, action

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, input_dim, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(T.ones(input_dim))
        self.bias = nn.Parameter(T.zeros(input_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        out = self.embed(x)
        return out
        
class SelfAttention(nn.Module):
    """
    Attention(Q,K,V) = softmax(QK_{T}/(d_{k})**0.5)*V
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.drout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        queries = self.query(Q)
        keys = self.key(K)
        values = self.value(V)
        scores = T.matmul(queries, keys.transpose(0, 1)) / (self.d_model ** 0.5)
        # scores = T.mm(queries, keys.transpose(0, 1))
        if mask is not None:
            print(f'mask shape:{mask.shape}')
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = self.drout(self.softmax(scores))
        weighted = T.matmul(attention, values)
        return weighted

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = T.matmul(Q, K.transpose(-2, -1)) / sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = self.softmax(attn_scores)
        output = T.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        seq_length, d_model = x.size()
        return x.view(seq_length, self.num_heads, self.d_k).transpose(0, 1)
        
    def combine_heads(self, x):
        _, seq_length, d_k = x.size()
        return x.transpose(0, 1).contiguous().view(seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class TRPO_GradientPolicy(nn.Module):
    """Gradient Policy Network (to predict an action following a normal distribution (infinit possible of actions) of an observation)
    Args:
        nn (_type_): neural network
    """
    def __init__(self, obs_dim, hidden_size, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(device)
        # get Normal distribution elements (mu and std) of possible actions
        self.linear_mu = nn.Linear(hidden_size, action_dim).to(device)
        self.linear_std = nn.Linear(hidden_size, action_dim).to(device)
        # self.linear_log_std = nn.Linear(hidden_size, action_dim)    
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
            # x = T.tensor(x).float().to(device)        
        x = self.net(x.float())

        loc = self.linear_mu(x)
        loc = T.tanh(loc) * 1
        scale = self.linear_std(x)
        scale = F.softplus(scale) + 1e-3

        return loc, scale
   
class TRPO_ValueNet(nn.Module):
    """Reinforcement Value Network to estimate the expected return
    Args:
        nn (_type_): Neural Network
    """
    def __init__(self, obs_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),           
            nn.Linear(hidden_size, 1)
        ).to(device)
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
        x = self.net(x.float())
        return x
 
class A2C_ValueNet(nn.Module):
    """Reinforcement Value Network to estimate the expected return
    Args:
        nn (_type_): Neural Network
    """
    def __init__(self, obs_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),       
            nn.Linear(hidden_size, 1)
        ).to(device)
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
        x = self.net(x.float())
        return x
    
class A2C_GradientPolicy(nn.Module):
    """Gradient Policy Network (to predict an action following a normal distribution (infinit possible of actions) of an observation)
    Args:
        nn (_type_): neural network
    """
    def __init__(self, obs_dim, hidden_size, action_dim):
        super().__init__()

        # self.max = T.from_numpy(max).to(device)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        ).to(device)
        # get Normal distribution elements (mu and std) of possible actions
        self.linear_mu = nn.Linear(hidden_size, action_dim).to(device)
        self.linear_std = nn.Linear(hidden_size, action_dim).to(device)
        # self.linear_log_std = nn.Linear(hidden_size, action_dim)    

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
            # x = T.tensor(x).float().to(device)        
        x = self.net(x.float())

        loc = self.linear_mu(x)
        loc = T.tanh(loc) * 1
        scale = self.linear_std(x)
        scale = F.softplus(scale) + 1e-3

        return loc, scale



'''************************************************************************'''
class HE_DQN(nn.Module):
    """Reinforcement Deep-Q Learning Network (to predict the reward of (observation+action))
    Args:
        nn (_type_): Neural Network
    """
    def __init__(self, hidden_size, obs_size, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),           
            nn.Linear(hidden_size, 1),
        ).to(device)
        self.softmax = nn.Softmax(dim=0)
    def forward(self, obs, action):
        if isinstance(obs, np.ndarray):
            obs = T.from_numpy(obs).to(device)
        # obs = T.tensor(obs).float().to(device)
        if isinstance(action, np.ndarray):
            action = T.from_numpy(action).to(device)
        # action = T.tensor(action).float().to(device)        
        # in_vector = T.hstack((obs.squeeze(), action.squeeze()))
        in_vector = T.cat((obs, action), dim=2)
        return self.net(in_vector.float()) #ERROR TODO return T._C._nn.linear(input, weight, bias) RuntimeError: mat1 dim 1 must match mat2 dim 0
    
class HE_GradientPolicy(nn.Module):
    """Gradient Policy Network (to predict an action following a normal distribution (infinit possible of actions) of an observation)
    Args:
        nn (_type_): neural network
    """
    def __init__(self, hidden_size, obs_size, action_dim):
        super().__init__()

        # self.max = T.from_numpy(max).to(device)

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(device)
        # get Normal distribution elements (mu and std) of possible actions
        self.linear_mu = nn.Linear(hidden_size, action_dim)
        self.linear_std = nn.Linear(hidden_size, action_dim)
        # self.linear_log_std = nn.Linear(hidden_size, action_dim)    

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
            # x = T.tensor(x).float().to(device)        
        x = self.net(x.float())

        loc = self.linear_mu(x)        
        scale = self.linear_std(x)
        scale = F.softplus(scale) + 1e-3

        dist = Normal(loc, scale)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= (2* (np.log(2) - action - F.softplus(-2*action))).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        action = T.tanh(action)
        # action = T.sigmoid(action)
        return action, log_prob, entropy
  
'''
class DQN(nn.Module):
    """Reinforcement Deep-Q Learning Network (to predict the reward of (observation+action))

    Args:
        nn (_type_): Neural Network
    """
    def __init__(self, hidden_size, obs_size, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),           
            nn.Linear(hidden_size, 1),
        )
        self.norm = nn.BatchNorm1d(obs_size)  
        self.obs_size = obs_size
        self.action_dim = action_dim

    def forward(self, obs, action):
        if isinstance(obs, np.ndarray):
            obs = T.from_numpy(obs).to(device)
        if isinstance(action, np.ndarray):
            action = T.from_numpy(action).to(device)
        # obs = self.norm(obs.float())
        # in_vector = T.hstack((obs.reshape(-1, self.obs_size), action.reshape(-1, self.action_dim)))
        in_vector = T.hstack((obs, action))
        # print(f'in_vector: {in_vector}')
        return T.sigmoid(self.net(in_vector.float())) #ERROR TODO return T._C._nn.linear(input, weight, bias) RuntimeError: mat1 dim 1 must match mat2 dim 0

class GradientPolicy(nn.Module):
    """Gradient Policy Network (to predict the action (infinit possible of actions) of an observation)

    Actions: is a multivariate normal distribution of a k-dimensional random vector X = ( X_1 , … , X_k )^T --> X ~ N(mu, seqma) 
    closest action values to mu are selected and then interpreted to the real environement action values

    Args:
        nn (_type_): neural network
    """
    def __init__(self, hidden_size, obs_size, action_dim, max):
        super().__init__()

        self.max = T.from_numpy(max).to(device)

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.norm = nn.BatchNorm1d(obs_size)  

        # # Softmax
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = T.from_numpy(obs).to(device)
        # obs = self.norm(obs.float())
        # action = T.tanh(action) * self.max # tanh to make action values between [-1, 1] and multiply by max value 1   
        return T.sigmoid(self.net(obs.float()))

class SAC(LightningModule):
    """ 
    """
    def __init__(self, env, capacity=100_000, batch_size=256, lr=1e-3, 
                hidden_size=256, gamma=0.99, loss_fn=F.smooth_l1_loss, optim=AdamW, 
                samples_per_epoch=1_000, tau=0.05, alpha=0.02, epsilon=0.05):
        super().__init__()  
        self.loop = asyncio.get_running_loop()
        self.counter = 0
        self.env = env

        # self.obs_size = self.env.observation_space.shape[1]
        # self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        self.obs_size = self.env.observation_space.shape[0]
        self.action_dims = self.env.action_space.shape[0]
        self.max_action = self.env.action_space.high

        self.q_net1 = DQN(hidden_size, self.obs_size, self.action_dims)
        self.q_net2 = DQN(hidden_size, self.obs_size, self.action_dims)
        self.policy = GradientPolicy(hidden_size, self.obs_size, self.action_dims, self.max_action)

        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)

        self.buffer = ReplayBuffer(capacity=capacity)

        self.save_hyperparameters('capacity', 'batch_size', 'lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 
            'tau', 'alpha', 'epsilon')

    @T.no_grad()
    def play_episodes(self, policy=None):
        obs, obs_nodes = self.env.getObs()
        done = False
        if policy and random.random() > self.hparams.epsilon:
            print('Get predicted action...')
            action = self.policy(obs)
            action = action.detach().cpu().numpy() #detach from GPU/.. to CPU and convert to numpy array
            # action = action.detach().numpy()
        else:
            # action = self.env.action_space.sample()
            logging.info('Get random action...')
            action = np.random.uniform(0, 1, size=(obs.shape[0],self.action_dims))
            # log_prob = np.random.uniform(0, 1, size=(1 ,self.action_dims))
            # action = np.random.normal(loc=0, scale=1, size=(1,self.action_dims))
            # action = np.random.standard_normal(size=(1,self.action_dims))
            #get zscore of action values  
        # print(f'action:\n{action}')               
        next_obs, reward, done, info = self.loop.run_until_complete(self.env.step(obs, obs_nodes, action))
        exp = (obs, action, reward, done, next_obs)
        self.counter +=1
        self.buffer.append(exp)
        pd.concat([pd.DataFrame(np.array(obs_nodes).reshape((-1, 1)), columns=['obs_nodes']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(action, columns=['action']),
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info']),
                pd.DataFrame(next_obs, columns=['nxt_obs'+str(i) for i in range(next_obs.shape[1])])], 
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
        obs = next_obs

    def forward(self, x):
        output = self.policy(x)
        return output

    def configure_optimizers(self):
        q_net_parameters = itertools.chain(self.q_net1.parameters(), self.q_net2.parameters())
        q_net_optimizer = self.hparams.optim(q_net_parameters, lr=self.hparams.lr)
        policy_optimizer = self.hparams.optim(self.policy.parameters(), lr=self.hparams.lr)
        return [q_net_optimizer, policy_optimizer]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):
        # inputs is batch of experiences exp -> {obs, actions, rewards, done, next_obs}
        # action_values are the result of trained DQN model when applying actions to states (obs)
        states, actions, rewards, done, next_states = batch
        # rewards = rewards.unsqueeze(2)
        # print(f'patch obs size: {states.shape}')
        # print(f'patch action size: {actions.shape}')
        # print(f'patch rewards size: {rewards.shape}')
        # print(f'patch done size: {done.shape}')
        # print(f'patch next_states size: {next_states.shape}')
        # print(rewards)
        states = states.reshape(-1, self.obs_size)
        actions = actions.reshape(-1, self.action_dims)
        rewards = rewards.reshape(-1, 1)
        next_states = next_states.reshape(-1, self.obs_size)
        # print(f'states: {states}')
        # print(f'actions: {actions}')
        # rewards = rewards.unsqueeze(1)
        # done = done.unsqueeze(1)

        if optimizer_idx == 0:
            #train Q-Networks:--------------------------------------------------------
            # (obs, actions)              ------> Q1              --> vals1
            # (obs, actions)              ------> Q1              --> vals2
            # (nxt_obs)                   ------> TPolicy         --> tactions, tprobs

            # (nxt_obs, tactions)         ------> TQ1             --> nxt_vals1
            # (nxt_obs, tactions)         ------> TQ2             --> nxt_vals2
            # min(nxt_vals1, nxt_vals2)                           --> nxt_vals

            # rewards + gamma * (nxt_vals - alpha * tprobs)       --> exp_vals
            # loss(vals1, exp_vals)                               --> q_loss1
            # loss(vals2, exp_vals)                               --> q_loss2
            # q_loss1 + q_loss2                                   --> q_loss
            #-------------------------------------------------------------------------

            action_values1 = self.q_net1(states, actions)
            action_values2 = self.q_net2(states, actions)

            target_actions = self.target_policy(next_states)
            # print(f'next_state size: {next_states.shape}')
            # print(f'target_action size: {target_actions.shape}')
            next_action_values = T.min(
                self.target_q_net1(next_states, target_actions),
                self.target_q_net2(next_states, target_actions)
            )
            # next_action_values[done] = 0.0 # as the network is dynamic the target is dynamic and the reward as well (this means that done is always false)

            expected_action_values = rewards + self.hparams.gamma * (next_action_values - self.hparams.alpha * target_actions)
            # print(f'rewards: {rewards}')
            # print(f'target_actions: {target_actions}')
            # print(f'next_action_values: {next_action_values}')
            # print(f'expected_action_values: {expected_action_values}')
            # print(f'action_values1: {action_values1}')
            # print(f'action_values2: {action_values2}')
            q_loss1 = self.hparams.loss_fn(action_values1.float(), expected_action_values.float())
            q_loss2 = self.hparams.loss_fn(action_values2.float(), expected_action_values.float())

            q_loss_total = q_loss1 + q_loss2
            self.log('episode/Q-Loss', q_loss_total)
            # self.log('episode/Q-Loss', q_loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log("episode/Q-performance", {"acc": acc, "recall": recall})
            return q_loss_total

        elif optimizer_idx == 1:
            #train the policy:--------------------------------------------------
            # (obs)                 ------> Policy         --> actions, probs
            # (obs, actions)        ------> Q1             --> vals1
            # (obs, actions)        ------> Q2             --> vals2
            # min(vals1, vals2)                            --> vals
            # alpha * probs - vals                         --> p_loss
            #------------------------------------------------------------------
            
            actions = self.policy(states)

            action_values = T.min(
                self.q_net1(states, actions),
                self.q_net2(states, actions)
            )

            policy_loss = (self.hparams.alpha * actions - action_values).mean()
            self.log('episode/Policy-Loss', policy_loss)
            # self.log('episode/Policy-Loss', policy_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log("episode/Policy-performance", {"acc": acc, "recall": recall})
            return policy_loss

    def training_epoch_end(self, training_step_outputs):
        self.play_episodes(policy=self.policy)
        #run async function as sync
        # self.loop.run_until_complete(self.play_episodes(policy=self.policy))

        polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
        polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)

        # self.log('episode_return', self.env.return_queue[-1])
        
    async def start(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/experiences.csv')
            quietRun('rm -r outputs/logs/lightning_logs/')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/content/lightning_logs/')
        except Exception as ex:
            print(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        print('START TRAINING ............................................')
        tb_logger = TensorBoardLogger(save_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            logger=tb_logger,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        
        # while len(self.buffer) < self.hparams.samples_per_epoch:
        #     await asyncio.gather(self.loop.run_in_executor(None, self.play_episodes))
        # await asyncio.gather(self.loop.run_in_executor(None, trainer.fit, self))

        while len(self.buffer) < self.hparams.samples_per_epoch:
            print(f'{len(self.buffer)} samples in experience buffer. Filling...')
            self.play_episodes()
        trainer.fit(self)
        
        await asyncio.sleep(0)

'''