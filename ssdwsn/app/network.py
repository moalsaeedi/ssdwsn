import nest_asyncio
nest_asyncio.apply()
import logging
from typing import Dict, Union, Tuple
from math import floor, sqrt, log, cos, sin, pi
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical, Normal
from os import cpu_count, path
from ssdwsn.util.utils import quietRun, CustomFormatter

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

class PPO_ValueNet_Pred(nn.Module):
    """Reinforcement Value Network to estimate the expected return
    Args:
        nn (_type_): Neural Network
    """
    def __init__(self, input_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim).to(device)
        self.ff = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.from_numpy(x).to(device)
        x = self.ff(self.norm(x.float()))
        return x
    
class PPO_Policy_Pred(nn.Module):
    """Gradient Policy Network (to predict an action following a normal distribution (infinit possible of actions) of an observation)
    Args:
        nn (_type_): neural network
    """
    def __init__(self, input_dim, hidden_size, action_dim, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(input_dim).to(device)
        # self.self_attn = SelfAttention(input_dim).to(device)
        self.self_attn = MultiHeadAttention(input_dim, input_dim).to(device)
        self.norm1 = nn.LayerNorm(input_dim).to(device)
        self.ff = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim)
        ).to(device)
        self.norm2 = nn.LayerNorm(input_dim).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

        self.mu_fc = nn.Linear(input_dim, action_dim).to(device)
        self.std_fc = nn.Linear(input_dim, action_dim).to(device)
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
        '''
        # '''
        # action prob distribution
        loc = self.mu_fc(x)
        loc = T.tanh(loc) * 1
        scale = self.std_fc(x)
        # scale = F.softplus(scale) + 1e-3
        scale = T.clip(scale, 1e-6, 1e-3)

        dist = Normal(loc, scale)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)        
        log_prob -= (2* (np.log(2) - action - F.softplus(-2*action))).sum(dim=-1, keepdim=True)
        # entropy = dist.entropy().sum(dim=-1, keepdim=True)
        # '''
        return log_prob, action
    
class PPO_ValueNet(nn.Module):
    """Reinforcement Value Network to estimate the expected return
    Args:
        nn (_type_): Neural Network
    """
    def __init__(self, input_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim).to(device)
        self.ff = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_size),
            NewGELU(),
            nn.Linear(hidden_size, hidden_size),
            NewGELU(),
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

        self.norm = nn.LayerNorm(input_dim).to(device)
        # self.self_attn = SelfAttention(input_dim).to(device)
        self.self_attn = MultiHeadAttention(input_dim, input_dim).to(device)
        self.norm1 = nn.LayerNorm(input_dim).to(device)
        self.ff = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_size),
            NewGELU(),
            nn.Linear(hidden_size, hidden_size),
            NewGELU(),
            nn.Linear(hidden_size, input_dim)
        ).to(device)
        self.norm2 = nn.LayerNorm(input_dim).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

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
        '''
        # '''
        # action prob distribution
        loc = self.fc(x)
        # loc = T.tanh(loc) * 1
        log_scale = -1e-3 * T.ones_like(loc, dtype=T.float)
        log_scale = T.nn.Parameter(log_scale)
        scale = T.exp(log_scale)

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
    
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + T.tanh(sqrt(2.0 / pi) * (x + 0.044715 * T.pow(x, 3.0))))
    
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