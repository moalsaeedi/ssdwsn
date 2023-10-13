import copy
import itertools
from typing import List, Tuple
from os import cpu_count, path
import random
import torch as T
import torch.nn.functional as F
import numpy as np
import pandas as pd
from ssdwsn.app.dataset import ReplayBufferSAC
from ssdwsn.app.network import ActorNetwork, CriticNetwork, ValueNetwork
import asyncio
import logging
from ssdwsn.util.utils import quietRun, CustomFormatter
from ssdwsn.app.utilts import plot_learning_curve
from ssdwsn.data.addr import Addr
from ssdwsn.openflow.action import ForwardUnicastAction, DropAction
from ssdwsn.openflow.entry import Entry
from ssdwsn.openflow.window import Window
from ssdwsn.util.constants import Constants as ct
import networkx as nx
import re
from collections import deque
# Pytorch libraries
import torch as T
import torch.nn.functional as F
from ssdwsn.util.utils import quietRun, CustomFormatter
from ssdwsn.app.lossFunction import lossBCE, lossCCE, lossMSE
from app.utilts import polyak_average
from ssdwsn.app.dataset import RLDataset_A2C, RLDataset_A2C_Shuffle, RLDataset_SAC, RLDataset_SAC_Shuffle, RLDataset_PPO, RLDataset_PPO_shuffle, RLDataset_GAE, RLDataset_GAE_shuffle, ExperienceSourceDataset
from ssdwsn.app.network import SAC_DQN, A2C_DQN, HE_DQN, CriticNetwork, A2C_GradientPolicy, ActorNetwork, ValueNetwork, SAC_GradientPolicy, HE_GradientPolicy, TD3_DQN, TD3_GredientPolicy, LSTM, Transformer, PPO_GradientPolicy, PPO_Policy, PPO_Att_Policy, PPO_Att_ValueNet, PPO_ValueNet, SelfAttention, A2C_ValueNet, REINFORCE_GradientPolicy
from torchmetrics import MeanSquaredError, R2Score
from torch.optim import Adam, AdamW, SGD, LBFGS
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet, SMAPE
from pytorch_forecasting.data.encoders import EncoderNormalizer,GroupNormalizer,MultiNormalizer,NaNLabelEncoder,TorchNormalizer
from lightning.pytorch.tuner import Tuner
from torch.utils.tensorboard import SummaryWriter
# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import time
from math import ceil, log, sqrt, atan2, pi

#logging----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
#-----------------------------------

seed_everything(33)
T.cuda.empty_cache()
num_gpus = T.cuda.device_count()
device = f'cuda:{num_gpus-1}' if T.cuda.is_available() else 'cpu'


class REINFORCE_Agent(LightningModule):
    """ 
    """
    def __init__(self, env, num_envs=50, batch_size=2, hidden_size=256, samples_per_epoch=2,
                 epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, gamma=0.97, epsilon=0.3, entropy_coef=0.1,
                 loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.current_episode = 0
        self.env = env

        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high
        self.obs_nds = None
        self.policy = REINFORCE_GradientPolicy(self.obs_dims, hidden_size, self.action_dims)

        self.buffer = []
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.save_hyperparameters('batch_size', 'policy_lr', 'value_lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')

    @T.no_grad()
    def play_episodes(self, policy=None):
        self.buffer = []
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.env.num_samples = 0
        for _ in range(self.hparams.samples_per_epoch):
            obs, nodes = self.env.reset()
            done = False
            done = np.zeros((1, 1))
            loc, scale = self.policy(obs)
            action = T.normal(loc, scale)
            action = action.detach().cpu().numpy()
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nodes))
            self.buffer.append((obs, action, reward, done, nxt_obs))
            self.env.num_samples += nxt_obs.shape[0]
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(action, columns=self.env.action_cols),
                pd.DataFrame(nxt_obs, columns=['nxt_obs'+str(i) for i in range(nxt_obs.shape[1])]), 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            self.obs = nxt_obs

    def configure_optimizers(self):
        policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)
        return policy_opt

    def train_dataloader(self):
        dataset = RLDataset_(self.buffer, self.env.num_samples, self.hparams.samples_per_epoch, self.hparams.gamma)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):
        obs_b, action_b, return_b, nxt_obs_b = batch
        
        rev_nxt_obs_b = self.env.max_obs * nxt_obs_b.detach().cpu().numpy()
        self.log('episode/Return', return_b.sum())
        self.log('episode/Performance/Delay', rev_nxt_obs_b[:,1].mean())
        self.log('episode/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
        self.log('episode/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
        self.log('episode/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
        self.log('episode/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,17].mean())

        loc, scale = self.policy(obs_b)
        dist = Normal(loc, scale)
        log_prob_b = dist.log_prob(action_b).sum(dim=-1, keepdim=True)

        policy_loss = - log_prob_b * return_b

        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        self.ep_policy_loss.append(policy_loss.mean().unsqueeze(0))
        self.ep_entropy.append(entropy.mean().unsqueeze(0))
        self.log('episode/Policy/Loss', policy_loss.mean())
        self.log('episode/Policy/Entropy', entropy.mean())
        return T.mean(policy_loss - self.hparams.entropy_coef * entropy)

    def training_epoch_end(self, training_step_outputs):     

        reshape_fn = lambda x: x.reshape(self.env.num_samples, -1)
        dataset = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, dataset)

        rev_nxt_obs_b = self.env.max_obs *nxt_obs_b
        self.log('epoch/ValueNet/Loss', T.cat(self.ep_value_loss).mean()) 
        self.log('epoch/Policy/Loss', T.cat(self.ep_policy_loss).mean())
        self.log('epoch/Policy/Entropy', T.cat(self.ep_entropy).mean())
        self.log('epoch/Return', reward_b.sum())
        self.log('epoch/Performance/Delay', rev_nxt_obs_b[:,1].mean())
        self.log('epoch/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
        self.log('epoch/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
        self.log('epoch/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
        self.log('epoch/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,17].mean())
        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/experiences.csv')
            quietRun('rm -r outputs/logs/lightning_logs/')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        tb_logger = TensorBoardLogger(save_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            logger=tb_logger,
            reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        
        self.play_episodes()
        trainer.fit(self) 

class TD3_Agent(LightningModule):

    def __init__(self, env, capacity=500, batch_size=8192, actor_lr=3e-4, critic_lr=3e-4, hidden_size=64, gamma=0.97, loss_fn=F.smooth_l1_loss,
                optim=AdamW, eps_start=1.0, eps_end=0.15, eps_last_episode=500, samples_per_epoch=10, epoch_repeat=4, tau=0.005):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.epoch_cntr = 1
        self.env = env
        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]        

        self.q_net1 = TD3_DQN(hidden_size, self.obs_dims, self.action_dims)
        self.q_net2 = TD3_DQN(hidden_size, self.obs_dims, self.action_dims)
        self.policy = TD3_GredientPolicy(hidden_size, self.obs_dims, self.action_dims, self.env.min_action_space, self.env.max_action_space)

        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)

        self.buffer = []

        self.save_hyperparameters('capacity', 'batch_size', 'actor_lr', 'critic_lr', 'hidden_size', 'gamma', 'loss_fn',
                'optim', 'eps_start', 'eps_end', 'eps_last_episode', 'samples_per_epoch', 'epoch_repeat', 'tau')

    @T.no_grad()
    def play_episodes(self, policy=None, epsilon=0.0):

        self.buffer = []
        ep_returns = 0
        self.env.num_samples = 0
        for _ in range(self.hparams.samples_per_epoch):
            obs, nodes = self.env.reset()
            done = False
            done = np.zeros((1, 1)) 
            if policy:
                logger.info('Get predicted action...')
                action = self.policy(obs, epsilon=epsilon)                
                action = action.detach().cpu().numpy()
            else:
                # action = self.env.action_space.sample()
                logger.info('Get random action...')
                action = np.random.uniform(-1, 1, size=(obs.shape[0],self.action_dims))
                # action = np.random.uniform(0, 1, size=(obs.shape[0],self.action_dims))
                # action = np.random.normal(loc=0, scale=1, size=(1,self.action_dims))
                # action = np.random.standard_normal(size=(1,self.action_dims))
                #get zscore of action values                 
            nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nodes))
            # next_obs, reward, info = self.loop.run_until_complete(self.env.step(action))
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            exp = (obs, action, reward, done, nxt_obs)
            # exp = (obs, action, reward, next_obs)
            self.buffer.append(exp)
            self.env.num_samples += nxt_obs.shape[0]
            ep_returns += reward.sum()
            # pd.concat([pd.DataFrame(nds, columns=['node']),
            #     pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
            #     pd.DataFrame(action, columns=self.env.action_cols),
            #     pd.DataFrame(nxt_obs, columns=['nxt_obs'+str(i) for i in range(next_obs.shape[1])]), 
            #     pd.DataFrame(reward, columns=['reward']),
            #     pd.DataFrame(done, columns=['done']),
            #     pd.DataFrame(info, columns=['info'])],
            #     axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            self.obs = nxt_obs
        
        self.epoch_cntr += 1
        return ep_returns/self.hparams.samples_per_epoch

    def forward(self, x):
        output = self.policy.mu(x)
        return output
        
    def configure_optimizers(self):
        q_net_parameters = itertools.chain(self.q_net1.parameters(), self.q_net2.parameters())
        q_net_optimizer = self.hparams.optim(q_net_parameters, lr=self.hparams.critic_lr)
        policy_optimizer = self.hparams.optim(self.policy.parameters(), lr=self.hparams.actor_lr)
        return [q_net_optimizer, policy_optimizer]

    def train_dataloader(self):
        dataset = dataset = RLDataset(self.buffer, self.env.num_samples, self.hparams.epoch_repeat)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), # No. of CPU cores
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):

        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.current_epoch / self.hparams.eps_last_episode
        )

        self.play_episodes(self.policy, epsilon=epsilon)

        obs, actions, rewards, dones, next_obs = batch

        batch_dictionary = {}
        # batch_dictionary['delay'] = next_obs[:,:,1].mean()
        # batch_dictionary['goodput'] = next_obs[:,:,2].mean()
        # batch_dictionary['engcons'] = next_obs[:,:,3].mean()
        batch_dictionary['reward'] = rewards.sum(0).flatten()


        # self.log('episode/Q-Loss', q_loss_total)
        # print(f'action:\n {action}')
        # print(f'next_obs:\n {next_obs}')
        # print(f'reward:\n {reward}')
        # self.log('episode/Reward', reward.sum(0).flatten())self.log('episode/Reward', T.tensor(np.mean(list(self.env.reward_history)), device=device))
        self.log('episode/Performance/Delay', T.tensor(np.mean(list(self.env.delay_history)), device=device))
        self.log('episode/Performance/Throughput', T.tensor(np.mean(list(self.env.throughput_history)), device=device))
        self.log('episode/Performance/Enrgy_Consumption', T.tensor(np.mean(list(self.env.energycons_history)), device=device))
        self.log('episode/Performance/TxPackets', T.tensor(np.mean(list(self.env.txpackets_history)), device=device))
        self.log('episode/Performance/TxBytes', T.tensor(np.mean(list(self.env.txbytes_history)), device=device))
        self.log('episode/Performance/RxPackets', T.tensor(np.mean(list(self.env.rxpackets_history)), device=device))
        self.log('episode/Performance/RxBytes', T.tensor(np.mean(list(self.env.rxbytes_history)), device=device))
        self.log('episode/Performance/TxPacketsIn', T.tensor(np.mean(list(self.env.txpacketsin_history)), device=device))
        self.log('episode/Performance/TxBytesIn', T.tensor(np.mean(list(self.env.txbytesin_history)), device=device))
        self.log('episode/Performance/RxPacketsOut', T.tensor(np.mean(list(self.env.rxpacketsout_history)), device=device))
        self.log('episode/Performance/RxBytesOut', T.tensor(np.mean(list(self.env.rxbytesout_history)), device=device))
        self.log('episode/Performance/Dropped_Packets', T.tensor(np.mean(list(self.env.drpackets_history)), device=device))
        self.log('episode/Performance/Resedual_Energy_var', T.tensor(np.mean(list(self.env.energyvar_history)), device=device))

        if optimizer_idx == 0:
            action_values1 = self.q_net1(obs, actions)
            action_values2 = self.q_net2(obs, actions)
            next_actions = self.target_policy(next_obs, epsilon=epsilon, noise_clip=0.05)

            next_action_values = T.min(
                self.target_q_net1(next_obs, next_actions),
                self.target_q_net2(next_obs, next_actions)
            )
            next_action_values[dones] = 0.0

            expected_action_values = rewards + self.hparams.gamma * next_action_values
            q_loss1 = self.hparams.loss_fn(action_values1.float(), expected_action_values.float())
            q_loss2 = self.hparams.loss_fn(action_values2.float(), expected_action_values.float())
            total_loss = q_loss1 + q_loss2

            batch_dictionary['opt'] = optimizer_idx
            batch_dictionary['loss'] = total_loss
            self.log("episode/Q-Loss", total_loss)
            return total_loss

        if optimizer_idx == 1 and batch_idx % 2 == 0:
            mu = self.policy.mu(obs)
            policy_loss = -self.q_net1(obs, mu).mean()
            batch_dictionary['opt'] = optimizer_idx
            batch_dictionary['loss'] = policy_loss
            self.log("episode/Policy Loss", policy_loss)
            return policy_loss

    def training_epoch_end(self, outputs):
        # '''
        polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
        polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)
        # '''
        # self.target_q_net1.load_state_dict(self.q_net1.state_dict()) 
        # self.target_q_net2.load_state_dict(self.q_net2.state_dict()) 
        # self.target_policy.load_state_dict(self.policy.state_dict()) 

        avg_ep_return = self.play_episodes(self.policy)

        # if self.current_epoch % 2 == 0:
        self.log('episode/Avg Return', avg_ep_return)

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/experiences.csv')
            quietRun('rm -r outputs/logs/lightning_logs/')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
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
        
        self.play_episodes()
        trainer.fit(self)
        
        await asyncio.sleep(0)  

class SAC_Agent(LightningModule):
    """ 
    """
    def __init__(self, env, batch_size=256, lr=1e-3, hidden_size=256, gamma=0.99, loss_fn=F.mse_loss, optim=AdamW,
                epoch_repeat=4, samples_per_epoch=1_000, tau=0.05, alpha=0.02, epsilon=0.05):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.epoch_cntr = 1
        self.env = env

        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high
        self.obs_nds = None
        self.q_net1 = SAC_DQN(self.obs_dims, hidden_size, self.action_dims)
        self.q_net2 = SAC_DQN(self.obs_dims, hidden_size, self.action_dims)
        self.policy = SAC_GradientPolicy(self.obs_dims, hidden_size, self.action_dims)

        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)
        self.target_policy = copy.deepcopy(self.policy)

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.save_hyperparameters('batch_size', 'lr', 'hidden_size', 'gamma', 
            'loss_fn', 'optim', 'samples_per_epoch', 'epoch_repeat', 'tau', 'alpha', 'epsilon')

    @T.no_grad()
    def play_episodes(self, policy=None):
        # self.buffer = []
        # self.env.num_samples = 0
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        obs, nodes = self.env.reset()
        for _ in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            if policy and random.random() > self.hparams.epsilon:
                logger.info('Get predicted action...')
                loc, scale = self.policy(obs)
                action = T.normal(loc, scale)
                action = action.detach().cpu().numpy()
                print(f'resampled action:\n{action}')
                # action = action.detach().numpy()
            else:
                # action = self.env.action_space.sample()
                logger.info('Get random action...')
                action = np.random.uniform(-1, 1, size=(obs.shape[0],self.action_dims))
                # action = np.random.uniform(0, 1, size=(obs.shape[0],self.action_dims))
                # action = np.random.normal(loc=0, scale=1, size=(1,self.action_dims))
                # action = np.random.standard_normal(size=(1,self.action_dims))
                #get zscore of action values       
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nodes))
            self.buffer.append((obs, action, reward, done, nxt_obs))
            self.num_samples.append(nxt_obs.shape[0])
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(action, columns=self.env.action_cols),
                pd.DataFrame(nxt_obs, columns=['nxt_obs'+str(i) for i in range(nxt_obs.shape[1])]), 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            obs = nxt_obs
            
    def configure_optimizers(self):
        q_net_parameters = itertools.chain(self.q_net1.parameters(), self.q_net2.parameters())
        q_net_optimizer = self.hparams.optim(q_net_parameters, lr=self.hparams.lr)
        policy_optimizer = self.hparams.optim(self.policy.parameters(), lr=self.hparams.lr)
        return q_net_optimizer, policy_optimizer

    def train_dataloader(self):
        dataset = RLDataset_SAC(self.buffer, sum(self.num_samples))
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):
        # inputs is batch of experiences exp -> {obs, action, reward, done, next_obs}
        # action_values are the result of trained DQN model when applying action to state (obs)
        obs, action, reward, done, next_obs = batch
       
        # rev_nxt_obs_b = self.env.max_obs * next_obs.detach().cpu().numpy()
        rev_nxt_obs_b = next_obs.detach().cpu().numpy()
        # rev_nxt_obs_b = self.env.scaler.inverse_transform(next_obs.detach().cpu().numpy())
        self.log('episode/Return', reward.sum())
        self.log('episode/Performance/Delay', rev_nxt_obs_b[:,1].mean())
        self.log('episode/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
        self.log('episode/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
        self.log('episode/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
        self.log('episode/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,18].mean())


        # self.log('episode/Reward', T.tensor(np.mean(list(self.env.reward_history)), device=device))
        # self.log('episode/Performance/Delay', T.tensor(np.mean(list(self.env.delay_history)), device=device))
        # self.log('episode/Performance/Throughput', T.tensor(np.mean(list(self.env.throughput_history)), device=device))
        # self.log('episode/Performance/Enrgy_Consumption', T.tensor(np.mean(list(self.env.energycons_history)), device=device))
        # self.log('episode/Performance/TxPackets', T.tensor(np.mean(list(self.env.txpackets_history)), device=device))
        # self.log('episode/Performance/TxBytes', T.tensor(np.mean(list(self.env.txbytes_history)), device=device))
        # self.log('episode/Performance/RxPackets', T.tensor(np.mean(list(self.env.rxpackets_history)), device=device))
        # self.log('episode/Performance/RxBytes', T.tensor(np.mean(list(self.env.rxbytes_history)), device=device))
        # self.log('episode/Performance/TxPacketsIn', T.tensor(np.mean(list(self.env.txpacketsin_history)), device=device))
        # self.log('episode/Performance/TxBytesIn', T.tensor(np.mean(list(self.env.txbytesin_history)), device=device))
        # self.log('episode/Performance/RxPacketsOut', T.tensor(np.mean(list(self.env.rxpacketsout_history)), device=device))
        # self.log('episode/Performance/RxBytesOut', T.tensor(np.mean(list(self.env.rxbytesout_history)), device=device))
        # self.log('episode/Performance/Dropped_Packets', T.tensor(np.mean(list(self.env.drpackets_history)), device=device))
        # self.log('episode/Performance/Resedual_Energy_var', T.tensor(np.mean(list(self.env.energyvar_history)), device=device))

        if optimizer_idx == 0:
            #train Q-Networks:--------------------------------------------------------
            # (obs, action)              ------> Q1              --> vals1
            # (obs, action)              ------> Q2              --> vals2
            # (nxt_obs)                   ------> TPolicy         --> taction, tprobs

            # (nxt_obs, taction)         ------> TQ1             --> nxt_vals1
            # (nxt_obs, taction)         ------> TQ2             --> nxt_vals2
            # min(nxt_vals1, nxt_vals2)                           --> nxt_vals

            # rewards + gamma * (nxt_vals - alpha * tprobs)       --> exp_vals
            # loss(vals1, exp_vals)                               --> q_loss1
            # loss(vals2, exp_vals)                               --> q_loss2
            # q_loss1 + q_loss2                                   --> q_loss
            #-------------------------------------------------------------------------

            action_values1 = self.q_net1(obs, action)
            action_values2 = self.q_net2(obs, action)

            # with T.no_grad():
            target_loc, target_scale = self.target_policy(next_obs)
            dist = Normal(target_loc, target_scale)
            target_action = dist.rsample()
            target_log_probs = dist.log_prob(target_action).sum(dim=-1, keepdim=True)
            target_log_probs -= (2* (np.log(2) - target_action - F.softplus(-2*target_action))).sum(dim=-1, keepdim=True)

            next_action_values = T.min(
                self.target_q_net1(next_obs, target_action),
                self.target_q_net2(next_obs, target_action)
            )
            next_action_values[done] = 0.0

            expected_action_values = reward + self.hparams.gamma * (next_action_values - self.hparams.alpha * target_log_probs)

            q_loss1 = self.hparams.loss_fn(action_values1, expected_action_values.float())
            q_loss2 = self.hparams.loss_fn(action_values2, expected_action_values.float())

            q_loss_total = q_loss1 + q_loss2

            self.ep_value_loss.append(q_loss_total.unsqueeze(0))
            # self.log('episode/ValueNet/Loss', q_loss_total)

            return q_loss_total

        elif optimizer_idx == 1:
            #train the policy:--------------------------------------------------
            # (obs)                 ------> Policy         --> action, probs
            # (obs, action)        ------> Q1             --> vals1
            # (obs, action)        ------> Q2             --> vals2
            # min(vals1, vals2)                            --> vals
            # alpha * probs - vals                         --> p_loss
            #------------------------------------------------------------------
            
            loc, scale = self.policy(obs)
            dist = Normal(loc, scale)
            action = dist.rsample()
            log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
            log_probs -= (2* (np.log(2) - action - F.softplus(-2*action))).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)

            action_values = T.min(
                self.q_net1(obs, action),
                self.q_net2(obs, action)
            )
            policy_loss = (self.hparams.alpha * log_probs - action_values).mean()
            self.ep_policy_loss.append(policy_loss.unsqueeze(0))
            self.ep_entropy.append(entropy.mean().unsqueeze(0))
            # self.log('episode/Policy/Loss', policy_loss)
            # self.log('episode/Policy/Entropy', entropy.mean())
            return policy_loss

    def training_epoch_end(self, training_step_outputs):
        polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
        polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)
        
        # self.target_q_net1.load_state_dict(self.q_net1.state_dict())        
        # self.target_q_net2.load_state_dict(self.q_net2.state_dict())        
        # self.target_policy.load_state_dict(self.policy.state_dict())        

        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        dataset = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, dataset)

        # rev_nxt_obs_b = self.env.max_obs *nxt_obs_b
        rev_nxt_obs_b = nxt_obs_b
        # rev_nxt_obs_b = self.env.scaler.inverse_transform(nxt_obs_b)
        self.log('epoch/ValueNet/Loss', T.cat(self.ep_value_loss).mean()) 
        self.log('epoch/Policy/Loss', T.cat(self.ep_policy_loss).mean())
        self.log('epoch/Policy/Entropy', T.cat(self.ep_entropy).mean())
        self.log('epoch/Return', reward_b.sum())
        self.log('epoch/Performance/Delay', rev_nxt_obs_b[:,1].mean())
        self.log('epoch/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
        self.log('epoch/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
        self.log('epoch/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
        self.log('epoch/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,18].mean())
        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()       

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/experiences.csv')
            quietRun('rm -r outputs/logs/lightning_logs/')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        tb_logger = TensorBoardLogger(save_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            logger=tb_logger,
            reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        
        # while len(self.buffer) < self.hparams.samples_per_epoch:
        #     await asyncio.gather(self.loop.run_in_executor(None, self.play_episodes))
        # await asyncio.gather(self.loop.run_in_executor(None, trainer.fit, self))
        
        self.play_episodes()
        trainer.fit(self) 

class A2C_Agent(LightningModule):
    """ 
    """
    def __init__(self, env, batch_size=256, samples_per_epoch=1_000, hidden_size=256, policy_lr=1e-4, value_lr=1e-3, 
                gamma=0.99, entropy_coef=0.01, loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()  
        self.loop = asyncio.get_running_loop()
        self.current_episode = 0
        self.env = env

        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high

        self.policy = A2C_GradientPolicy(self.obs_dims, hidden_size, self.action_dims)
        self.value_net = A2C_ValueNet(self.obs_dims, hidden_size)

        self.target_value_net = copy.deepcopy(self.value_net)

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.save_hyperparameters('batch_size', 'policy_lr', 'value_lr',
            'hidden_size', 'gamma', 'entropy_coef', 'loss_fn', 'optim', 'samples_per_epoch')

    @T.no_grad()
    def play_episodes(self, policy=None):
        # self.buffer = []
        # self.env.num_samples = 0
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        obs, nodes = self.env.reset()
        ep_ret = 0
        ep_returns = []
        for _ in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            loc, scale = self.policy(obs)
            action = T.normal(loc, scale)
            action = action.detach().cpu().numpy()
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nodes))
            ep_ret += reward.sum()
            ep_returns.append(ep_ret)
            self.buffer.append((obs, action, reward, done, nxt_obs))
            self.num_samples.append(nxt_obs.shape[0])
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(action, columns=self.env.action_cols),
                pd.DataFrame(nxt_obs, columns=['nxt_obs'+str(i) for i in range(nxt_obs.shape[1])]), 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            obs = nxt_obs
        return sum(ep_returns)/self.hparams.samples_per_epoch
        
    def configure_optimizers(self):
        value_optimizer = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        policy_optimizer = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)
        return value_optimizer, policy_optimizer

    def train_dataloader(self):
        dataset = RLDataset_A2C(self.buffer, sum(self.num_samples))
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):
        obs, action, reward, done, next_obs = batch

        # rev_nxt_obs_b = self.env.max_obs * next_obs.detach().cpu().numpy()
        rev_nxt_obs_b = next_obs.detach().cpu().numpy()
        # rev_nxt_obs_b = self.env.scaler.inverse_transform(next_obs.detach().cpu().numpy())
        self.log('episode/Return', reward.sum())
        self.log('episode/Performance/Delay', rev_nxt_obs_b[:,1].mean())
        self.log('episode/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
        self.log('episode/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
        self.log('episode/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
        self.log('episode/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,18].mean())

        state_values = self.value_net(obs)
        
        with T.no_grad():
            next_state_values = self.target_value_net(next_obs)
            next_state_values[done] = 0.0
            target = reward + self.hparams.gamma * next_state_values

        # self.log('episode/Reward', T.tensor(np.mean(list(self.env.reward_history)), device=device))
        # self.log('episode/Performance/Delay', T.tensor(np.mean(list(self.env.delay_history)), device=device))
        # self.log('episode/Performance/Throughput', T.tensor(np.mean(list(self.env.throughput_history)), device=device))
        # self.log('episode/Performance/Enrgy_Consumption', T.tensor(np.mean(list(self.env.energycons_history)), device=device))
        # self.log('episode/Performance/TxPackets', T.tensor(np.mean(list(self.env.txpackets_history)), device=device))
        # self.log('episode/Performance/TxBytes', T.tensor(np.mean(list(self.env.txbytes_history)), device=device))
        # self.log('episode/Performance/RxPackets', T.tensor(np.mean(list(self.env.rxpackets_history)), device=device))
        # self.log('episode/Performance/RxBytes', T.tensor(np.mean(list(self.env.rxbytes_history)), device=device))
        # self.log('episode/Performance/TxPacketsIn', T.tensor(np.mean(list(self.env.txpacketsin_history)), device=device))
        # self.log('episode/Performance/TxBytesIn', T.tensor(np.mean(list(self.env.txbytesin_history)), device=device))
        # self.log('episode/Performance/RxPacketsOut', T.tensor(np.mean(list(self.env.rxpacketsout_history)), device=device))
        # self.log('episode/Performance/RxBytesOut', T.tensor(np.mean(list(self.env.rxbytesout_history)), device=device))
        # self.log('episode/Performance/Dropped_Packets', T.tensor(np.mean(list(self.env.drpackets_history)), device=device))
        # self.log('episode/Performance/Resedual_Energy_var', T.tensor(np.mean(list(self.env.energyvar_history)), device=device))

        self.current_episode += 1

        if optimizer_idx == 0:
            loss = self.hparams.loss_fn(state_values.float(), target.float())
            self.ep_value_loss.append(loss.unsqueeze(0))
            # self.log('episode/ValueNet/Loss', loss)
            return loss

        elif optimizer_idx == 1:
            advantages = (target - state_values).detach()
            loc, scale = self.policy(obs)
            dist = Normal(loc, scale)
            log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)

            policy_loss = - log_probs * advantages
            loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()

            self.ep_policy_loss.append(loss.unsqueeze(0))
            self.ep_entropy.append(entropy.mean().unsqueeze(0))
            # self.log('episode/Policy/Loss', loss)
            # self.log('episode/Policy/Entropy', entropy.mean())
            return loss

    def training_epoch_end(self, training_step_outputs):
        self.target_value_net.load_state_dict(self.value_net.state_dict())        

        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        dataset = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, dataset)

        # rev_nxt_obs_b = self.env.max_obs *nxt_obs_b
        rev_nxt_obs_b = nxt_obs_b
        # rev_nxt_obs_b = self.env.scaler.inverse_transform(nxt_obs_b)
        self.log('epoch/ValueNet/Loss', T.cat(self.ep_value_loss).mean()) 
        self.log('epoch/Policy/Loss', T.cat(self.ep_policy_loss).mean())
        self.log('epoch/Policy/Entropy', T.cat(self.ep_entropy).mean())
        self.log('epoch/Return', reward_b.sum())
        self.log('epoch/Performance/Delay', rev_nxt_obs_b[:,1].mean())
        self.log('epoch/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
        self.log('epoch/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
        self.log('epoch/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
        self.log('epoch/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,18].mean())
        print(f'END EPOCH: {self.current_epoch}*****************')
        ep_returns = self.play_episodes()
        self.log('epoch/Return_', ep_returns)

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/experiences.csv')
            quietRun('rm -r outputs/logs/lightning_logs/')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        tb_logger = TensorBoardLogger(save_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            logger=tb_logger,
            reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        
        self.play_episodes()
        trainer.fit(self)  

class PPO_Agent(LightningModule):
    """ 
    """
    def __init__(self, env, num_envs=50, batch_size=2, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, gamma=0.94, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.env = env

        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high
        self.obs_nds = None
        # self.policy = PPO_Policy(self.obs_dims, hidden_size, self.action_dims)
        # self.target_policy = copy.deepcopy(self.policy)

        self.policy_ch = PPO_Policy(self.obs_dims, hidden_size, 1)
        self.target_policy_ch = copy.deepcopy(self.policy_ch)
        self.policy_nh = PPO_Policy(self.obs_dims, hidden_size, 1)
        self.target_policy_nh = copy.deepcopy(self.policy_nh)
        self.policy_rt = PPO_Policy(self.obs_dims+2, hidden_size, 1)
        self.target_policy_rt = copy.deepcopy(self.policy_rt)

        self.value_net = PPO_ValueNet(self.obs_dims+self.action_dims, hidden_size)
        self.target_val_net = copy.deepcopy(self.value_net) 

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.ep_step = 0

        self.save_hyperparameters('batch_size', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')

    @T.no_grad()
    def play_episodes(self, policy=None):
        # self.buffer = []
        # self.env.num_samples = 0
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        delay = []
        throughput = []
        engcons = []
        droppackets = []
        resenergy = []
        returns = [0]
        obs, nodes = self.env.reset()        
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            log_prob_ch, action_ch = self.policy_ch(obs)
            action_ch = action_ch.detach().cpu().numpy()
            log_prob_ch = log_prob_ch.detach().cpu().numpy()
            log_prob_nh, action_nh = self.policy_nh(obs)
            action_nh = action_nh.detach().cpu().numpy()
            log_prob_nh = log_prob_nh.detach().cpu().numpy()
            log_prob_rt, action_rt = self.policy_rt(np.hstack((obs, action_ch, action_nh)))
            action_rt = action_rt.detach().cpu().numpy()
            log_prob_rt = log_prob_rt.detach().cpu().numpy()

            log_prob = np.hstack((log_prob_ch,log_prob_nh,log_prob_rt))
            action = np.hstack((action_ch,action_nh,action_rt))

            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nodes))
            self.buffer.append((obs, log_prob, action, reward, done, nxt_obs))
            self.num_samples.append(nxt_obs.shape[0])
            returns.append(reward.sum())
            rev_nxt_obs = self.env.scaler.inverse_transform(nxt_obs)
            delay.append(rev_nxt_obs[:,1].mean())
            throughput.append(rev_nxt_obs[:,2].mean())
            engcons.append(rev_nxt_obs[:,3].mean())
            droppackets.append(rev_nxt_obs[:,10].mean())
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(action, columns=self.env.action_cols),
                pd.DataFrame(nxt_obs, columns=['nxt_obs'+str(i) for i in range(nxt_obs.shape[1])]), 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            
            obs = nxt_obs

            self.tb_logger.add_scalars('Steps/Return', {
                'R': reward.sum()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Steps/Delay', {
                'NE': rev_nxt_obs[:,1].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Steps/Throughput', {
                'NT': rev_nxt_obs[:,2].mean() 
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Steps/Energy_Consumption', {
                'NGC': rev_nxt_obs[:,3].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Steps/Dropped_Packets', {
                'NDR': rev_nxt_obs[:,10].mean()
                }, global_step=self.ep_step
            )

            self.ep_step += 1

    def _dataset(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(sum(self.num_samples)):
            yield obs_b[i], log_prob_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def _dataset_shuffle(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.hparams.epoch_repeat):
            idx = list(range(sum(self.num_samples)))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], log_prob_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def configure_optimizers(self):
        value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        policy_opt_ch = self.hparams.optim(self.policy_ch.parameters(), lr=self.hparams.policy_lr)
        policy_opt_nh = self.hparams.optim(self.policy_nh.parameters(), lr=self.hparams.policy_lr)
        policy_opt_rt = self.hparams.optim(self.policy_rt.parameters(), lr=self.hparams.policy_lr)
     
        return value_opt, policy_opt_ch, policy_opt_nh, policy_opt_rt

    def optimizer_step(self, *args, **kwargs):
        """
        Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic
        for each data sample.
        """
        for i in range(self.hparams.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceSourceDataset(self._dataset)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):

        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = batch        

        # rev_nxt_obs = self.env.scaler.inverse_transform(T.clone(nxt_obs_b.data).detach().cpu().numpy())
        
        state_values = self.value_net(T.hstack((obs_b, action_b)))
       
        with T.no_grad():
          
            _, nxt_action_ch = self.target_policy_ch(nxt_obs_b)
            _, nxt_action_nh = self.target_policy_nh(nxt_obs_b)
            _, nxt_action_rt = self.target_policy_rt(T.hstack((nxt_obs_b,nxt_action_ch,nxt_action_nh)))
            nxt_action = T.hstack((nxt_action_ch,nxt_action_nh,nxt_action_rt))
            nxt_state_values = self.target_val_net(T.hstack((nxt_obs_b, nxt_action)))
            nxt_state_values[done_b] = 0.0
            target = reward_b + self.hparams.gamma * nxt_state_values
        
        advantages = (target - state_values).detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if optimizer_idx == 0:
            loss = self.hparams.loss_fn(state_values.float(), target.float())
            self.ep_value_loss.append(loss.unsqueeze(0))
            # self.log('value_net/loss', loss)  
            self.tb_logger.add_scalars('Steps/Value/Loss', {
                'loss': loss
                }, global_step=self.global_step
            )         
            return loss
        
        elif optimizer_idx == 1:
            log_prob_ch, _ = self.policy_ch(obs_b)
            prv_log_prob_ch = log_prob_b[:,0].reshape(-1,1)
            
            rho = T.exp(log_prob_ch - prv_log_prob_ch)
            # rho = log_prob / prv_log_prob

            surrogate_1 = rho * advantages
            surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            policy_loss = - T.minimum(surrogate_1, surrogate_2)
            entropy = -T.sum(action_b[:,0].reshape(-1,1)*log_prob_b[:,0].reshape(-1,1), dim=-1, keepdim=True)
            loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()

            self.ep_policy_loss.append(policy_loss.mean().unsqueeze(0))
            self.tb_logger.add_scalars('Steps/Policy/Loss', {
                'CH': policy_loss.mean(), 
                }, global_step=self.global_step
            )
            self.ep_entropy.append(entropy.mean().unsqueeze(0))
            self.tb_logger.add_scalars('Steps/Policy/Entropy', {
                'CH': entropy.mean()
                }, global_step=self.global_step
            )
            return loss
        
        elif optimizer_idx == 2:
            log_prob_nh, _ = self.policy_nh(obs_b)
            prv_log_prob_nh = log_prob_b[:,1].reshape(-1,1)
            
            rho = T.exp(log_prob_nh - prv_log_prob_nh)
            # rho = log_prob / prv_log_prob

            surrogate_1 = rho * advantages
            surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            policy_loss = - T.minimum(surrogate_1, surrogate_2)
            entropy = -T.sum(action_b[:,1].reshape(-1,1)*log_prob_b[:,1].reshape(-1,1), dim=-1, keepdim=True)
            loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()

            self.ep_policy_loss.append(policy_loss.mean().unsqueeze(0))
            self.tb_logger.add_scalars('Steps/Policy/Loss', {
                'NH': policy_loss.mean(), 
                }, global_step=self.global_step
            )
            self.ep_entropy.append(entropy.mean().unsqueeze(0))
            self.tb_logger.add_scalars('Steps/Policy/Entropy', {
                'NH': entropy.mean()
                }, global_step=self.global_step
            )
            return loss

        elif optimizer_idx == 3:
            log_prob_rt, _ = self.policy_rt(T.hstack((obs_b, action_b[:,0].reshape(-1,1), action_b[:,1].reshape(-1,1))))
            prv_log_prob_rt = log_prob_b[:,2].reshape(-1,1)
            
            rho = T.exp(log_prob_rt - prv_log_prob_rt)
            # rho = log_prob / prv_log_prob

            surrogate_1 = rho * advantages
            surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            policy_loss = - T.minimum(surrogate_1, surrogate_2)
            entropy = -T.sum(action_b[:,2].reshape(-1,1)*log_prob_b[:,2].reshape(-1,1), dim=-1, keepdim=True)
            loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()

            self.ep_policy_loss.append(policy_loss.mean().unsqueeze(0))
            self.tb_logger.add_scalars('Steps/Policy/Loss', {
                'RT': policy_loss.mean(), 
                }, global_step=self.global_step
            )
            self.ep_entropy.append(entropy.mean().unsqueeze(0))
            self.tb_logger.add_scalars('Steps/Policy/Entropy', {
                'RT': entropy.mean()
                }, global_step=self.global_step
            )
            return loss
        
    def training_epoch_end(self, training_step_outputs):
        self.target_val_net.load_state_dict(self.value_net.state_dict())   
        self.target_policy_ch.load_state_dict(self.policy_ch.state_dict())      
        self.target_policy_nh.load_state_dict(self.policy_nh.state_dict())      
        self.target_policy_rt.load_state_dict(self.policy_rt.state_dict())      

        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/')
            # quietRun('rm -r outputs/logs/experiences.csv')
            # quietRun('rm -r outputs/logs/lightning_logs/version_0')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            # logger=self.tb_logger,
            # reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        
        self.play_episodes()
        trainer.fit(self) 


class PPO_Agent2(LightningModule):
    """ 
    """
    def __init__(self, env, num_envs=50, batch_size=2, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, gamma=0.94, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.env = env

        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high
        self.obs_nds = None
        # self.policy = PPO_Policy(self.obs_dims, hidden_size, self.action_dims)
        # self.target_policy = copy.deepcopy(self.policy)

        self.policy_ch = PPO_Policy(self.obs_dims, hidden_size, 1)
        self.target_policy_ch = copy.deepcopy(self.policy_ch)
        self.policy_nh = PPO_Policy(self.obs_dims, hidden_size, 1)
        self.target_policy_nh = copy.deepcopy(self.policy_nh)
        self.policy_rt = PPO_Policy(self.obs_dims, hidden_size, 1)
        self.target_policy_rt = copy.deepcopy(self.policy_rt)

        self.value_net = PPO_ValueNet(self.obs_dims+1, hidden_size)
        self.target_val_net = copy.deepcopy(self.value_net) 

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []

        self.save_hyperparameters('batch_size', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')

    @T.no_grad()
    def play_episodes(self, policy=None):
        # self.buffer = []
        # self.env.num_samples = 0
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        delay = []
        throughput = []
        engcons = []
        droppackets = []
        resenergy = []
        returns = [0]
        obs, nodes = self.env.reset()        
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            # log_prob, action = self.policy(obs)
            # action = action.detach().cpu().numpy()
            # log_prob = log_prob.detach().cpu().numpy()

            log_prob_ch, action_ch = self.policy_ch(obs)
            action_ch = action_ch.detach().cpu().numpy()
            log_prob_ch = log_prob_ch.detach().cpu().numpy()
            log_prob_nh, action_nh = self.policy_nh(obs)
            action_nh = action_nh.detach().cpu().numpy()
            log_prob_nh = log_prob_nh.detach().cpu().numpy()
            log_prob_rt, action_rt = self.policy_rt(obs)
            action_rt = action_rt.detach().cpu().numpy()
            log_prob_rt = log_prob_rt.detach().cpu().numpy()

            log_prob = np.hstack((log_prob_ch,log_prob_nh,log_prob_rt))
            action = np.hstack((action_ch,action_nh,action_rt))

            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nodes))
            self.buffer.append((obs, log_prob, action, reward, done, nxt_obs))
            self.num_samples.append(nxt_obs.shape[0])
            # rev_nxt_obs_b = self.env.max_obs * nxt_obs_b.detach().cpu().numpy()
            # rev_nxt_obs_b = nxt_obs_b.detach().cpu().numpy() * self.env.max_obs
            # rev_nxt_obs = self.env.scaler.inverse_transform(nxt_obs.detach().cpu().numpy())
            # rev_nxt_obs_b = nxt_obs_b.detach() * np.sqrt(self.env.obs_rms.var + self.env.epsilon) + self.env.obs_rms.mean        
            
            returns.append(reward.sum())
            rev_nxt_obs = self.env.scaler.inverse_transform(nxt_obs)
            delay.append(rev_nxt_obs[:,1].mean())
            throughput.append(rev_nxt_obs[:,2].mean())
            engcons.append(rev_nxt_obs[:,3].mean())
            droppackets.append(rev_nxt_obs[:,10].mean())
            resenergy.append(rev_nxt_obs[:,-1].mean())
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(action, columns=self.env.action_cols),
                pd.DataFrame(nxt_obs, columns=['nxt_obs'+str(i) for i in range(nxt_obs.shape[1])]), 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            
            obs = nxt_obs

        #     self.tb_logger.add_scalars('Episode/Return', {
        #         'R': reward.sum()
        #         }, global_step=self.global_step
        #     )
        #     self.tb_logger.add_scalars('Episode/Delay', {
        #         'NE': rev_nxt_obs[:,1].mean()
        #         }, global_step=self.global_step
        #     )
        #     self.tb_logger.add_scalars('Episode/Throughput', {
        #         'NT': rev_nxt_obs[:,2].mean() 
        #         }, global_step=self.global_step
        #     )
        #     self.tb_logger.add_scalars('Episode/Energy_Consumption', {
        #         'NGC': rev_nxt_obs[:,3].mean()
        #         }, global_step=self.global_step
        #     )
        #     self.tb_logger.add_scalars('Episode/Dropped_Packets', {
        #         'NDR': rev_nxt_obs[:,10].mean()
        #         }, global_step=self.global_step
        #     )
        #     self.tb_logger.add_scalars('Episode/Resedual_Energy_Variance', {
        #         'NGVar': rev_nxt_obs[:,-1].mean()
        #         }, global_step=self.global_step
        #     )

        # self.tb_logger.add_scalars('Epoch/Return', {
        #     'R': sum(returns)/len(returns)
        #     }, global_step=self.global_step
        # )
        # self.tb_logger.add_scalars('Epoch/Delay', {
        #     'NE': sum(delay)/len(delay)
        #     }, global_step=self.global_step
        # )
        # self.tb_logger.add_scalars('Epoch/Throughput', {
        #     'NT': sum(throughput)/len(throughput) 
        #     }, global_step=self.global_step
        # )
        # self.tb_logger.add_scalars('Epoch/Energy_Consumption', {
        #     'NGC': sum(engcons)/len(engcons)
        #     }, global_step=self.global_step
        # )
        # self.tb_logger.add_scalars('Epoch/Dropped_Packets', {
        #     'NDR': sum(droppackets)/len(droppackets)
        #     }, global_step=self.global_step
        # )
        # self.tb_logger.add_scalars('Epoch/Resedual_Energy_Variance', {
        #     'NGVar': sum(resenergy)/len(resenergy)
        #     }, global_step=self.global_step
        # )

    def _dataset(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(sum(self.num_samples)):
            yield obs_b[i], log_prob_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def _dataset_shuffle(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.hparams.epoch_repeat):
            idx = list(range(sum(self.num_samples)))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], log_prob_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def configure_optimizers(self):
        value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        # policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)
        policy_opt_ch = self.hparams.optim(self.policy_ch.parameters(), lr=self.hparams.policy_lr)
        policy_opt_nh = self.hparams.optim(self.policy_nh.parameters(), lr=self.hparams.policy_lr)
        policy_opt_rt = self.hparams.optim(self.policy_rt.parameters(), lr=self.hparams.policy_lr)
        # return value_opt, policy_opt
        # return value_opt, policy_opt
        return value_opt, policy_opt_ch, policy_opt_nh, policy_opt_rt

    # def optimizer_step(self, *args, **kwargs):
    #     """
    #     Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic
    #     for each data sample.
    #     """
    #     for i in range(self.hparams.nb_optim_iters):
    #         super().optimizer_step(*args, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceSourceDataset(self._dataset)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader
    
    # def _dataloader(self) -> DataLoader:
    #     """Initialize the Replay Buffer dataset used for retrieving experiences"""
    #     dataset = RLDataset_PPO_shuffle(self.buffer, sum(self.num_samples), self.hparams.epoch_repeat)
    #     dataloader = DataLoader(dataset=dataset, 
    #         batch_size=self.hparams.batch_size,
    #         num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
    #     )
    #     return dataloader

    # def train_dataloader(self):
    #     dataset = RLDataset_PPO_shuffle(self.buffer, sum(self.num_samples), self.hparams.epoch_repeat)
    #     dataloader = DataLoader(
    #         dataset=dataset,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
    #     )
    #     return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):

        # self.log("epoch/Return", sum(self.ep_returns), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Delay', sum(self.delay)/len(self.delay), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Throughput', sum(self.throughput)/len(self.throughput), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Enrgy_Consumption', sum(self.engcons)/len(self.engcons), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Dropped_Packets', sum(self.droppackets)/len(self.droppackets), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Resedual_Energy_var', sum(self.resenergy)/len(self.resenergy), on_step=False, on_epoch=True)
            
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = batch        

        rev_nxt_obs = self.env.scaler.inverse_transform(T.clone(nxt_obs_b.data).detach().cpu().numpy())
        # self.log("episode/Return", reward_b.sum(), enable_graph=True)
        # self.log('episode/Performance/Delay', rev_nxt_obs[:,1].mean(), enable_graph=True)
        # self.log('episode/Performance/Throughput', rev_nxt_obs[:,2].mean(), enable_graph=True)
        # self.log('episode/Performance/Enrgy_Consumption', rev_nxt_obs[:,3].mean(), enable_graph=True)
        # self.log('episode/Performance/Dropped_Packets', rev_nxt_obs[:,10].mean(), enable_graph=True)
        # self.log('episode/Performance/Resedual_Energy_var', rev_nxt_obs[:,-1].mean(), enable_graph=True)
        
        # if self.global_step == 0 or self.global_step == 1:            
        #     self.tb_logger._experiment.add_scalars('Steps/Return', {
        #         'R': 0.0
        #         }, global_step=self.global_step
        #     )
        # else:
        # '''
        self.tb_logger._experiment.add_scalars('Steps/Return', {
            'R': reward_b.sum()
            }, global_step=self.global_step
        )
        self.tb_logger._experiment.add_scalars('Steps/Delay', {
            'NE': rev_nxt_obs[:,1].mean()
            }, global_step=self.global_step
        )
        self.tb_logger._experiment.add_scalars('Steps/Throughput', {
            'NT': rev_nxt_obs[:,2].mean() 
            }, global_step=self.global_step
        )
        self.tb_logger._experiment.add_scalars('Steps/Energy_Consumption', {
            'NGC': rev_nxt_obs[:,3].mean()
            }, global_step=self.global_step
        )
        self.tb_logger._experiment.add_scalars('Steps/Dropped_Packets', {
            'NDR': rev_nxt_obs[:,10].mean()
            }, global_step=self.global_step
        )
        self.tb_logger._experiment.add_scalars('Steps/Resedual_Energy_Variance', {
            'NGVar': rev_nxt_obs[:,-1].mean()
            }, global_step=self.global_step
        )
        # '''
        state_values_ch = self.value_net(T.hstack((obs_b, action_b[:,0].reshape(-1,1))))
        state_values_nh = self.value_net(T.hstack((obs_b, action_b[:,1].reshape(-1,1))))
        state_values_rt = self.value_net(T.hstack((obs_b, action_b[:,2].reshape(-1,1))))
       
        with T.no_grad():
            # nxt_new_loc, nxt_new_scale = self.target_policy(nxt_obs_b)
            # nxt_action = T.normal(nxt_new_loc, nxt_new_scale)
            # nxt_state_values = self.target_val_net(T.hstack((nxt_obs_b,nxt_action)))
            # '''
            # _, nxt_action = self.target_policy(nxt_obs_b)
            _, nxt_action_ch = self.target_policy_ch(nxt_obs_b)
            _, nxt_action_nh = self.target_policy_nh(nxt_obs_b)
            _, nxt_action_rt = self.target_policy_rt(nxt_obs_b)
            nxt_state_values_ch = self.target_val_net(T.hstack((nxt_obs_b, nxt_action_ch)))
            nxt_state_values_nh = self.target_val_net(T.hstack((nxt_obs_b, nxt_action_nh)))
            nxt_state_values_rt = self.target_val_net(T.hstack((nxt_obs_b, nxt_action_rt)))
            # '''
            nxt_state_values_ch[done_b] = 0.0
            nxt_state_values_nh[done_b] = 0.0
            nxt_state_values_rt[done_b] = 0.0
            target_ch = reward_b + self.hparams.gamma * nxt_state_values_ch
            target_nh = reward_b + self.hparams.gamma * nxt_state_values_nh
            target_rt = reward_b + self.hparams.gamma * nxt_state_values_rt
        
        advantages_ch = (target_ch - state_values_ch).detach()
        advantages_nh = (target_nh - state_values_nh).detach()
        advantages_rt = (target_rt - state_values_rt).detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if optimizer_idx == 0:
            loss_ch = self.hparams.loss_fn(state_values_ch.float(), target_ch.float())
            loss_nh = self.hparams.loss_fn(state_values_nh.float(), target_nh.float())
            loss_rt = self.hparams.loss_fn(state_values_rt.float(), target_rt.float())
            loss = loss_ch+loss_nh+loss_rt
            self.ep_value_loss.append(loss.unsqueeze(0))
            # self.log('value_net/loss', loss)  
            self.tb_logger._experiment.add_scalars('Steps/Value/Loss', {
                'loss': loss
                }, global_step=self.global_step
            )         
            return loss

        # elif optimizer_idx == 1:
        #     log_prob, _ = self.policy(obs_b)
        #     prv_log_prob = log_prob_b
            
        #     rho = T.exp(log_prob - prv_log_prob)
        #     # rho = log_prob / prv_log_prob

        #     surrogate_1 = rho * advantages
        #     surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

        #     policy_loss = - T.minimum(surrogate_1, surrogate_2)
        #     entropy = -T.sum(action_b*log_prob_b, dim=-1, keepdim=True)
        #     loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()

        #     self.ep_policy_loss.append(policy_loss.mean().unsqueeze(0))
        #     self.tb_logger._experiment.add_scalars('Steps/Policy/Loss', {
        #         'CH': policy_loss.mean(), 
        #         }, global_step=self.global_step
        #     )
        #     self.ep_entropy.append(policy_loss.mean().unsqueeze(0))
        #     self.tb_logger._experiment.add_scalars('Steps/Policy/Entropy', {
        #         'CH': entropy.mean()
        #         }, global_step=self.global_step
        #     )
        #     return loss
        
        elif optimizer_idx == 1:
            log_prob_ch, _ = self.policy_ch(obs_b)
            prv_log_prob_ch = log_prob_b[:,0].reshape(-1,1)
            
            rho = T.exp(log_prob_ch - prv_log_prob_ch)
            # rho = log_prob / prv_log_prob

            surrogate_1 = rho * advantages_ch
            surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages_ch

            policy_loss = - T.minimum(surrogate_1, surrogate_2)
            entropy = -T.sum(action_b[:,0].reshape(-1,1)*log_prob_b[:,0].reshape(-1,1), dim=-1, keepdim=True)
            loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()

            self.ep_policy_loss.append(policy_loss.mean().unsqueeze(0))
            self.tb_logger._experiment.add_scalars('Steps/Policy/Loss', {
                'CH': policy_loss.mean(), 
                }, global_step=self.global_step
            )
            self.ep_entropy.append(entropy.mean().unsqueeze(0))
            self.tb_logger._experiment.add_scalars('Steps/Policy/Entropy', {
                'CH': entropy.mean()
                }, global_step=self.global_step
            )
            return loss
        
        elif optimizer_idx == 2:
            log_prob_nh, _ = self.policy_nh(obs_b)
            prv_log_prob_nh = log_prob_b[:,1].reshape(-1,1)
            
            rho = T.exp(log_prob_nh - prv_log_prob_nh)
            # rho = log_prob / prv_log_prob

            surrogate_1 = rho * advantages_nh
            surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages_nh

            policy_loss = - T.minimum(surrogate_1, surrogate_2)
            entropy = -T.sum(action_b[:,1].reshape(-1,1)*log_prob_b[:,1].reshape(-1,1), dim=-1, keepdim=True)
            loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()

            self.ep_policy_loss.append(policy_loss.mean().unsqueeze(0))
            self.tb_logger._experiment.add_scalars('Steps/Policy/Loss', {
                'NH': policy_loss.mean(), 
                }, global_step=self.global_step
            )
            self.ep_entropy.append(entropy.mean().unsqueeze(0))
            self.tb_logger._experiment.add_scalars('Steps/Policy/Entropy', {
                'NH': entropy.mean()
                }, global_step=self.global_step
            )
            return loss

        elif optimizer_idx == 3:
            log_prob_rt, _ = self.policy_rt(obs_b)
            prv_log_prob_rt = log_prob_b[:,2].reshape(-1,1)
            
            rho = T.exp(log_prob_rt - prv_log_prob_rt)
            # rho = log_prob / prv_log_prob

            surrogate_1 = rho * advantages_rt
            surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages_rt

            policy_loss = - T.minimum(surrogate_1, surrogate_2)
            entropy = -T.sum(action_b[:,2].reshape(-1,1)*log_prob_b[:,2].reshape(-1,1), dim=-1, keepdim=True)
            loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()

            self.ep_policy_loss.append(policy_loss.mean().unsqueeze(0))
            self.tb_logger._experiment.add_scalars('Steps/Policy/Loss', {
                'RT': policy_loss.mean(), 
                }, global_step=self.global_step
            )
            self.ep_entropy.append(entropy.mean().unsqueeze(0))
            self.tb_logger._experiment.add_scalars('Steps/Policy/Entropy', {
                'RT': entropy.mean()
                }, global_step=self.global_step
            )
            return loss
        
    def training_epoch_end(self, training_step_outputs):
        self.target_val_net.load_state_dict(self.value_net.state_dict())   
        # self.target_policy.load_state_dict(self.policy.state_dict())      
        self.target_policy_ch.load_state_dict(self.policy_ch.state_dict())      
        self.target_policy_nh.load_state_dict(self.policy_nh.state_dict())      
        self.target_policy_rt.load_state_dict(self.policy_rt.state_dict())      

        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        dataset = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, dataset)

        # rev_nxt_obs_b = self.env.max_obs *nxt_obs_b
        # rev_nxt_obs_b = nxt_obs_b * self.env.max_obs
        rev_nxt_obs_b = self.env.scaler.inverse_transform(nxt_obs_b)
        # rev_nxt_obs_b = nxt_obs_b.detach() * np.sqrt(self.env.obs_rms.var + self.env.epsilon) + self.env.obs_rms.mean
        # self.log('epoch/ValueNet/Loss', T.cat(self.ep_value_loss).mean()) 
        # self.log('epoch/Policy/Loss', T.cat(self.ep_policy_loss).mean())
        # self.log('epoch/Policy/Entropy', T.cat(self.ep_entropy).mean())

        # self.log('epoch/Return', reward_b.sum())
        # self.log('epoch/Performance/Delay', rev_nxt_obs_b[:,1].mean())
        # self.log('epoch/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
        # self.log('epoch/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
        # self.log('epoch/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
        # self.log('epoch/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,-1].mean())
        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/lightning_logs/')
            quietRun('rm -r outputs/logs/experiences.csv')
            # quietRun('rm -r outputs/logs/lightning_logs/version_0')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        self.tb_logger = TensorBoardLogger(save_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            logger=self.tb_logger,
            # reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        
        self.play_episodes()
        trainer.fit(self) 

class PPO_Agent1(LightningModule):
    """ 
    """
    def __init__(self, env, num_envs=50, batch_size=2, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, gamma=0.94, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.env = env

        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high
        self.obs_nds = None
        self.policy = PPO_Policy(self.obs_dims, hidden_size, self.action_dims)
        self.target_policy = copy.deepcopy(self.policy)
        self.value_net = PPO_ValueNet(self.obs_dims+self.action_dims, hidden_size)
        self.target_val_net = copy.deepcopy(self.value_net) 

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []

        self.save_hyperparameters('batch_size', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')

    @T.no_grad()
    def play_episodes(self, policy=None):
        # self.buffer = []
        # self.env.num_samples = 0
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        delay = []
        throughput = []
        engcons = []
        droppackets = []
        resenergy = []
        returns = [0]
        obs, nodes = self.env.reset()        
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            log_prob, action = self.policy(obs)
            print(f'actual log prob: {log_prob}')
            print(f'actual action: {action}')
            action = action.detach().cpu().numpy()
            log_prob = log_prob.detach().cpu().numpy()

            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nodes))
            self.buffer.append((obs, log_prob, action, reward, done, nxt_obs))
            self.num_samples.append(nxt_obs.shape[0])
            # rev_nxt_obs_b = self.env.max_obs * nxt_obs_b.detach().cpu().numpy()
            # rev_nxt_obs_b = nxt_obs_b.detach().cpu().numpy() * self.env.max_obs
            # rev_nxt_obs = self.env.scaler.inverse_transform(nxt_obs.detach().cpu().numpy())
            # rev_nxt_obs_b = nxt_obs_b.detach() * np.sqrt(self.env.obs_rms.var + self.env.epsilon) + self.env.obs_rms.mean        
            
            returns.append(reward.sum())
            # rev_nxt_obs = self.env.scaler.inverse_transform(nxt_obs)
            rev_nxt_obs = nxt_obs
            delay.append(rev_nxt_obs[:,1].mean())
            throughput.append(rev_nxt_obs[:,2].mean())
            engcons.append(rev_nxt_obs[:,3].mean())
            droppackets.append(rev_nxt_obs[:,10].mean())
            resenergy.append(rev_nxt_obs[:,-1].mean())
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(action, columns=self.env.action_cols),
                pd.DataFrame(nxt_obs, columns=['nxt_obs'+str(i) for i in range(nxt_obs.shape[1])]), 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            
            obs = nxt_obs
            # scalerdict = { k:v for (k,v) in zip(nodes.flatten().tolist(), rev_nxt_obs[:,-3].flatten().tolist())} 
            # self.tb_logger.add_scalars('Episode/Rptti', scalerdict, global_step=self.global_step
            # )
            # self.tb_logger.add_scalars('Episode/Rptti_var', {
            #     'Rptti_var': rev_nxt_obs[:,-2].mean()
            #     }, global_step=self.global_step
            # )
            
            # self.tb_logger.add_scalars('Episode/Return', {
            #     'R': reward.sum()
            #     }, global_step=self.global_step
            # )
            # self.tb_logger.add_scalars('Episode/Delay', {
            #     'NE': rev_nxt_obs[:,1].mean()
            #     }, global_step=self.global_step
            # )
            # self.tb_logger.add_scalars('Episode/Throughput', {
            #     'NT': rev_nxt_obs[:,2].mean() 
            #     }, global_step=self.global_step
            # )
            # self.tb_logger.add_scalars('Episode/Energy_Consumption', {
            #     'NGC': rev_nxt_obs[:,3].mean()
            #     }, global_step=self.global_step
            # )
            # self.tb_logger.add_scalars('Episode/Dropped_Packets', {
            #     'NDR': rev_nxt_obs[:,10].mean()
            #     }, global_step=self.global_step
            # )
            # self.tb_logger.add_scalars('Episode/Resedual_Energy_Variance', {
            #     'NGVar': rev_nxt_obs[:,-1].mean()
            #     }, global_step=self.global_step
            # )

        # self.tb_logger.add_scalars('Epoch/Return', {
        #     'R': sum(returns)/len(returns)
        #     }, global_step=self.global_step
        # )
        # self.tb_logger.add_scalars('Epoch/Delay', {
        #     'NE': sum(delay)/len(delay)
        #     }, global_step=self.global_step
        # )
        # self.tb_logger.add_scalars('Epoch/Throughput', {
        #     'NT': sum(throughput)/len(throughput) 
        #     }, global_step=self.global_step
        # )
        # self.tb_logger.add_scalars('Epoch/Energy_Consumption', {
        #     'NGC': sum(engcons)/len(engcons)
        #     }, global_step=self.global_step
        # )
        # self.tb_logger.add_scalars('Epoch/Dropped_Packets', {
        #     'NDR': sum(droppackets)/len(droppackets)
        #     }, global_step=self.global_step
        # )
        # self.tb_logger.add_scalars('Epoch/Resedual_Energy_Variance', {
        #     'NGVar': sum(resenergy)/len(resenergy)
        #     }, global_step=self.global_step
        # )

    def _dataset(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(sum(self.num_samples)):
            yield obs_b[i], log_prob_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def _dataset_shuffle(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.hparams.epoch_repeat):
            idx = list(range(sum(self.num_samples)))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], log_prob_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def configure_optimizers(self):
        value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)   
        return value_opt, policy_opt

    # def optimizer_step(self, *args, **kwargs):
    #     """
    #     Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic
    #     for each data sample.
    #     """
    #     for i in range(self.hparams.nb_optim_iters):
    #         super().optimizer_step(*args, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceSourceDataset(self._dataset)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader
    
    # def _dataloader(self) -> DataLoader:
    #     """Initialize the Replay Buffer dataset used for retrieving experiences"""
    #     dataset = RLDataset_PPO_shuffle(self.buffer, sum(self.num_samples), self.hparams.epoch_repeat)
    #     dataloader = DataLoader(dataset=dataset, 
    #         batch_size=self.hparams.batch_size,
    #         num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
    #     )
    #     return dataloader

    # def train_dataloader(self):
    #     dataset = RLDataset_PPO_shuffle(self.buffer, sum(self.num_samples), self.hparams.epoch_repeat)
    #     dataloader = DataLoader(
    #         dataset=dataset,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
    #     )
    #     return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):

        # self.log("epoch/Return", sum(self.ep_returns), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Delay', sum(self.delay)/len(self.delay), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Throughput', sum(self.throughput)/len(self.throughput), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Enrgy_Consumption', sum(self.engcons)/len(self.engcons), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Dropped_Packets', sum(self.droppackets)/len(self.droppackets), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Resedual_Energy_var', sum(self.resenergy)/len(self.resenergy), on_step=False, on_epoch=True)
            
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = batch        

        # rev_nxt_obs = self.env.scaler.inverse_transform(T.clone(nxt_obs_b.data).detach().cpu().numpy())
        rev_nxt_obs = nxt_obs_b
        # self.log("episode/Return", reward_b.sum(), enable_graph=True)
        # self.log('episode/Performance/Delay', rev_nxt_obs[:,1].mean(), enable_graph=True)
        # self.log('episode/Performance/Throughput', rev_nxt_obs[:,2].mean(), enable_graph=True)
        # self.log('episode/Performance/Enrgy_Consumption', rev_nxt_obs[:,3].mean(), enable_graph=True)
        # self.log('episode/Performance/Dropped_Packets', rev_nxt_obs[:,10].mean(), enable_graph=True)
        # self.log('episode/Performance/Resedual_Energy_var', rev_nxt_obs[:,-1].mean(), enable_graph=True)
        
        # if self.global_step == 0 or self.global_step == 1:            
        #     self.tb_logger._experiment.add_scalars('Steps/Return', {
        #         'R': 0.0
        #         }, global_step=self.global_step
        #     )
        # else:
        self.log('Steps/Return', reward_b.sum())
        self.log('Steps/Delay', rev_nxt_obs[:,1].mean())
        self.log('Steps/Throughput', rev_nxt_obs[:,2].mean() )
        self.log('Steps/Energy_Consumption', rev_nxt_obs[:,3].mean())
        self.log('Steps/Dropped_Packets', rev_nxt_obs[:,10].mean())
        self.log('Steps/Report_TTI_Variance', rev_nxt_obs[:,-2].mean())
        self.log('Steps/Resedual_Energy_Variance', rev_nxt_obs[:,-1].mean())
        '''
        self.tb_logger.add_scalars('Steps/Return', {
            'R': reward_b.sum()
            }, global_step=self.global_step
        )
        self.tb_logger.add_scalars('Steps/Delay', {
            'NE': rev_nxt_obs[:,1].mean()
            }, global_step=self.global_step
        )
        self.tb_logger.add_scalars('Steps/Throughput', {
            'NT': rev_nxt_obs[:,2].mean() 
            }, global_step=self.global_step
        )
        self.tb_logger.add_scalars('Steps/Energy_Consumption', {
            'NGC': rev_nxt_obs[:,3].mean()
            }, global_step=self.global_step
        )
        self.tb_logger.add_scalars('Steps/Dropped_Packets', {
            'NDR': rev_nxt_obs[:,10].mean()
            }, global_step=self.global_step
        )
        self.tb_logger.add_scalars('Steps/Report_TTI_Variance', {
            'RTTIVar': rev_nxt_obs[:,-2].mean()
            }, global_step=self.global_step
        )
        self.tb_logger.add_scalars('Steps/Resedual_Energy_Variance', {
            'NGVar': rev_nxt_obs[:,-1].mean()
            }, global_step=self.global_step
        )
        '''
        state_values = self.value_net(T.hstack((obs_b, action_b)))
       
        with T.no_grad():
            # nxt_new_loc, nxt_new_scale = self.target_policy(nxt_obs_b)
            # nxt_action = T.normal(nxt_new_loc, nxt_new_scale)
            # nxt_state_values = self.target_val_net(T.hstack((nxt_obs_b,nxt_action)))
            # '''
            _, nxt_action = self.target_policy(nxt_obs_b)
            nxt_state_values = self.target_val_net(T.hstack((nxt_obs_b, nxt_action)))
            # '''
            nxt_state_values[done_b] = 0.0 
            target = reward_b + self.hparams.gamma * nxt_state_values
        
        advantages = (target - state_values).detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if optimizer_idx == 0:
            loss = self.hparams.loss_fn(state_values.float(), target.float())
            self.ep_value_loss.append(loss.unsqueeze(0))
            self.log('value_net/loss', loss)  
            # self.tb_logger.add_scalars('Steps/Value/Loss', {
            #     'loss': loss
            #     }, global_step=self.global_step
            # )         
            return loss

        elif optimizer_idx == 1:
            log_prob, _ = self.policy(obs_b)
            prv_log_prob = log_prob_b
            
            rho = T.exp(log_prob - prv_log_prob)
            # rho = log_prob / prv_log_prob

            surrogate_1 = rho * advantages
            surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            policy_loss = - T.minimum(surrogate_1, surrogate_2)
            entropy = -T.sum(action_b*log_prob_b, dim=-1, keepdim=True)
            loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()

            self.log('Steps/Policy/Loss', policy_loss.mean())
            self.log('Steps/Policy/Entropy', entropy.mean())
            self.ep_policy_loss.append(policy_loss.mean().unsqueeze(0))
            # self.tb_logger.add_scalars('Steps/Policy/Loss', {
            #     'loss': policy_loss.mean() 
            #     }, global_step=self.global_step
            # )
            self.ep_entropy.append(entropy.mean().unsqueeze(0))
            # self.tb_logger.add_scalars('Steps/Policy/Entropy', {
            #     'entropy': entropy.mean()
            #     }, global_step=self.global_step
            # )
            return loss
        
        
    def training_epoch_end(self, training_step_outputs):
        self.target_val_net.load_state_dict(self.value_net.state_dict())            
        self.target_policy.load_state_dict(self.policy.state_dict())            

        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        dataset = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, dataset)

        # rev_nxt_obs_b = self.env.max_obs *nxt_obs_b
        # rev_nxt_obs_b = nxt_obs_b * self.env.max_obs
        # rev_nxt_obs_b = self.env.scaler.inverse_transform(nxt_obs_b)
        rev_nxt_obs_b = nxt_obs_b
        # rev_nxt_obs_b = nxt_obs_b.detach() * np.sqrt(self.env.obs_rms.var + self.env.epsilon) + self.env.obs_rms.mean
        # self.log('epoch/ValueNet/Loss', T.cat(self.ep_value_loss).mean()) 
        # self.log('epoch/Policy/Loss', T.cat(self.ep_policy_loss).mean())
        # self.log('epoch/Policy/Entropy', T.cat(self.ep_entropy).mean())

        # self.log('epoch/Return', reward_b.sum())
        # self.log('epoch/Performance/Delay', rev_nxt_obs_b[:,1].mean())
        # self.log('epoch/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
        # self.log('epoch/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
        # self.log('epoch/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
        # self.log('epoch/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,-1].mean())
        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/lightning_logs/')
            # quietRun('rm -r outputs/logs/')
            quietRun('rm -r outputs/logs/experiences.csv')
            # quietRun('rm -r outputs/logs/lightning_logs/version_0')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        self.tb_logger = TensorBoardLogger(save_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            logger=self.tb_logger,
            # reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        T.autograd.detect_anomaly(True)
        self.play_episodes()
        trainer.fit(self) 

class PPO_Agent2(LightningModule):
    """ 
    """
    def __init__(self, env, num_envs=50, batch_size=2, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, gamma=0.94, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.env = env

        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high
        self.obs_nds = None
        # self.policy = PPO_GradientPolicy(self.obs_dims, hidden_size, self.action_dims)
        # self.target_policy = copy.deepcopy(self.policy)
        # '''
        self.policy_ch = PPO_GradientPolicy(self.obs_dims, hidden_size, 1)
        self.policy_nh = PPO_GradientPolicy(self.obs_dims, hidden_size, 1)
        self.policy_rt = PPO_GradientPolicy(self.obs_dims, hidden_size, 1)
        self.target_policy_ch = copy.deepcopy(self.policy_ch)
        self.target_policy_nh = copy.deepcopy(self.policy_nh)
        self.target_policy_rt = copy.deepcopy(self.policy_rt)
        # '''
        self.value_net = PPO_ValueNet(self.obs_dims+self.action_dims, hidden_size)
        self.target_val_net = copy.deepcopy(self.value_net) 


        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        # self.ep_policy_loss = []
        # self.ep_entropy = []
        # '''
        self.ep_policy_loss_ch = []
        self.ep_policy_loss_nh = []
        self.ep_policy_loss_rt = []
        self.ep_entropy_ch = []
        self.ep_entropy_nh = []
        self.ep_entropy_rt = []
        # '''
        self.ep_returns = []
        self.delay = []
        self.throughput = []
        self.engcons = []
        self.droppackets = []
        self.resenergy = []

        self.save_hyperparameters('batch_size', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')

    @T.no_grad()
    def play_episodes(self, policy=None):
        # self.buffer = []
        # self.env.num_samples = 0
        self.ep_value_loss = []
        # self.ep_policy_loss = []
        # self.ep_entropy = []
        # '''
        self.ep_policy_loss_ch = []
        self.ep_policy_loss_nh = []
        self.ep_policy_loss_rt = []
        self.ep_entropy_ch = []
        self.ep_entropy_nh = []
        self.ep_entropy_rt = []
        # '''
        self.ep_returns = []
        self.delay = []
        self.throughput = []
        self.engcons = []
        self.droppackets = []
        self.resenergy = []
        returns = [0]
        obs, nodes = self.env.reset()        
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            # loc, scale = self.policy(obs)
            # action = T.normal(loc, scale)
            # action = action.detach().cpu().numpy()
            # loc = loc.detach().cpu().numpy()
            # scale = scale.detach().cpu().numpy()
            # '''
            loc_ch, scale_ch = self.policy_ch(obs)
            action_ch = T.normal(loc_ch, scale_ch)
            action_ch = action_ch.detach().cpu().numpy()
            loc_ch = loc_ch.detach().cpu().numpy()
            scale_ch = scale_ch.detach().cpu().numpy()

            loc_nh, scale_nh = self.policy_nh(obs)
            action_nh = T.normal(loc_nh, scale_nh)
            action_nh = action_nh.detach().cpu().numpy()
            loc_nh = loc_nh.detach().cpu().numpy()
            scale_nh = scale_nh.detach().cpu().numpy()

            loc_rt, scale_rt = self.policy_rt(obs)
            action_rt = T.normal(loc_rt, scale_rt)
            action_rt = action_rt.detach().cpu().numpy()
            loc_rt = loc_rt.detach().cpu().numpy()
            scale_rt = scale_rt.detach().cpu().numpy()

            loc = np.hstack((loc_ch,loc_nh,loc_rt))
            scale = np.hstack((scale_ch,scale_nh,scale_rt))
            action = np.hstack((action_ch,action_nh,action_rt))
            # '''
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nodes))
            self.buffer.append((obs, loc, scale, action, reward, done, nxt_obs))
            self.num_samples.append(nxt_obs.shape[0])
            self.ep_returns.append(returns[step])
            returns.append(reward.sum())
            # rev_nxt_obs_b = self.env.max_obs * nxt_obs_b.detach().cpu().numpy()
            # rev_nxt_obs_b = nxt_obs_b.detach().cpu().numpy() * self.env.max_obs
            # rev_nxt_obs = self.env.scaler.inverse_transform(nxt_obs.detach().cpu().numpy())
            # rev_nxt_obs_b = nxt_obs_b.detach() * np.sqrt(self.env.obs_rms.var + self.env.epsilon) + self.env.obs_rms.mean        
            
            rev_nxt_obs = self.env.scaler.inverse_transform(nxt_obs)
            self.delay.append(rev_nxt_obs[:,1].mean())
            self.throughput.append(rev_nxt_obs[:,2].mean())
            self.engcons.append(rev_nxt_obs[:,3].mean())
            self.droppackets.append(rev_nxt_obs[:,10].mean())
            self.resenergy.append(rev_nxt_obs[:,-1].mean())
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(action, columns=self.env.action_cols),
                pd.DataFrame(nxt_obs, columns=['nxt_obs'+str(i) for i in range(nxt_obs.shape[1])]), 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            
            obs = nxt_obs

    def _dataset(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, loc_b, scale_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(sum(self.num_samples)):
            yield obs_b[i], loc_b[i], scale_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def _dataset_shuffle(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, loc_b, scale_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.hparams.epoch_repeat):
            idx = list(range(sum(self.num_samples)))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], loc_b[i], scale_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def configure_optimizers(self):
        value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        # policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)
        policy_ch_opt = self.hparams.optim(self.policy_ch.parameters(), lr=self.hparams.policy_lr)
        policy_nh_opt = self.hparams.optim(self.policy_nh.parameters(), lr=self.hparams.policy_lr)
        policy_rt_opt = self.hparams.optim(self.policy_rt.parameters(), lr=self.hparams.policy_lr)
        # return value_opt, policy_opt
        return value_opt, policy_ch_opt, policy_nh_opt, policy_rt_opt

    # def optimizer_step(self, *args, **kwargs):
    #     """
    #     Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic
    #     for each data sample.
    #     """
    #     for i in range(self.hparams.nb_optim_iters):
    #         super().optimizer_step(*args, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceSourceDataset(self._dataset)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader
    
    # def _dataloader(self) -> DataLoader:
    #     """Initialize the Replay Buffer dataset used for retrieving experiences"""
    #     dataset = RLDataset_PPO_shuffle(self.buffer, sum(self.num_samples), self.hparams.epoch_repeat)
    #     dataloader = DataLoader(dataset=dataset, 
    #         batch_size=self.hparams.batch_size,
    #         num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
    #     )
    #     return dataloader

    # def train_dataloader(self):
    #     dataset = RLDataset_PPO_shuffle(self.buffer, sum(self.num_samples), self.hparams.epoch_repeat)
    #     dataloader = DataLoader(
    #         dataset=dataset,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
    #     )
    #     return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):


        # self.log("epoch/Return", sum(self.ep_returns), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Delay', sum(self.delay)/len(self.delay), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Throughput', sum(self.throughput)/len(self.throughput), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Enrgy_Consumption', sum(self.engcons)/len(self.engcons), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Dropped_Packets', sum(self.droppackets)/len(self.droppackets), on_step=False, on_epoch=True)
        # self.log('epoch/Performance/Resedual_Energy_var', sum(self.resenergy)/len(self.resenergy), on_step=False, on_epoch=True)
            
        obs_b, loc_b, scale_b, action_b, reward_b, done_b, nxt_obs_b = batch        

        rev_nxt_obs = self.env.scaler.inverse_transform(T.clone(nxt_obs_b.data).detach().cpu().numpy())
        # self.log("episode/Return", reward_b.sum(), enable_graph=True)
        # self.log('episode/Performance/Delay', rev_nxt_obs[:,1].mean(), enable_graph=True)
        # self.log('episode/Performance/Throughput', rev_nxt_obs[:,2].mean(), enable_graph=True)
        # self.log('episode/Performance/Enrgy_Consumption', rev_nxt_obs[:,3].mean(), enable_graph=True)
        # self.log('episode/Performance/Dropped_Packets', rev_nxt_obs[:,10].mean(), enable_graph=True)
        # self.log('episode/Performance/Resedual_Energy_var', rev_nxt_obs[:,-1].mean(), enable_graph=True)
        
        # if self.global_step == 0 or self.global_step == 1:            
        #     self.tb_logger._experiment.add_scalars('Steps/Return', {
        #         'R': 0.0
        #         }, global_step=self.global_step
        #     )
        # else:
        self.tb_logger._experiment.add_scalars('Steps/Return', {
            'R': reward_b.sum()
            }, global_step=self.global_step
        )
        self.tb_logger._experiment.add_scalars('Steps/Delay', {
            'NE': rev_nxt_obs[:,1].mean()
            }, global_step=self.global_step
        )
        self.tb_logger._experiment.add_scalars('Steps/Throughput', {
            'NT': rev_nxt_obs[:,2].mean() 
            }, global_step=self.global_step
        )
        self.tb_logger._experiment.add_scalars('Steps/Energy_Consumption', {
            'NGC': rev_nxt_obs[:,3].mean()
            }, global_step=self.global_step
        )
        self.tb_logger._experiment.add_scalars('Steps/Dropped_Packets', {
            'NDR': rev_nxt_obs[:,10].mean()
            }, global_step=self.global_step
        )
        self.tb_logger._experiment.add_scalars('Steps/Resedual_Energy_Variance', {
            'NGVar': rev_nxt_obs[:,-1].mean()
            }, global_step=self.global_step
        )
        
        state_values = self.value_net(T.hstack((obs_b,action_b)))
       
        with T.no_grad():
            # nxt_new_loc, nxt_new_scale = self.target_policy(nxt_obs_b)
            # nxt_action = T.normal(nxt_new_loc, nxt_new_scale)
            # nxt_state_values = self.target_val_net(T.hstack((nxt_obs_b,nxt_action)))
            # '''
            nxt_new_loc_ch, nxt_new_scale_ch = self.target_policy_ch(nxt_obs_b)
            nxt_action_ch = T.normal(nxt_new_loc_ch, nxt_new_scale_ch)
            nxt_new_loc_nh, nxt_new_scale_nh = self.target_policy_nh(nxt_obs_b)
            nxt_action_nh = T.normal(nxt_new_loc_nh, nxt_new_scale_nh)
            nxt_new_loc_rt, nxt_new_scale_rt = self.target_policy_rt(nxt_obs_b)
            nxt_action_rt = T.normal(nxt_new_loc_rt, nxt_new_scale_rt)
            nxt_state_values = self.target_val_net(T.hstack((nxt_obs_b,nxt_action_ch,nxt_action_nh,nxt_action_rt)))
            # '''
            nxt_state_values[done_b] = 0.0
            target = reward_b + self.hparams.gamma * nxt_state_values
        
        advantages = (target - state_values).detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if optimizer_idx == 0:
            loss = self.hparams.loss_fn(state_values.float(), target.float())
            self.ep_value_loss.append(loss.unsqueeze(0))
            # self.log('value_net/loss', loss)  
            self.tb_logger._experiment.add_scalars('Steps/Value/Loss', {
                'loss': loss
                }, global_step=self.global_step
            )         
            return loss

        elif optimizer_idx == 1:
            # new_loc, new_scale = self.policy(obs_b)
            # dist = Normal(new_loc, new_scale)
            # log_prob = dist.log_prob(action_b).sum(dim=-1, keepdim=True)

            # prv_dist = Normal(loc_b, scale_b)
            # prv_log_prob = prv_dist.log_prob(action_b).sum(dim=-1, keepdim=True)
            
            # rho = T.exp(log_prob - prv_log_prob)

            # surrogate_1 = rho * advantages
            # surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            # policy_loss = - T.minimum(surrogate_1, surrogate_2)
            # entropy = dist.entropy().sum(dim=-1, keepdim=True)
            # loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()

            # self.ep_policy_loss.append(policy_loss.mean().unsqueeze(0))
            # self.ep_entropy.append(entropy.mean().unsqueeze(0))
            # self.log('episode/Policy/Loss', policy_loss.mean())
            # self.log('episode/Policy/Entropy', entropy.mean())

            # return loss
            # '''
            new_loc_ch, new_scale_ch = self.policy_ch(obs_b)
            dist_ch = Normal(new_loc_ch, new_scale_ch)
            log_prob_ch = dist_ch.log_prob(action_b[:,0]).sum(dim=-1, keepdim=True)

            prv_dist_ch = Normal(loc_b[:,0], scale_b[:,0])
            prv_log_prob_ch = prv_dist_ch.log_prob(action_b[:,0]).sum(dim=-1, keepdim=True)
            
            rho_ch = T.exp(log_prob_ch - prv_log_prob_ch)
            # rho = log_prob / prv_log_prob

            surrogate_1_ch = rho_ch * advantages
            surrogate_2_ch = rho_ch.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            policy_loss_ch = - T.minimum(surrogate_1_ch, surrogate_2_ch)
            entropy_ch = dist_ch.entropy().sum(dim=-1, keepdim=True)
            loss_ch = (policy_loss_ch - self.hparams.entropy_coef * entropy_ch).mean()

            self.ep_policy_loss_ch.append(policy_loss_ch.mean().unsqueeze(0))
            self.ep_entropy_ch.append(entropy_ch.mean().unsqueeze(0))
            self.tb_logger._experiment.add_scalars('Steps/Policy/Loss', {
                'CH': policy_loss_ch.mean(), 
                }, global_step=self.global_step
            )
            self.tb_logger._experiment.add_scalars('Steps/Policy/Entropy', {
                'CH': entropy_ch.mean()
                }, global_step=self.global_step
            )

            return loss_ch
            
        elif optimizer_idx == 2:

            new_loc_nh, new_scale_nh = self.policy_nh(obs_b)
            dist_nh = Normal(new_loc_nh, new_scale_nh)
            log_prob_nh = dist_nh.log_prob(action_b[:,1]).sum(dim=-1, keepdim=True)

            prv_dist_nh = Normal(loc_b[:,1], scale_b[:,1])
            prv_log_prob_nh = prv_dist_nh.log_prob(action_b[:,1]).sum(dim=-1, keepdim=True)
            
            rho_nh = T.exp(log_prob_nh - prv_log_prob_nh)
            # rho = log_prob / prv_log_prob

            surrogate_1_nh = rho_nh * advantages
            surrogate_2_nh = rho_nh.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            policy_loss_nh = - T.minimum(surrogate_1_nh, surrogate_2_nh)
            entropy_nh = dist_nh.entropy().sum(dim=-1, keepdim=True)
            loss_nh = (policy_loss_nh - self.hparams.entropy_coef * entropy_nh).mean()

            self.ep_policy_loss_nh.append(policy_loss_nh.mean().unsqueeze(0))
            self.ep_entropy_nh.append(entropy_nh.mean().unsqueeze(0))
            self.tb_logger._experiment.add_scalars('Steps/Policy/Loss', {
                'NH': policy_loss_nh.mean(), 
                }, global_step=self.global_step
            )
            self.tb_logger._experiment.add_scalars('Steps/Policy/Entropy', {
                'NH': entropy_nh.mean()
                }, global_step=self.global_step
            )

            return loss_nh

        elif optimizer_idx == 3:

            new_loc_rt, new_scale_rt = self.policy_rt(obs_b)
            dist_rt = Normal(new_loc_rt, new_scale_rt)
            log_prob_rt = dist_rt.log_prob(action_b[:,2]).sum(dim=-1, keepdim=True)

            prv_dist_rt = Normal(loc_b[:,2], scale_b[:,2])
            prv_log_prob_rt = prv_dist_rt.log_prob(action_b[:,2]).sum(dim=-1, keepdim=True)
            
            rho_rt = T.exp(log_prob_rt - prv_log_prob_rt)
            # rho = log_prob / prv_log_prob

            surrogate_1_rt = rho_rt * advantages
            surrogate_2_rt = rho_rt.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            policy_loss_rt = - T.minimum(surrogate_1_rt, surrogate_2_rt)
            entropy_rt = dist_rt.entropy().sum(dim=-1, keepdim=True)
            loss_rt = (policy_loss_rt - self.hparams.entropy_coef * entropy_rt).mean()

            self.ep_policy_loss_rt.append(policy_loss_rt.mean().unsqueeze(0))
            self.ep_entropy_rt.append(entropy_rt.mean().unsqueeze(0))
            self.tb_logger._experiment.add_scalars('Steps/Policy/Loss', {
                'RT': policy_loss_rt.mean(), 
                }, global_step=self.global_step
            )
            self.tb_logger._experiment.add_scalars('Steps/Policy/Entropy', {
                'RT': entropy_rt.mean()
                }, global_step=self.global_step
            )

            return loss_rt
            # '''

    def training_epoch_end(self, training_step_outputs):
        self.target_val_net.load_state_dict(self.value_net.state_dict())   
        # self.target_policy.load_state_dict(self.policy.state_dict())    
        self.target_policy_ch.load_state_dict(self.policy_ch.state_dict())   
        self.target_policy_nh.load_state_dict(self.policy_nh.state_dict())   
        self.target_policy_rt.load_state_dict(self.policy_rt.state_dict())   

        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        dataset = map(np.vstack, zip(*self.buffer))
        obs_b, loc_b, scale_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, dataset)

        # rev_nxt_obs_b = self.env.max_obs *nxt_obs_b
        # rev_nxt_obs_b = nxt_obs_b * self.env.max_obs
        rev_nxt_obs_b = self.env.scaler.inverse_transform(nxt_obs_b)
        # rev_nxt_obs_b = nxt_obs_b.detach() * np.sqrt(self.env.obs_rms.var + self.env.epsilon) + self.env.obs_rms.mean
        # self.log('epoch/ValueNet/Loss', T.cat(self.ep_value_loss).mean()) 
        # self.log('epoch/Policy/Loss', T.cat(self.ep_policy_loss).mean())
        # self.log('epoch/Policy/Entropy', T.cat(self.ep_entropy).mean())

        # self.log('epoch/Return', reward_b.sum())
        # self.log('epoch/Performance/Delay', rev_nxt_obs_b[:,1].mean())
        # self.log('epoch/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
        # self.log('epoch/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
        # self.log('epoch/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
        # self.log('epoch/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,-1].mean())
        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/experiences.csv')
            quietRun('rm -r outputs/logs/lightning_logs/version_1')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        self.tb_logger = TensorBoardLogger(save_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            logger=self.tb_logger,
            # reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        
        self.play_episodes()
        trainer.fit(self) 

class PPO_Agent3(LightningModule):
    """ 
    """
    def __init__(self, env, num_envs=50, batch_size=2, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, gamma=0.94, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.env = env

        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high
        self.obs_nds = None
        self.policy = PPO_GradientPolicy(self.obs_dims, hidden_size, self.action_dims)
        self.target_policy = copy.deepcopy(self.policy)

        self.value_net = PPO_ValueNet(self.obs_dims+self.action_dims, hidden_size)
        self.target_val_net = copy.deepcopy(self.value_net) 


        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.ep_returns = []
        self.delay = []
        self.throughput = []
        self.engcons = []
        self.droppackets = []
        self.resenergy = []
        self.ep_step = 0

        self.save_hyperparameters('batch_size', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')

    @T.no_grad()
    def play_episodes(self, policy=None):
        # self.buffer = []
        # self.env.num_samples = 0
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.ep_returns = []
        self.delay = []
        self.throughput = []
        self.engcons = []
        self.droppackets = []
        self.resenergy = []
        returns = [0]
        obs, nodes = self.env.reset()   
        print('COLLECTING DATA ....')       
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            loc, scale = self.policy(obs)
            action = T.normal(loc, scale)
            action = action.detach().cpu().numpy()
            loc = loc.detach().cpu().numpy()
            scale = scale.detach().cpu().numpy()

            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nodes))
            self.buffer.append((obs, loc, scale, action, reward, done, nxt_obs))
            self.num_samples.append(nxt_obs.shape[0])
            self.ep_returns.append(returns[step])
            returns.append(reward.sum())
            rev_nxt_obs = self.env.scaler.inverse_transform(nxt_obs)
            self.delay.append(rev_nxt_obs[:,0].mean())
            self.throughput.append(rev_nxt_obs[:,1].mean())
            self.engcons.append(rev_nxt_obs[:,2].mean())
            self.droppackets.append(rev_nxt_obs[:,9].mean())
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(action, columns=self.env.action_cols),
                pd.DataFrame(nxt_obs, columns=['nxt_obs'+str(i) for i in range(nxt_obs.shape[1])]), 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            
            obs = nxt_obs

            self.tb_logger.add_scalars('Episode/Return', {
                'R': reward.sum()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Delay', {
                'DE': rev_nxt_obs[:,0].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Throughput', {
                'TH': rev_nxt_obs[:,1].mean() 
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Energy_Consumption', {
                'EC': rev_nxt_obs[:,2].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Dropped_Packets', {
                'DP': rev_nxt_obs[:,9].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Report_TTI', {
                'RTTI': rev_nxt_obs[:,-1].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Energy_Consumption_Var', {
                'DCVar': np.sqrt(np.square((rev_nxt_obs[:,2])-np.mean(rev_nxt_obs[:,2]))).mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Report_TTI_Var', {
                'RTTIVar': np.sqrt(np.square((rev_nxt_obs[:,-1])-np.mean(rev_nxt_obs[:,-1]))).mean()
                }, global_step=self.ep_step
            )

            self.ep_step += 1

    def _dataset(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, loc_b, scale_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(sum(self.num_samples)):
            yield obs_b[i], loc_b[i], scale_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def _dataset_shuffle(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, loc_b, scale_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.hparams.epoch_repeat):
            idx = list(range(sum(self.num_samples)))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], loc_b[i], scale_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def configure_optimizers(self):
        value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)
        return value_opt, policy_opt

    # def optimizer_step(self, *args, **kwargs):
    #     """
    #     Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic
    #     for each data sample.
    #     """
    #     for i in range(self.hparams.nb_optim_iters):
    #         super().optimizer_step(*args, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceSourceDataset(self._dataset)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader
    
    def training_step(self, batch, batch_idx, optimizer_idx):


        obs_b, loc_b, scale_b, action_b, reward_b, done_b, nxt_obs_b = batch        

        # rev_nxt_obs = self.env.scaler.inverse_transform(T.clone(nxt_obs_b.data).detach().cpu().numpy())

        state_values = self.value_net(T.hstack((obs_b,action_b)))
       
        with T.no_grad():
            nxt_new_loc, nxt_new_scale = self.target_policy(nxt_obs_b)
            nxt_action = T.normal(nxt_new_loc, nxt_new_scale)
            nxt_state_values = self.target_val_net(T.hstack((nxt_obs_b,nxt_action)))

            nxt_state_values[done_b] = 0.0
            target = reward_b + self.hparams.gamma * nxt_state_values
        
        advantages = (target - state_values).detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if optimizer_idx == 0:
            loss = self.hparams.loss_fn(state_values.float(), target.float())
            self.ep_value_loss.append(loss.unsqueeze(0))
            self.tb_logger.add_scalars('Steps/Value/Loss', {
                'loss': loss
                }, global_step=self.global_step
            )         
            return loss

        elif optimizer_idx == 1:
            new_loc, new_scale = self.policy(obs_b)
            dist = Normal(new_loc, new_scale)
            log_prob = dist.log_prob(action_b).sum(dim=-1, keepdim=True)

            prv_dist = Normal(loc_b, scale_b)
            prv_log_prob = prv_dist.log_prob(action_b).sum(dim=-1, keepdim=True)
            
            rho = T.exp(log_prob - prv_log_prob)

            surrogate_1 = rho * advantages
            surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            policy_loss = - T.minimum(surrogate_1, surrogate_2)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()

            self.ep_policy_loss.append(policy_loss.mean().unsqueeze(0))
            self.ep_entropy.append(entropy.mean().unsqueeze(0))
            self.tb_logger.add_scalars('Steps/Policy/Loss', {
                'loss': policy_loss.mean()
                }, global_step=self.global_step
            )       
            self.tb_logger.add_scalars('Steps/Policy/Entropy', {
                'entropy': entropy.mean()
                }, global_step=self.global_step
            )       

            return loss

    def training_epoch_end(self, training_step_outputs):
        self.target_val_net.load_state_dict(self.value_net.state_dict())   
        self.target_policy.load_state_dict(self.policy.state_dict())     

        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/')
            # quietRun('rm -r outputs/logs/experiences.csv')
            # quietRun('rm -r outputs/logs/lightning_logs/version_0')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            # logger=self.tb_logger,
            # reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        
        self.play_episodes()
        trainer.fit(self) 

class PPO_Agent4(LightningModule):
    """ 
    """
    def __init__(self, ctrl, num_envs=50, batch_size=2, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, gamma=0.99, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.ctrl = ctrl
        self.networkGraph = self.ctrl.networkGraph

        self.obs_cols = ['port', 'intftypeval', 'datatypeval', 'distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', 'engcons', \
        'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'rptti'] 
        self.cal_cols = ['engcons_var', 'rptti_mean']
        self.action_cols = ['clhop', 'nxhop', 'rptti']
        self.action_space = np.empty((0, len(self.action_cols)))
        self.observation_space = np.empty((0, len(self.obs_cols)+len(self.cal_cols)))

        self.obs_dims = self.observation_space.shape[1]
        self.action_dim = self.action_space.shape[1]
        
        self.obs_nds = None
        self.policy = PPO_Policy(self.obs_dims, hidden_size, self.action_dim)
        self.target_policy = copy.deepcopy(self.policy)
        self.value_net = PPO_ValueNet(self.obs_dims+self.action_dim, hidden_size)
        # self.value_net = PPO_ValueNet(self.obs_dims, hidden_size)
        self.target_val_net = copy.deepcopy(self.value_net) 

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.ep_step = 0

        self.save_hyperparameters('batch_size', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')
        
    def reset(self, nds=None):
        """Get network state observation"""
        state, nodes, sink_state, sink_state_nodes = self.networkGraph.getState(nodes=nds, cols=self.obs_cols)                
        obs = np.column_stack((state, np.repeat(np.sqrt(np.square(state[:,10] - np.mean(state[:,10]))), state.shape[0], axis=0).reshape(-1,1), np.repeat(state[:,-1].mean(), state.shape[0], axis=0).reshape(-1,1)))

        data = pd.concat([pd.DataFrame(obs, columns=self.obs_cols+self.cal_cols, dtype=float)], axis=1)
        return data, nodes
    
    def getReward(self, obs, prv_obs):
        # R = np.repeat(np.mean((obs['throughput']/250  - obs['delay']/ct.MAX_DELAY - obs['drpackets']/100).to_numpy()), obs.shape[0], axis=0).reshape(-1,1)
        # R = (obs['throughput']/250 - obs['delay']/ct.MAX_DELAY - obs['drpackets']/100).to_numpy().reshape(-1,1)
        # R = (obs['throughput'] - obs['delay']- obs['drpackets']).to_numpy().reshape(-1,1)
        R = (1 - (obs['delay']/obs['throughput'])).to_numpy().reshape(-1,1)
        # R = (obs['throughput']-obs['delay']).to_numpy().reshape(-1,1)
        # R = ct.MAX_DELAY - np.mod(np.sqrt(np.square(obs['rptti'].to_numpy().reshape(-1,1)-np.mean(obs['rptti']))), ct.MAX_DELAY)
            # ((obs['rptti'].to_numpy().reshape(-1,1) - ct.MAX_RP_TTI) / ct.MAX_DELAY) - \
            # np.abs(((obs['rptti'].to_numpy().reshape(-1,1) - ct.MAX_RP_TTI) / ct.MAX_DELAY))
        # R = np.repeat(np.sqrt(np.mean(np.square(obs['rptti'].to_numpy().reshape(-1,1) - np.mean(obs['rptti'])))), obs.shape[0], axis=0)
        # R = (np.mean(obs['rptti']) - obs['rptti'].to_numpy().reshape(-1,1))*(obs['distance'].to_numpy().reshape(-1,1) - np.mean(obs['distance']))
        # R = np.sqrt(np.square(obs['rptti'].to_numpy().reshape(-1,1) - np.mean(obs['rptti'])))
        
        # R = np.nan_to_num(R, nan=0)
        # R = throughput-delay-engcons-drpackets-\
        #         np.sqrt(np.square(engcons-np.mean(engcons)))+\
        #         np.sqrt(np.square(rptti-np.mean(rptti)))+\
        #         (np.mean(rptti) - rptti)*(dist - np.mean(dist))

        # R = 1 / (1 + np.exp(-R)) # sigmoid
        # R = np.log1p(np.exp(-np.abs(R))) + np.maximum(R, 0) # softplus
        return np.nan_to_num(R, nan=0)
    
    def step(self, action, obs, obs_nds):
        # print(f'obs:\n{rev_obs}')
        # print(f'action:\n{action}')
        # print(f'obs_nds:\n{obs_nds}')
        action = MinMaxScaler((0,1)).fit_transform(action)
        # action = np.argmax(action, axis=-1).reshape(-1,1)
        act_nds = dict(zip(obs_nds.flatten(), action))
        # print(f'act_nds:\n{act_nds}')
        
        isaggr_idxs = np.where((action[:,0] > 0.50))
        # isaggr_idxs = np.where((obs[:,0]<(obs[:,0].max()*action[:,0]+obs[:,0].max()*action[:,0]*0.25)) & (obs[:,0]>(obs[:,0].max()*action[:,0]-obs[:,0].max()*action[:,0]*0.25)))
        # isaggr_idxs = np.where((obs[:,0]<(obs[:,0].max()*action[:,0]+obs[:,0].max()*action[:,0]*0.25)) & (obs[:,0]>(obs[:,0].max()*action[:,0]-obs[:,0].max()*action[:,0]*0.25)) &
        #                        (obs[:,5]<(obs[:,5].max()*action[:,1]+obs[:,5].max()*action[:,1]*0.25)) & (obs[:,5]>(obs[:,5].max()*action[:,1]-obs[:,5].max()*action[:,1]*0.25)))
        isaggr_action_nodes = obs_nds[isaggr_idxs,:].flatten().tolist()
        try:
            for nd, act in act_nds.items():
                neighbors = [edge[1] for edge in list(self.networkGraph.getGraph().edges(nbunch=nd, data=True, keys=True))]
                nd_pos = self.networkGraph.getPosition(nd)
                act_angle = act[1] * pi # angle between 0 and π radians
                act_nxh_node = nd
                act_nxh_value = pi
                for nr in neighbors:
                    src_dist = self.networkGraph.getDistance(nd)
                    dst_dist = self.networkGraph.getDistance(nr)
                    nd_nr_edge = self.networkGraph.getEdge(nd, nr)
                    nr_pos = self.networkGraph.getPosition(nr)
                    y = nr_pos[1] - nd_pos[1]
                    x = nr_pos[0] - nd_pos[0]
                    angle = atan2(y, x)
                    angle = angle if angle > 0 else (angle + (2*pi))
                    vl = abs(act_angle - angle)
                    if vl < act_nxh_value and src_dist > dst_dist:
                        act_nxh_node = nr

                # action value
                val = int.to_bytes(ct.DRL_CH_INDEX, 1, 'big', signed=False)+int.to_bytes(1 if nd in isaggr_action_nodes else 0, ct.DRL_CH_LEN, 'big', signed=False)+\
                    int.to_bytes(ct.DRL_NH_INDEX, 1, 'big', signed=False)+int.to_bytes(Addr(re.sub(r'^.*?.', '', act_nxh_node)[1:]).intValue(), ct.DRL_NH_LEN, 'big', signed=False)+\
                    int.to_bytes(ct.DRL_RT_INDEX, 1, 'big', signed=False)+int.to_bytes(int(act[2] * (ct.MAX_RP_TTI-ct.MIN_RP_TTI))+ct.MIN_RP_TTI, ct.DRL_RT_LEN, 'big', signed=False)
                
                    # int.to_bytes(ct.DRL_CH_INDEX, 1, 'big', signed=False)+int.to_bytes(1 if nd in isaggr_action_nodes else 0, ct.DRL_CH_LEN, 'big', signed=False)+\
                    # int.to_bytes(ct.DRL_NH_INDEX, 1, 'big', signed=False)+int.to_bytes(Addr(re.sub(r'^.*?.', '', act_nxh_node)[1:]).intValue(), ct.DRL_NH_LEN, 'big', signed=False)+\
                    # int.to_bytes(ct.DRL_RT_INDEX, 1, 'big', signed=False)+int.to_bytes(int(act[2] * int((ct.MAX_RP_TTI-ct.MIN_RP_TTI)/ct.MAX_DELAY))*ct.MAX_DELAY + ct.MIN_RP_TTI, ct.DRL_RT_LEN, 'big', signed=False)
                    # int.to_bytes(ct.DRL_RT_INDEX, 1, 'big', signed=False)+int.to_bytes(int(act[2] * (ct.MAX_RP_TTI-ct.MIN_RP_TTI))+ct.MIN_RP_TTI, ct.DRL_RT_LEN, 'big', signed=False)
                    # int.to_bytes(ct.DRL_RT_INDEX, 1, 'big', signed=False)+int.to_bytes(int(act[2] * (obs['rptti'].max()-obs['rptti'].min()))+int(obs['rptti'].min()), ct.DRL_RT_LEN, 'big', signed=False)

                # send the action to the data-plane
                sel_route = []
                init_len = ct.DIST_MAX + 1  
                sinkId = None
                for sink in self.ctrl.sinks:
                    route = nx.shortest_path(self.networkGraph.getGraph(), source=nd, target=sink)
                    if len(route) < init_len:
                        sel_route = route
                        init_len = len(sel_route)
                        sinkId = sink

                if sel_route:
                    route = [Addr(re.sub(r'^.*?.', '', x)[1:]) for x in sel_route]
                    route.reverse()
                    asyncio.get_running_loop().run_until_complete(self.ctrl.setDRLAction(net=int(sinkId.split('.')[0]), sinkId=sinkId, sinkAddr=route[0], dst=route[-1], newVal=val, path=route))
        except Exception as ex:
            logger.warn(ex)
        # '''

        time.sleep(15)
        next_obs, _ = self.reset(obs_nds)
        reward = self.getReward(next_obs, obs)
        # print(f"result:\n {result}")
        # print(f'reward\n{reward}')
        # print(f'reward\n:{reward}')
        # self.ctrl.logger.warn(f'next obs: {next_obs}')
        # self.ctrl.logger.warn(f'reward: {reward}')
        done = np.zeros((next_obs.shape[0],1), dtype=bool) #TODO change the logic when to set done to True (since it is a continouse process target optimization are always changing as per the network progress and resource drained)           
        info = np.empty((next_obs.shape[0],1), dtype=str)
        return next_obs, reward, done, info
        # return next_obs, reward, info

    @T.no_grad()
    def play_episodes(self, policy=None):
        # self.buffer = []
        # self.env.num_samples = 0
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        delay = []
        throughput = []
        engcons = []
        droppackets = []
        resenergy = []
        returns = [0]
        obs, nodes = self.reset()
        print('COLLECTING DATA ....')     
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            log_prob, action = self.policy(obs.to_numpy())
            action = action.detach().cpu().numpy()
            log_prob = log_prob.detach().cpu().numpy()
            # print(f'action\n{action}')
            # print(f'log_prob\n{log_prob}')
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.step(action, obs, nodes)
            # print(f'obs:{obs.shape} reward: {reward.shape} done: {done.shape} info: {info.shape} nxt_obs: {nxt_obs.shape}')
            self.buffer.append((obs.to_numpy(), log_prob, action, reward, done, nxt_obs.to_numpy()))
            self.num_samples.append(nxt_obs.shape[0])
            returns.append(reward.sum())
            # rev_nxt_obs = self.scaler.inverse_transform(nxt_obs)
            # rev_nxt_obs = np.log(np.divide(nxt_obs,1-nxt_obs))
            rev_nxt_obs = nxt_obs
            delay.append(rev_nxt_obs['delay'].mean())
            throughput.append(rev_nxt_obs['throughput'].mean())
            engcons.append(rev_nxt_obs['engcons'].mean())
            droppackets.append(rev_nxt_obs['drpackets'].mean())
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                obs, 
                pd.DataFrame(action, columns=self.action_cols),
                nxt_obs, 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
                
            obs = nxt_obs

            self.tb_logger.add_scalars('Episode/Return', {
                'R': reward.sum()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Delay', {
                'DE': rev_nxt_obs['delay'].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Throughput', {
                'TH': rev_nxt_obs['throughput'].mean() 
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Energy_Consumption', {
                'EC': rev_nxt_obs['engcons'].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Dropped_Packets', {
                'DP': rev_nxt_obs['drpackets_val'].mean()
                }, global_step=self.ep_step
            )
            # self.tb_logger.add_scalars('Episode/Report_TTI', {
            #     'RTTI': rev_nxt_obs['rptti'].mean()
            #     }, global_step=self.ep_step
            # )
            # self.tb_logger.add_scalars('Episode/Energy_Consumption_Var', {
            #     'DCVar': np.sqrt(np.square((rev_nxt_obs['batt'])-np.mean(rev_nxt_obs['batt']))).mean()
            #     }, global_step=self.ep_step
            # )
            # self.tb_logger.add_scalars('Episode/Report_TTI_Var', {
            #     'RTTIVar': np.sqrt(np.square((rev_nxt_obs['rptti'])-np.mean(rev_nxt_obs['rptti']))).mean()
            #     }, global_step=self.ep_step
            # )

            self.ep_step += 1

        self.tb_logger.add_scalars('Epoch/Return', {
            'R': np.array(returns).sum()
            }, global_step=self.ep_step
        )

    def _dataset(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(sum(self.num_samples)):
            yield obs_b[i], log_prob_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def _dataset_shuffle(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.hparams.epoch_repeat):
            idx = list(range(sum(self.num_samples)))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], log_prob_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def configure_optimizers(self):
        value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)   
        return value_opt, policy_opt

    # def optimizer_step(self, *args, **kwargs):
    #     """
    #     Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic
    #     for each data sample.
    #     """
    #     for i in range(self.hparams.nb_optim_iters):
    #         super().optimizer_step(*args, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceSourceDataset(self._dataset)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):

        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = batch        

        # rev_nxt_obs = self.env.scaler.inverse_transform(T.clone(nxt_obs_b.data).detach().cpu().numpy())
        
        state_values = self.value_net(T.hstack((obs_b, action_b)))
        # state_values = self.value_net(obs_b)
       
        with T.no_grad():
            _, nxt_action = self.target_policy(nxt_obs_b)
            nxt_state_values = self.target_val_net(T.hstack((nxt_obs_b, nxt_action)))
            # nxt_state_values = self.target_val_net(nxt_obs_b)
            nxt_state_values[done_b] = 0.0 
            target = reward_b + self.hparams.gamma * nxt_state_values
        
        advantages = (target - state_values).detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if optimizer_idx == 0:
            loss = self.hparams.loss_fn(state_values.float(), target.float())
            self.ep_value_loss.append(loss.unsqueeze(0))
            self.tb_logger.add_scalars('Training/Value/Loss', {
                'loss': loss
                }, global_step=self.global_step
            )          
            return loss

        elif optimizer_idx == 1:
            log_prob, _ = self.policy(obs_b)
            prv_log_prob = log_prob_b
            
            rho = T.exp(log_prob - prv_log_prob)
            # rho = log_prob / prv_log_prob

            surrogate_1 = rho * advantages
            surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            policy_loss = - T.minimum(surrogate_1, surrogate_2)
            entropy = -T.sum(action_b*log_prob_b, dim=-1, keepdim=True)
            loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()
            self.tb_logger.add_scalars('Training/Policy/Loss', {
                'loss': policy_loss.mean()
                }, global_step=self.global_step
            )       
            self.tb_logger.add_scalars('Training/Policy/Entropy', {
                'entropy': entropy.mean()
                }, global_step=self.global_step
            )       
            return loss
                
    def training_epoch_end(self, training_step_outputs):
        self.target_val_net.load_state_dict(self.value_net.state_dict())            
        self.target_policy.load_state_dict(self.policy.state_dict())
        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/')
            # quietRun('rm -r outputs/logs/experiences.csv')
            # quietRun('rm -r outputs/logs/lightning_logs/version_0')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            # logger=self.tb_logger,
            # reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        T.autograd.detect_anomaly(True)
        self.play_episodes()
        trainer.fit(self) 

class PPO_Agent5(LightningModule):
    """ 
    """
    def __init__(self, ctrl, num_envs=50, batch_size=2, obs_time = 20, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, gamma=0.99, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.ctrl = ctrl
        self.networkGraph = self.ctrl.networkGraph

        self.obs_cols = ['port', 'intftypeval', 'datatypeval', 'distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', 'engcons', \
        'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val'] 
        self.cal_cols = ['ts']
        self.action_cols = ['batt', 'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val']
        self.action_space = np.empty((0, len(self.action_cols)))
        self.observation_space = np.empty((0, len(self.cal_cols)+len(self.obs_cols)))

        self.obs_dims = self.observation_space.shape[1]
        self.action_dims = self.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high
        self.obs_nds = None
        self.policy = PPO_Att_Policy(self.obs_dims, hidden_size, self.action_dims)
        self.target_policy = copy.deepcopy(self.policy)
        # self.value_net = PPO_Att_ValueNet(self.obs_dims+self.action_dims, hidden_size)
        self.value_net = PPO_Att_ValueNet(self.obs_dims, hidden_size)
        self.target_val_net = copy.deepcopy(self.value_net) 

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.ep_step = 0
        
        self.save_hyperparameters('batch_size', 'obs_time', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')
        
    def reset(self, nds=None):
        """Get network state observation for certain amount of time"""
        obs_time = self.hparams.obs_time
        obs = self.observation_space
        _nodes = np.empty((0,1))
        while obs_time > 0:
            state, nodes, sink_state, sink_state_nodes = self.networkGraph.getState(nodes=nds, cols=self.obs_cols)                
            obs = np.vstack((obs, np.column_stack((np.repeat(int(time.time()), state.shape[0], axis=0).reshape(-1,1), state))))
            _nodes = np.vstack((_nodes, nodes))
            nds = nodes
            time.sleep(1)
            obs_time -= 1
            # while int(time.time()) - ts < 1:
            #     continue
            # obs_time -= (int(time.time()) - ts)
            # ts = int(time.time())
        data = pd.concat([pd.DataFrame(obs, columns=self.cal_cols+self.obs_cols, dtype=float)], axis=1)
        return data, _nodes
    
    def getReward(self, obs, prv_obs, action):
        # R = -np.square(prv_obs.get(self.action_cols).to_numpy().max() - action.to_numpy().min()) - \
        #     -np.square(obs.get(self.action_cols).to_numpy().min() - action.to_numpy().max())
        R = (((action - prv_obs.get(self.action_cols)) * (obs.get(self.action_cols) - action)) / (prv_obs.get(self.action_cols).max())**2).to_numpy().sum(axis=-1, keepdims=True)
        # R = np.nan_to_num(R, nan=0).mean(axis=-1, keepdims=True)
        R = np.nan_to_num(R, nan=0)
        # print(f'R:\n{R}')
        # R = np.tanh(R)
        # R = 1 / (1 + np.exp(-R)) # sigmoid
        # R = np.log(1 + np.exp(-np.abs(R))) + np.maximum(R,0) # softplus
        # R = 1 / (1 + np.exp(np.log(np.sqrt(np.square(R)))))
        # R = np.log(1/(1+np.exp(-R)))        
        # print(f'R size: {R.shape}')
        # print(f'R:\n{R}')
        # R = prv_obs['throughput']-prv_obs['delay']-prv_obs['engcons']-prv_obs['drpackets_val']-\
        #     np.sqrt(np.square(prv_obs['engcons']-np.mean(prv_obs['engcons'])))
        return R

    def step(self, action, obs, obs_nds):
        # print(f'ops_nodes\n:{obs_nds}')
        # print(f'action\n:{action}')
        # print(f'obs\n:{obs}')
        # act_nds = dict(zip(obs_nds, action))
        # action = MinMaxScaler((0,1)).fit_transform(action)
        action =  obs.get(self.action_cols).to_numpy() + obs.get(self.action_cols).to_numpy() * action
        nds = np.unique(obs_nds, axis=0)
        for nd in nds.flatten().tolist():
            sel_route = []
            init_len = ct.DIST_MAX + 1  
            sinkId = None
            for sink in self.ctrl.sinks:
                route = nx.shortest_path(self.networkGraph.getGraph(), source=nd, target=sink)
                if len(route) < init_len:
                    sel_route = route
                    init_len = len(sel_route)
                    sinkId = sink

            if sel_route:
                route = [Addr(re.sub(r'^.*?.', '', x)[1:]) for x in sel_route]
                route.reverse()
                # '''
                try:
                    entry = Entry()
                    entry.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1)
                        .setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST)
                        .setRhs(route[0].intValue()))
                    entry.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1)
                        .setLhsOperandType(ct.PACKET).setLhs(ct.SRC_INDEX).setRhsOperandType(ct.CONST)
                        .setRhs(route[-1].intValue()))
                    entry.addWindow(Window.fromString("P.TYP == "+ str(ct.REPORT)))
                    entry.addAction(DropAction())
                    entry.getStats().setTtl(int(time.time()))
                    entry.getStats().setIdle(self.hparams.obs_time)
                    asyncio.get_running_loop().run_until_complete(self.ctrl.setNodeRule(net=int(sinkId.split('.')[0]), sinkId=sinkId, sinkAddr=route[0], dst=route[-1], newVal=entry, path=route))
                except Exception as ex:
                    logger.warn(ex)

        prd_time = self.hparams.obs_time
        obs_tss = obs['ts'].unique().flatten().tolist()# obervation timestamps
        act_tss = np.hstack((obs['ts'].to_numpy().reshape(-1,1), action))
        i = 0
        while prd_time > 0:
            ts = int(time.time())
            pred_act = pd.DataFrame(act_tss[np.where(act_tss[:,0] == obs_tss[i])[0], 1:], columns=self.action_cols)
            # print(pred_act)
            state, nodes, _, _ = self.networkGraph.getState(nodes=nds, cols=self.action_cols)                
            actual_act = pd.DataFrame(obs, columns=self.action_cols)
            # print(actual_act)
            self.tb_logger.add_scalars('Episode/Battery', {
                'Prediction': pred_act['batt'].mean(),
                'actual':actual_act['batt'].mean()
                }, global_step=ts
            )
            self.tb_logger.add_scalars('Episode/txpackets_val', {
                'Prediction': pred_act['txpackets_val'].mean(),
                'actual':actual_act['txpackets_val'].mean()
                }, global_step=ts
            )
            self.tb_logger.add_scalars('Episode/txbytes_val', {
                'Prediction': pred_act['txbytes_val'].mean(),
                'actual':actual_act['txbytes_val'].mean()
                }, global_step=ts
            )
            self.tb_logger.add_scalars('Episode/rxpackets_val', {
                'Prediction': pred_act['rxpackets_val'].mean(),
                'actual':actual_act['rxpackets_val'].mean()
                }, global_step=ts
            )
            self.tb_logger.add_scalars('Episode/rxbytes_val', {
                'Prediction': pred_act['rxbytes_val'].mean(),
                'actual':actual_act['rxbytes_val'].mean()
                }, global_step=ts
            )
            # set predicted value in ctrl graph
            time.sleep(1)
            prd_time -= 1
            # while int(time.time()) - ts < 1:
            #     continue
            # prd_time -= (int(time.time()) - ts)
            # ts = int(time.time())
            i += 1
        
        next_obs, _ = self.reset(nds)            
        reward = self.getReward(next_obs, obs, pd.DataFrame(action, columns=self.action_cols))
        # print(f'reward\n:{reward}')
        # self.ctrl.logger.warn(f'next obs: {next_obs}')
        # self.ctrl.logger.warn(f'reward: {reward}')
        done = np.zeros((next_obs.shape[0],1), dtype=bool) #TODO change the logic when to set done to True (since it is a continouse process target optimization are always changing as per the network progress and resource drained)           
        info = np.empty((next_obs.shape[0],1), dtype=str)
        return next_obs, reward, done, info
        # return next_obs, reward, info 

    @T.no_grad()
    def play_episodes(self, policy=None):
        # self.buffer = []
        # self.env.num_samples = 0
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        delay = []
        throughput = []
        engcons = []
        droppackets = []
        resenergy = []
        returns = [0]
        obs, nodes = self.reset()
        print('COLLECTING DATA ....')     
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            log_prob, action = self.policy(obs.to_numpy())
            action = action.detach().cpu().numpy()
            log_prob = log_prob.detach().cpu().numpy()
            # print(f'action\n{action}')
            # print(f'log_prob\n{log_prob}')
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.step(action, obs, nodes)
            # print(f'obs:{obs.shape} reward: {reward.shape} done: {done.shape} info: {info.shape} nxt_obs: {nxt_obs.shape}')
            self.buffer.append((obs.to_numpy(), log_prob, action, reward, done, nxt_obs.to_numpy()))
            self.num_samples.append(nxt_obs.shape[0])
            returns.append(reward.sum())
            # rev_nxt_obs = self.scaler.inverse_transform(nxt_obs)
            # rev_nxt_obs = np.log(np.divide(nxt_obs,1-nxt_obs))
            rev_nxt_obs = nxt_obs
            delay.append(rev_nxt_obs['delay'].mean())
            throughput.append(rev_nxt_obs['throughput'].mean())
            engcons.append(rev_nxt_obs['engcons'].mean())
            droppackets.append(rev_nxt_obs['drpackets_val'].mean())
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                obs, 
                pd.DataFrame(action, columns=self.action_cols),
                nxt_obs, 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
                
            obs = nxt_obs

            self.tb_logger.add_scalars('Episode/Return', {
                'R': reward.sum()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Delay', {
                'DE': rev_nxt_obs['delay'].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Throughput', {
                'TH': rev_nxt_obs['throughput'].mean() 
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Energy_Consumption', {
                'EC': rev_nxt_obs['engcons'].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Dropped_Packets', {
                'DP': rev_nxt_obs['drpackets_val'].mean()
                }, global_step=self.ep_step
            )

            self.ep_step += 1

        self.tb_logger.add_scalars('Epoch/Return', {
            'R': np.array(returns).sum()
            }, global_step=self.ep_step
        )

    def _dataset(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(sum(self.num_samples)):
            yield obs_b[i], log_prob_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def _dataset_shuffle(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.hparams.epoch_repeat):
            idx = list(range(sum(self.num_samples)))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], log_prob_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def configure_optimizers(self):
        value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)   
        return value_opt, policy_opt

    # def optimizer_step(self, *args, **kwargs):
    #     """
    #     Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic
    #     for each data sample.
    #     """
    #     for i in range(self.hparams.nb_optim_iters):
    #         super().optimizer_step(*args, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceSourceDataset(self._dataset)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):

        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = batch

        # rev_nxt_obs = self.scaler.inverse_transform(T.clone(nxt_obs_b.data).detach().cpu().numpy())
        
        # state_values = self.value_net(T.hstack((obs_b, action_b)))
        state_values = self.value_net(obs_b)
       
        with T.no_grad():
            # _, nxt_action = self.target_policy(nxt_obs_b)
            # nxt_state_values = self.target_val_net(T.hstack((nxt_obs_b, nxt_action)))
            nxt_state_values = self.target_val_net(nxt_obs_b)
            nxt_state_values[done_b] = 0.0
            target = reward_b + self.hparams.gamma * nxt_state_values
        
        advantages = (target - state_values).detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if optimizer_idx == 0:
            loss = self.hparams.loss_fn(state_values.float(), target.float())
            self.ep_value_loss.append(loss.unsqueeze(0))
            self.tb_logger.add_scalars('Training/Value/Loss', {
                'loss': loss
                }, global_step=self.global_step
            )          
            return loss

        elif optimizer_idx == 1:
            log_prob, _ = self.policy(obs_b)
            prv_log_prob = log_prob_b
            
            rho = T.exp(log_prob - prv_log_prob)
            # rho = log_prob / prv_log_prob

            surrogate_1 = rho * advantages
            surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            policy_loss = - T.minimum(surrogate_1, surrogate_2)
            entropy = -T.sum(action_b*log_prob_b, dim=-1, keepdim=True)
            loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()
            self.tb_logger.add_scalars('Training/Policy/Loss', {
                'loss': policy_loss.mean()
                }, global_step=self.global_step
            )       
            self.tb_logger.add_scalars('Training/Policy/Entropy', {
                'entropy': entropy.mean()
                }, global_step=self.global_step
            )       
            return loss
                
    def training_epoch_end(self, training_step_outputs):
        self.target_val_net.load_state_dict(self.value_net.state_dict())            
        self.target_policy.load_state_dict(self.policy.state_dict())
        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/')
            # quietRun('rm -r outputs/logs/experiences.csv')
            # quietRun('rm -r outputs/logs/lightning_logs/version_0')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            # logger=self.tb_logger,
            # reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        T.autograd.detect_anomaly(True)
        self.play_episodes()
        trainer.fit(self) 

class PPO_MultiAgent(LightningModule):
    """ 
    """
    def __init__(self, env, num_envs=50, batch_size=2, hidden_size=256, samples_per_epoch=2,
                 epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, gamma=0.97, epsilon=0.3, entropy_coef=0.1,
                 loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.current_episode = 0
        self.env = env
        self.id = id
        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high
        self.obs_nds = None
        self.policies = {}
        self.values = {}
        self.targetvalues = {}
        
        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.save_hyperparameters('batch_size', 'policy_lr', 'value_lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')

    @T.no_grad()
    def play_episodes(self, policy=None):
        # self.buffer = []
        # self.env.num_samples = 0
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        obs, nodes = self.env.reset()
        for _ in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            _action = np.array([]).reshape(0,self.action_dims)
            _loc = np.array([]).reshape(0,self.action_dims)
            _scale = np.array([]).reshape(0,self.action_dims)
            for idx, val in np.ndenumerate(nodes):
                if self.policies.get(val):
                    loc, scale = self.policies[val](obs[idx[0]])
                else:
                    self.values[val] = PPO_ValueNet(self.obs_dims, self.hparams.hidden_size)
                    self.targetvalues[val] = copy.deepcopy(self.values[val])
                    self.policies[val] = PPO_GradientPolicy(self.obs_dims, self.hparams.hidden_size, self.action_dims)
                    loc, scale = self.policies[val](obs[idx[0]])
                action = T.normal(loc, scale)
                action = action.detach().cpu().numpy()
                loc = loc.detach().cpu().numpy()
                scale = scale.detach().cpu().numpy()
                _action = np.vstack((_action, action))
                _loc = np.vstack((_loc, loc))
                _scale = np.vstack((_scale, scale))
                # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(_action, obs, nodes))
            print(f'nodes size: {nodes.shape} type: {nodes.dtype}')
            print(f'obs size: {obs.shape} type: {obs.dtype}')
            print(f'_loc size: {_loc.shape} type: {_loc.dtype}')
            print(f'_scale size: {_scale.shape} type: {_scale.dtype}')
            print(f'_action size: {_action.shape} type: {_action.dtype}')
            print(f'reward size: {reward.shape} type: {reward.dtype}')
            print(f'done size: {done.shape} type: {done.dtype}')
            print(f'nxt_obs size: {nxt_obs.shape} type: {nxt_obs.dtype}')
            self.buffer.append((nodes.view(np.uint8), obs, _loc, _scale, _action, reward, done, nxt_obs))
            self.num_samples.append(nxt_obs.shape[0])
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(_action, columns=self.env.action_cols),
                pd.DataFrame(nxt_obs, columns=['nxt_obs'+str(i) for i in range(nxt_obs.shape[1])]), 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv(f'outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            obs = nxt_obs

    def configure_optimizers(self):
        value_parameters = itertools.chain(*[value.parameters() for value in self.values.values()])
        policy_parameters = itertools.chain(*[policy.parameters() for policy in self.policies.values()])
        value_opt = self.hparams.optim(value_parameters, lr=self.hparams.value_lr)
        policy_opt = self.hparams.optim(policy_parameters, lr=self.hparams.policy_lr)
        return value_opt, policy_opt

    def train_dataloader(self):
        dataset = RLDataset_PPO_shuffle(self.buffer, sum(self.num_samples), self.hparams.epoch_repeat)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):
        nodes_b, obs_b, loc_b, scale_b, action_b, reward_b, done_b, nxt_obs_b = batch
        with T.no_grad():
            _nodes_b = nodes_b.detach().cpu().numpy()
            _nodes = np.array([]).reshape(0,1)
            nds = []
            for i in range(nodes_b.shape[0]):
                _nodes = np.vstack((_nodes, np.array([''.join(c for c in _nodes_b[i].view('U1'))]).reshape(-1, 1)))
                nds.append(''.join(c for c in _nodes_b[i].view('U1')))
            nds = set(nds)
            # rev_nxt_obs_b = self.env.max_obs * nxt_obs_b.detach().cpu().numpy()
            rev_nxt_obs_b = nxt_obs_b.detach().cpu().numpy()
            # rev_nxt_obs_b = self.env.scaler.inverse_transform(nxt_obs_b.detach().cpu().numpy())
            self.log('episode/Return', reward_b.sum())
            self.log('episode/Performance/Delay', rev_nxt_obs_b[:,1].mean())
            self.log('episode/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
            self.log('episode/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
            self.log('episode/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
            self.log('episode/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,18].mean())
        
        self.current_episode += 1
        val_loss = 0
        pol_loss = 0
        entr = 0
        for nd in nds:
            idxs = np.where((_nodes[:,0] == nd))
            state_values = self.values[nd](obs_b)
            
            with T.no_grad():
                nxt_state_values = self.targetvalues[nd](nxt_obs_b)
                nxt_state_values[done_b] = 0.0
                target = reward_b + self.hparams.gamma * nxt_state_values

            if optimizer_idx == 0:
                loss = self.hparams.loss_fn(state_values.float(), target.float())
                val_loss += loss
                # self.ep_value_loss.append(loss.unsqueeze(0))

            elif optimizer_idx == 1:
                advantages = (target - state_values).detach()
                prv_dist = Normal(loc_b[idxs[0]], scale_b[idxs[0]])
                prv_log_prob = prv_dist.log_prob(action_b[idxs[0]]).sum(dim=-1, keepdim=True)
                new_loc, new_scale = self.policies[nd](obs_b[idxs[0]])
                dist = Normal(new_loc, new_scale)
                log_prob = dist.log_prob(action_b[idxs[0]]).sum(dim=-1, keepdim=True)

                rho = T.exp(log_prob - prv_log_prob)

                surrogate_1 = rho * advantages[idxs[0]]
                surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages[idxs[0]]

                policy_loss = - T.minimum(surrogate_1, surrogate_2)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
                loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()
                # loss = Variable(loss, requires_grad = True)
                pol_loss += loss
                entr += entropy.mean()
                # self.ep_policy_loss.append(loss.unsqueeze(0))
                # self.ep_entropy.append(entropy.mean().unsqueeze(0))
            
        if optimizer_idx == 0:
            self.log('episode/ValueNet/Loss', val_loss)    
            return val_loss
        
        if optimizer_idx == 1:
            self.log('episode/Policy/Loss', pol_loss)
            self.log('episode/Policy/Entropy', entr)
            return pol_loss
        
    # def backward(self, loss):
    #     loss.backward(retain_graph=True)

    def training_epoch_end(self, training_step_outputs):
        for idx, value_net in self.values.items():
            self.targetvalues[idx].load_state_dict(value_net.state_dict())

        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        dataset = map(np.vstack, zip(*self.buffer))
        nodes_b, obs_b, loc_b, scale_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, dataset)

        # rev_nxt_obs_b = self.env.max_obs *nxt_obs_b
        rev_nxt_obs_b = nxt_obs_b
        # rev_nxt_obs_b = self.env.scaler.inverse_transform(nxt_obs_b)
        # self.log('epoch/ValueNet/Loss', T.cat(self.ep_value_loss).mean()) 
        # self.log('epoch/Policy/Loss', T.cat(self.ep_policy_loss).mean())
        # self.log('epoch/Policy/Entropy', T.cat(self.ep_entropy).mean())
        self.log('epoch/Return', reward_b.sum())
        self.log('epoch/Performance/Delay', rev_nxt_obs_b[:,1].mean())
        self.log('epoch/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
        self.log('epoch/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
        self.log('epoch/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
        self.log('epoch/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,18].mean())
        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        try:
            quietRun(f'rm -r outputs/logs/{self.id}_experiences.csv')
            quietRun('rm -r outputs/logs/lightning_logs/')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        tb_logger = TensorBoardLogger(save_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            logger=tb_logger,
            reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        
        self.play_episodes()
        trainer.fit(self) 

class PPO_GAE_Agent(LightningModule):
    """ 
    """
    def __init__(self, env, num_envs=50, batch_size=2, hidden_size=256, samples_per_epoch=2,
                 epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, gamma=0.97, lamb=0.95, epsilon=0.3, entropy_coef=0.1,
                 loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.current_episode = 0
        self.env = env

        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high
        self.obs_nds = None
        self.policy = PPO_GradientPolicy(self.obs_dims, hidden_size, self.action_dims)
        self.value_net = PPO_ValueNet(self.obs_dims, hidden_size)

        self.target_val_net = copy.deepcopy(self.value_net) 

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.save_hyperparameters('batch_size', 'policy_lr', 'value_lr', 
            'hidden_size', 'gamma', 'lamb', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')

    @T.no_grad()
    def play_episodes(self, policy=None):
        # self.buffer = []
        # self.env.num_samples = 0
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        obs, nodes = self.env.reset()
        # obs = self.env.scaler.fit_transform(obs)
        transitions = []
        for _ in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            loc, scale = self.policy(obs)
            action = T.normal(loc, scale)
            action = action.detach().cpu().numpy()
            loc = loc.detach().cpu().numpy()
            scale = scale.detach().cpu().numpy()
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nodes))
            # nxt_obs = self.env.scaler.transform(nxt_obs)
            transitions.append((obs, loc, scale, action, reward, done, nxt_obs))
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(action, columns=self.env.action_cols),
                pd.DataFrame(nxt_obs, columns=['nxt_obs'+str(i) for i in range(nxt_obs.shape[1])]), 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            obs = nxt_obs
       
        print(obs)
        running_gae = np.zeros((transitions[-1][-1].shape[0], 1), dtype=np.float32)
        gae_b = np.array([], dtype=np.float32).reshape(0,1)
        target_b = np.array([], dtype=np.float32).reshape(0,1)
        for row in range(self.hparams.samples_per_epoch - 1, -1, -1):
            obs_b = transitions[row][0]
            nxt_obs_b = transitions[row][-1]
            done_b = transitions[row][-2]
            reward_b = transitions[row][-3]
            value_b = self.value_net(obs_b)

            nxt_value_b = self.value_net(nxt_obs_b)
            td_error_b = reward_b + (1-done_b.astype(int)) * self.hparams.gamma * nxt_value_b.detach().cpu().numpy() - value_b.detach().cpu().numpy()
            running_gae = td_error_b + (1- done.astype(int)) * self.hparams.gamma * self.hparams.lamb * running_gae
            gae_b = np.vstack((gae_b, running_gae))        
            target_b = np.vstack((target_b, running_gae + value_b.detach().cpu().numpy()))

        transitions = map(np.vstack, zip(*transitions))
        obs_b, loc_b, scale_b, action_b, reward_b, done_b, nxt_obs_b = transitions
        self.buffer.append((obs_b, loc_b, scale_b, action_b, reward_b, gae_b, target_b, done_b, nxt_obs_b))
        self.num_samples.append(nxt_obs_b.shape[0])

    def configure_optimizers(self):
        value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)
        return value_opt, policy_opt

    def train_dataloader(self):
        dataset = RLDataset_GAE_shuffle(self.buffer, sum(self.num_samples), self.hparams.epoch_repeat)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):
        obs_b, loc_b, scale_b, action_b, reward_b, gae_b, target_b, done_b, nxt_obs_b = batch
        
        # rev_nxt_obs_b = self.env.max_obs * nxt_obs_b.detach().cpu().numpy()
        rev_nxt_obs_b = nxt_obs_b.detach().cpu().numpy()
        # rev_nxt_obs_b = self.env.scaler.inverse_transform(nxt_obs_b.detach().cpu().numpy())
        self.log('episode/Return', reward_b.sum())
        self.log('episode/Performance/Delay', rev_nxt_obs_b[:,1].mean())
        self.log('episode/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
        self.log('episode/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
        self.log('episode/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
        self.log('episode/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,18].mean())

        state_values = self.value_net(obs_b)
        
        # with T.no_grad():
        #     nxt_state_values = self.target_val_net(nxt_obs_b)
        #     nxt_state_values[done_b] = 0.0
        #     target = reward_b + self.hparams.gamma * nxt_state_values
        
        # self.log('episode/Reward', T.tensor(np.mean(list(self.env.reward_history)), device=device))
        # self.log('episode/Performance/Delay', T.tensor(np.mean(list(self.env.delay_history)), device=device))
        # self.log('episode/Performance/Throughput', T.tensor(np.mean(list(self.env.throughput_history)), device=device))
        # self.log('episode/Performance/Enrgy_Consumption', T.tensor(np.mean(list(self.env.energycons_history)), device=device))
        # self.log('episode/Performance/TxPackets', T.tensor(np.mean(list(self.env.txpackets_history)), device=device))
        # self.log('episode/Performance/TxBytes', T.tensor(np.mean(list(self.env.txbytes_history)), device=device))
        # self.log('episode/Performance/RxPackets', T.tensor(np.mean(list(self.env.rxpackets_history)), device=device))
        # self.log('episode/Performance/RxBytes', T.tensor(np.mean(list(self.env.rxbytes_history)), device=device))
        # self.log('episode/Performance/TxPacketsIn', T.tensor(np.mean(list(self.env.txpacketsin_history)), device=device))
        # self.log('episode/Performance/TxBytesIn', T.tensor(np.mean(list(self.env.txbytesin_history)), device=device))
        # self.log('episode/Performance/RxPacketsOut', T.tensor(np.mean(list(self.env.rxpacketsout_history)), device=device))
        # self.log('episode/Performance/RxBytesOut', T.tensor(np.mean(list(self.env.rxbytesout_history)), device=device))
        # self.log('episode/Performance/Dropped_Packets', T.tensor(np.mean(list(self.env.drpackets_history)), device=device))
        # self.log('episode/Performance/Resedual_Energy_var', T.tensor(np.mean(list(self.env.energyvar_history)), device=device))
        self.current_episode += 1

        if optimizer_idx == 0:
            loss = self.hparams.loss_fn(state_values.float(), target_b.float())
            self.ep_value_loss.append(loss.unsqueeze(0))
            self.log('episode/ValueNet/Loss', loss)           
            return loss

        elif optimizer_idx == 1:
            # advantages = (target_b - state_values).detach()
            new_loc, new_scale = self.policy(obs_b)
            dist = Normal(new_loc, new_scale)
            log_prob = dist.log_prob(action_b).sum(dim=-1, keepdim=True)

            prv_dist = Normal(loc_b, scale_b)
            prv_log_prob = prv_dist.log_prob(action_b).sum(dim=-1, keepdim=True)
            
            rho = T.exp(log_prob - prv_log_prob)

            surrogate_1 = rho * gae_b
            surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * gae_b

            policy_loss = - T.minimum(surrogate_1, surrogate_2)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()
            
            self.ep_policy_loss.append(loss.unsqueeze(0))
            self.ep_entropy.append(entropy.mean().unsqueeze(0))
            self.log('episode/Policy/Loss', loss)
            self.log('episode/Policy/Entropy', entropy.mean())
            self.log('episode/Policy/Reward', reward_b.mean())
            return loss

    def training_epoch_end(self, training_step_outputs):
        self.target_val_net.load_state_dict(self.value_net.state_dict())        

        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        dataset = map(np.vstack, zip(*self.buffer))
        obs_b, loc_b, scale_b, action_b, reward_b, gae_b, target_b, done_b, nxt_obs_b = map(reshape_fn, dataset)

        # rev_nxt_obs_b = self.env.max_obs *nxt_obs_b
        rev_nxt_obs_b = nxt_obs_b
        # rev_nxt_obs_b = self.env.scaler.inverse_transform(nxt_obs_b)
        self.log('epoch/ValueNet/Loss', T.cat(self.ep_value_loss).mean()) 
        self.log('epoch/Policy/Loss', T.cat(self.ep_policy_loss).mean())
        self.log('epoch/Policy/Entropy', T.cat(self.ep_entropy).mean())
        self.log('epoch/Return', reward_b.sum())
        self.log('epoch/Performance/Delay', rev_nxt_obs_b[:,1].mean())
        self.log('epoch/Performance/Throughput', rev_nxt_obs_b[:,2].mean())
        self.log('epoch/Performance/Enrgy_Consumption', rev_nxt_obs_b[:,3].mean())
        self.log('epoch/Performance/Dropped_Packets', rev_nxt_obs_b[:,10].mean())
        self.log('epoch/Performance/Resedual_Energy_var', rev_nxt_obs_b[:,18].mean())
        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/experiences.csv')
            quietRun('rm -r outputs/logs/lightning_logs/')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        tb_logger = TensorBoardLogger(save_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            logger=tb_logger,
            reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        
        self.play_episodes()
        trainer.fit(self) 

class LSTM_Agent(LightningModule):
    """LSTM Regression Model"""
    def __init__(self, env, hidden_dim=250, num_layers=1, seq_len=10, batch_size=1, test_size=0.10, lr=0.001, dropout=0.5, loss_fn=F.mse_loss, optim=Adam):
        
        super().__init__()  
        self.loop = asyncio.get_running_loop()
        self.epoch_cntr = 1
        self.env = env

        self.input_dim = self.env.observation_space.shape[1]
        self.output_dim = self.input_dim

        # self.buffer = TSReplayBuffer(500, seq_len)
        self.buffer = deque(maxlen=500)

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None

        self.model  = LSTM(input_dim=self.input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, output_dim=self.output_dim)        

        self.save_hyperparameters('hidden_dim', 'num_layers', 'seq_len', 'batch_size', 'test_size', 'lr', 'dropout', 'loss_fn', 'optim')

    #test the env
    @T.no_grad()
    async def play_episodes(self):
        ts = 0   
        while True:
            self.buffer.append(self.env.getObs())
            await asyncio.sleep(1)
            # pd.concat([pd.DataFrame(nds, columns=['node']),
            #         pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
            #         pd.DataFrame(action, columns=self.env.action_cols),
            #         pd.DataFrame(next_obs, columns=['nxt_obs'+str(i) for i in range(next_obs.shape[1])]), 
            #         pd.DataFrame(reward, columns=['reward']),
            #         pd.DataFrame(done, columns=['done']),
            #         pd.DataFrame(info, columns=['info'])],
            #         axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            # self.epoch_cntr += 1
                # nsamples += 1
            # return obs

    @T.no_grad()
    def datasetgen(self):  
        # if stage == 'fit' and self.X_train is not None:
        #     return 
        # if stage == 'test' and self.X_test is not None:
        #     return
        # if stage is None and self.X_train is not None and self.X_test is not None:  
        #     return  
        data = np.array(list(itertools.islice(self.buffer, 0, len(self.buffer)))).squeeze()
        data = self.env.scaler.fit_transform(data)
        X = data[0:len(data)-1]
        y = data[1:len(data)]
        # X = np.array(list(itertools.islice(self.buffer, 0, len(self.buffer) - 1))).squeeze()
        # y = np.array(list(itertools.islice(self.buffer, 1, len(self.buffer)))).squeeze()
        # X_cv, self.X_test, y_cv, self.y_test = train_test_split(
        #     X, y, test_size=0.2, shuffle=False
        # )
    
        # self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
        #     X_cv, y_cv, test_size=0.25, shuffle=False
        # )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.hparams.test_size, shuffle=False
        )
    def configure_optimizers(self):
        optimizer = self.hparams.optim(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self):
        train_dataset = TimeseriesDataset(self.X_train, 
            self.y_train, 
            seq_len=self.hparams.seq_len)
        # train_dataset = TSDataset(self.buffer, self.hparams.batch_size)
        train_loader = DataLoader(train_dataset, 
            batch_size = self.hparams.batch_size, 
            shuffle = False, 
            num_workers = cpu_count())
        
        return train_loader

    def val_dataloader(self):
        val_dataset = TimeseriesDataset(self.X_val, 
            self.y_val, 
            seq_len=self.hparams.seq_len)

        # val_dataset = TSTestDataset(self.buffer, self.hparams.batch_size)
        val_loader = DataLoader(val_dataset, 
            batch_size = self.hparams.batch_size, 
            shuffle = False, 
            num_workers = cpu_count())

        return val_loader

    # def test_dataloader(self):
    #     test_dataset = TimeseriesDataset(self.X_test, 
    #         self.y_test, 
    #         seq_len=self.hparams.seq_len)
    #     test_loader = DataLoader(test_dataset, 
    #         batch_size = self.hparams.batch_size, 
    #         shuffle = False, 
    #         num_workers = cpu_count())

    #     return test_loader
    
    def training_step(self, batch, batch_idx):
        # inputs is batch of experiences exp -> {obs, actions, rewards, done, next_obs}
        # action_values are the result of trained DQN model when applying actions to states (obs)
        x, y = batch
        print(f'train_x:{x.shape}\n{x}')
        print(f'train_y:{y.shape}\n{y}')

        outputs = self.model(x)
        print(f'train_outpus:{outputs.shape}\n{outputs}')
        batch_dictionary = {}
        # batch_dictionary['delay'] = next_obs[:,:,1].mean()
        # batch_dictionary['goodput'] = next_obs[:,:,2].mean()
        # batch_dictionary['engcons'] = next_obs[:,:,3].mean()
        
        # batch_dictionary['reward'] = reward.mean()
        # self.log('episode/Performance/Delay', next_obs[:,:,1].mean())
        # self.log('episode/Performance/Throughput', next_obs[:,:,2].mean())
        # self.log('episode/Performance/Enrgy_Consumption', next_obs[:,:,3].mean())
        # self.log('episode/Performance/Txpackets', next_obs[:,:,6].mean())
        # self.log('episode/Performance/Txbytes', next_obs[:,:,7].mean())
        # self.log('episode/Performance/Rxpackets', next_obs[:,:,8].mean())
        # self.log('episode/Performance/Rxbytes', next_obs[:,:,9].mean())
        # self.log('episode/Performance/Dropped_Packets', next_obs[:,:,10].mean())
        # self.log('episode/Performance/Resedual_Energy_std', next_obs[:,:,13].mean())

        loss = self.hparams.loss_fn(outputs, y)
        # self.log('episode/Q-Loss', q_loss_total)
        # self.log('episode/Reward', rewards.sum(0).flatten())
        # batch_dictionary={
        # "opt": optimizer_idx,
        # "loss": q_loss
        # }
        batch_dictionary['loss'] = loss
        self.log('episode/LSTM/Loss', loss)
        data_pred = self.env.scaler.inverse_transform(outputs.detach().cpu().numpy())
        data_y = self.env.scaler.inverse_transform(y.detach().cpu().numpy())
        for i in range(data_pred.shape[0]):
            self.writer.add_scalars('episode/LSTM/training', 
                                            {'data_pred': data_pred[i,0], 
                                                'data_y': data_y[i,0]}, 
                                            global_step=self.global_step)
        # self.log('episode/Q-Loss', q_loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("episode/Q-performance", {"acc": acc, "recall": recall})
        # return q_loss_total
        return batch_dictionary
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        print(f'val_x:{x.shape}\n{x}')
        print(f'val_y:{y.shape}\n{y}')
        outputs = self.model(x)
        print(f'val_outpus:{outputs.shape}\n{outputs}')

        loss = self.hparams.loss_fn(outputs, y)
        # self.log('episode/Q-Loss', q_loss_total)
        # self.log('episode/Reward', rewards.sum(0).flatten())
        # batch_dictionary={
        # "opt": optimizer_idx,
        # "loss": q_loss
        # }
        self.log('episode/LSTM/val_Loss', loss)
        data_pred = self.env.scaler.inverse_transform(outputs.detach().cpu().numpy())
        data_y = self.env.scaler.inverse_transform(y.detach().cpu().numpy())
        for i in range(data_pred.shape[0]):
            self.writer.add_scalars('episode/LSTM/vald', 
                                            {'data_pred': data_pred[i,0], 
                                                'data_y': data_y[i,0]}, 
                                            global_step=self.global_step)
        # self.log('episode/Q-Loss', q_loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("episode/Q-performance", {"acc": acc, "recall": recall})
        # return q_loss_total

    # def test_step(self, batch, batch_idx):
    #     x, y = batch

    #     print(f'test_x:{x.shape}\n{x}')
    #     print(f'test_y:{y.shape}\n{y}')
    #     outputs = self.lstm(x)
    #     print(f'test_outpus:{outputs.shape}\n{outputs}')

    #     loss = self.hparams.loss_fn(outputs, y)        
    #     print(f'test_loss:{loss}')
    #     # self.log('episode/Q-Loss', q_loss_total)
    #     # self.log('episode/Reward', rewards.sum(0).flatten())
    #     # batch_dictionary={
    #     # "opt": optimizer_idx,
    #     # "loss": q_loss
    #     # }
    #     self.log('episode/LSTM/test_Loss', loss)
    #     # self.log('episode/Q-Loss', q_loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     # self.log("episode/Q-performance", {"acc": acc, "recall": recall})
    #     # return q_loss_total

    def training_epoch_end(self, training_step_outputs):
        self.datasetgen()
        print(f'x_train: {self.X_train.shape} y_train: {self.y_train.shape} x_val: {self.X_val.shape} y_val: {self.y_val.shape}')
        # obs = obs.flatten()
        #run async function as sync
        # self.loop.run_until_complete(self.play_episodes(policy=self.policy))
        # polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
        # polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        # polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)
        if self.current_epoch > 0 and self.current_epoch % 5 == 0:
            self.model.load_state_dict(self.model.state_dict())
        
        print(training_step_outputs)
        # if self.current_epoch % 100 == 0:
        #     print("Epoch: %d, loss: %1.5f" % (self.current_epoch, training_step_outputs['loss'].item()))

        # self.log('episode_return', self.env.return_queue[-1])
        # calculating average loss  

        # if isinstance(training_step_outputs, dict):

        # print(training_step_outputs)
        # obs, _ = self.env.getObs()
        # q_loss = []
        # p_loss = []
        # reward = []
        # entropy = []
        # for output in training_step_outputs:
        #     for item in output:
        #         if item['opt'] == 1:
        #             p_loss.append(item['loss'])
        #             entropy.append(item['entropy'])
        #             # p_mse.append(item['mse'])
        #         else:
        #             q_loss.append(item['loss'])
        #         reward.append(item['reward'])
        # self.log('Policy/Loss', T.stack(p_loss).mean())
        # self.log('Q/Loss', T.stack(q_loss).mean())
        # self.log('Policy/Entropy', T.stack(entropy).mean())
        # self.log('Episode/Reward', T.stack(reward).mean())
        # self.log('Performance/Delay', obs[:,1].mean())
        # self.log('Performance/Goodput', obs[:,2].mean())
        # self.log('Performance/Enrgy_Consumption', obs[:,3].mean())
        # self.log('Policy/Loss', T.stack(p_loss).mean())
        # self.log('Policy/Entropy', T.stack(entropy).mean())
        # self.log('Q/Loss', T.stack(q_loss).mean())
        # self.log('Policy/MSE', T.stack(p_mse).mean())
        # correct=sum([x[0]["correct"] for  x in training_step_outputs])
        # total=sum([x[0]["total"] for  x in training_step_outputs])          

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/experiences.csv')
            quietRun('rm -r outputs/logs/lightning_logs/')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        tb_logger = TensorBoardLogger(save_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            logger=tb_logger,
            reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        
        # while len(self.buffer) < self.hparams.samples_per_epoch:
        #     await asyncio.gather(self.loop.run_in_executor(None, self.play_episodes))
        # await asyncio.gather(self.loop.run_in_executor(None, trainer.fit, self))

        self.loop.create_task(self.play_episodes())
        while len(self.buffer) * self.hparams.test_size < self.hparams.batch_size * self.hparams.seq_len:
        # while len(self.buffer.buffer) * 0.10 < self.hparams.seq_len:
            logger.info(f'{len(self.buffer)} samples in sequance buffer. Filling...')            
            await asyncio.sleep(2)
        self.datasetgen()
        trainer.fit(self)
        self.writer.close()


class TFT_Agent(LightningModule):
    """TemporalFusionTransformer Regression Model"""
    def __init__(self, env,  hidden_size=256, samples_per_epoch=2, num_layers=1, seq_len=10, batch_size=1, test_size=0.10, policy_lr=1e-4, value_lr=1e-3, dropout=0.5, loss_fn=SMAPE(), optim=AdamW):
        
        super().__init__()  
        self.loop = asyncio.get_running_loop()
        self.epoch_cntr = 1
        self.env = env

        self.input_dim = self.env.observation_space.shape[1]
        self.output_dim = self.input_dim

        # self.buffer = TSReplayBuffer(500, seq_len)
        self.policy = None
        # self.target_policy = copy.deepcopy(self.policy)
        # self.value_net = PPO_ValueNet(self.input_dim+self.output_dim, hidden_size)
        # self.target_val_net = copy.deepcopy(self.value_net) 
        
        self.training_dataset = None
        self.valid_dataset = None
        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.ep_step = 0
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None

        self.model  = None        

        self.save_hyperparameters('hidden_size', 'samples_per_epoch', 'num_layers', 'seq_len', 'batch_size', 'test_size', 'policy_lr', 'value_lr', 'dropout', 'loss_fn', 'optim', ignore=['loss', 'logging_metrics'])


    @T.no_grad()
    def play_episodes(self, policy=None):
        # self.buffer = []
        # self.env.num_samples = 0
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        delay = []
        throughput = []
        engcons = []
        droppackets = []
        resenergy = []
        returns = [0]
        

        obs, nodes = self.env.reset()
        train = obs.iloc[:len(obs)]
        train.drop_duplicates(ignore_index=True)
        train.rename(columns = {'ts':'time_idx'}, inplace = True)
        # train.drop(columns=train.columns[0], axis=1, inplace=True)
        train = train.astype({'time_idx':'int', 'id':'str', 'port':'float', 'intftypeval':'float', 'datatypeval':'float', 'distance':'float', \
            'denisty':'float', 'alinks':'float', 'flinks':'float', 'x':'float', 'y':'float', 'z':'float', 'batt':'float', 'delay':'float', \
            'throughput':'float', 'engcons':'float', 'txpackets_val':'float', 'txbytes_val':'float', 'rxpackets_val':'float', 'rxbytes_val':'float', \
            'drpackets_val':'float', 'txpacketsin_val':'float', 'txbytesin_val':'float', 'rxpacketsout_val':'float', 'rxbytesout_val':'float'})
        max_prediction_length = self.env.prd_time
        max_encoder_length = train.env.obs_time
        training_cutoff = train["time_idx"].max() - max_prediction_length
        # print(obs)
        # print(train)
        # print(train['time_idx'].max())
        # print(train['time_idx'].dtype.kind)
        # print(train.dtypes)
        self.training_dataset = TimeSeriesDataSet(
            train[lambda x: x.time_idx <= training_cutoff],
            time_idx='time_idx',
            # target=['batt', 'delay', 'throughput', 'engcons', 'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val'],
            target='batt',
            group_ids=['id', 'port', 'intftypeval', 'datatypeval'],
            min_encoder_length=max_encoder_length//2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=['id'],
            static_reals=['port', 'intftypeval', 'datatypeval'],
            time_varying_known_categoricals=[],
            time_varying_unknown_categoricals=[],
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_reals=['distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', 'engcons', 'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val'],
            # target_normalizer=MultiNormalizer([EncoderNormalizer(), TorchNormalizer()]),
            target_normalizer=GroupNormalizer(
                groups=['id', 'port', 'intftypeval', 'datatypeval'], transformation="softplus"
            ),
            categorical_encoders={
                'port':NaNLabelEncoder(add_nan=True), 
                'intftypeval':NaNLabelEncoder(add_nan=True), 
                'datatypeval':NaNLabelEncoder(add_nan=True)
            },
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )        
        self.train_dataloader()
        self.valid_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, train, predict=True, stop_randomization=True)
        action = self.trainer.predict(self.policy, self.val_dataloader())
        action = action.detach().cpu().numpy()
        nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nodes))
        '''
        print('COLLECTING DATA ....')     
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            log_prob, action = self.policy(obs)
            action = action.detach().cpu().numpy()
            log_prob = log_prob.detach().cpu().numpy()

            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nodes))
            self.buffer.append((obs, log_prob, action, reward, done, nxt_obs))
            self.num_samples.append(nxt_obs.shape[0])
            returns.append(reward.sum())
            rev_nxt_obs = self.env.scaler.inverse_transform(nxt_obs)
            # rev_nxt_obs = np.log(np.divide(nxt_obs,1-nxt_obs))
            # rev_nxt_obs = nxt_obs
            delay.append(rev_nxt_obs[:,0].mean())
            throughput.append(rev_nxt_obs[:,1].mean())
            engcons.append(rev_nxt_obs[:,2].mean())
            droppackets.append(rev_nxt_obs[:,9].mean())
            pd.concat([pd.DataFrame(nodes, columns=['node']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(action, columns=self.env.action_cols),
                pd.DataFrame(nxt_obs, columns=['nxt_obs'+str(i) for i in range(nxt_obs.shape[1])]), 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            obs = nxt_obs

            self.tb_logger.add_scalars('Episode/Return', {
                'R': reward.sum()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Delay', {
                'DE': rev_nxt_obs[:,0].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Throughput', {
                'TH': rev_nxt_obs[:,1].mean() 
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Energy_Consumption', {
                'EC': rev_nxt_obs[:,2].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Dropped_Packets', {
                'DP': rev_nxt_obs[:,9].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Report_TTI', {
                'RTTI': rev_nxt_obs[:,-1].mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Energy_Consumption_Var', {
                'DCVar': np.sqrt(np.square((rev_nxt_obs[:,2])-np.mean(rev_nxt_obs[:,2]))).mean()
                }, global_step=self.ep_step
            )
            self.tb_logger.add_scalars('Episode/Report_TTI_Var', {
                'RTTIVar': np.sqrt(np.square((rev_nxt_obs[:,-1])-np.mean(rev_nxt_obs[:,-1]))).mean()
                }, global_step=self.ep_step
            )

            self.ep_step += 1
        '''    

    def _dataset(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(sum(self.num_samples)):
            yield obs_b[i], log_prob_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def configure_optimizers(self):
        # value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)   
        return policy_opt

    # def optimizer_step(self, *args, **kwargs):
    #     """
    #     Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic
    #     for each data sample.
    #     """
    #     for i in range(self.hparams.nb_optim_iters):
    #         super().optimizer_step(*args, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        # dataset = ExperienceSourceDataset(self._dataset)   
        dataloader = self.training_dataset.to_dataloader(train=True, batch_size=self.hparams.batch_size, num_workers=cpu_count())     
        # dataloader = DataLoader(
        #     dataset=self.training_dataset, 
        #     batch_size=self.hparams.batch_size,
        #     num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        # )
        return dataloader
    
    def val_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        # dataset = ExperienceSourceDataset(self._dataset)
        dataloader = self.valid_dataset.to_dataloader(train=False, batch_size=self.hparams.batch_size*10, num_workers=cpu_count())     
        # dataloader = DataLoader(
        #     dataset=self.valid_dataset, 
        #     batch_size=self.hparams.batch_size*10,
        #     num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        # )
        return dataloader
    

    # def test_dataloader(self):
    #     test_dataset = TimeseriesDataset(self.X_test, 
    #         self.y_test, 
    #         seq_len=self.hparams.seq_len)
    #     test_loader = DataLoader(test_dataset, 
    #         batch_size = self.hparams.batch_size, 
    #         shuffle = False, 
    #         num_workers = cpu_count())

    #     return test_loader
    
    def training_step(self, batch, batch_idx):
        # inputs is batch of experiences exp -> {obs, actions, rewards, done, next_obs}
        # action_values are the result of trained DQN model when applying actions to states (obs)
        x, y = batch
        # print(f'train_y:\n{y}')

        outputs = self.policy(x)
        # print(f'train_y:\n{y}')
        # print(f'train_y_pred:\n{outputs[0]}')
        batch_dictionary = {}
        # batch_dictionary['delay'] = next_obs[:,:,1].mean()
        # batch_dictionary['goodput'] = next_obs[:,:,2].mean()
        # batch_dictionary['engcons'] = next_obs[:,:,3].mean()
        
        # batch_dictionary['reward'] = reward.mean()
        # self.log('episode/Performance/Delay', next_obs[:,:,1].mean())
        # self.log('episode/Performance/Throughput', next_obs[:,:,2].mean())
        # self.log('episode/Performance/Enrgy_Consumption', next_obs[:,:,3].mean())
        # self.log('episode/Performance/Txpackets', next_obs[:,:,6].mean())
        # self.log('episode/Performance/Txbytes', next_obs[:,:,7].mean())
        # self.log('episode/Performance/Rxpackets', next_obs[:,:,8].mean())
        # self.log('episode/Performance/Rxbytes', next_obs[:,:,9].mean())
        # self.log('episode/Performance/Dropped_Packets', next_obs[:,:,10].mean())
        # self.log('episode/Performance/Resedual_Energy_std', next_obs[:,:,13].mean())

        loss = self.hparams.loss_fn.loss(outputs[0], y[0])
        # print(f'LOSS:\n {loss.mean()}')
        self.tb_logger.add_scalars('Training/Loss', {
            'loss': loss.mean()
            }, global_step=self.global_step
        )     
        # self.log('episode/Q-Loss', q_loss_total)
        # self.log('episode/Reward', rewards.sum(0).flatten())
        # batch_dictionary={
        # "opt": optimizer_idx,
        # "loss": q_loss
        # }
        # batch_dictionary['loss'] = loss
        # self.log('episode/LSTM/Loss', loss)
        # data_pred = self.env.scaler.inverse_transform(outputs.detach().cpu().numpy())
        # data_y = self.env.scaler.inverse_transform(y.detach().cpu().numpy())
        # for i in range(data_pred.shape[0]):
        #     self.writer.add_scalars('episode/LSTM/training', 
        #                                     {'data_pred': data_pred[i,0], 
        #                                         'data_y': data_y[i,0]}, 
        #                                     global_step=self.global_step)
        # self.log('episode/Q-Loss', q_loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("episode/Q-performance", {"acc": acc, "recall": recall})
        # return q_loss_total
        return loss.mean()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        # print(f'val_x:\n{x}')
        # print(f'val_y:\n{y}')
        outputs = self.policy(x)
        # print(f'valid_y:\n{y}')
        # print(f'valid_y_pred:\n{outputs[0]}')
        loss = self.hparams.loss_fn.loss(outputs[0], y[0])
        print(f'LOSS:\n {loss.mean()}')
        self.tb_logger.add_scalars('Validation/Loss', {
            'val_loss': loss.mean()
            }, global_step=self.global_step
        )   
        return loss.mean()       
        # self.log('episode/Q-Loss', q_loss_total)
        # self.log('episode/Reward', rewards.sum(0).flatten())
        # batch_dictionary={
        # "opt": optimizer_idx,
        # "loss": q_loss
        # }
        # self.log('episode/LSTM/val_Loss', loss)
        # data_pred = self.env.scaler.inverse_transform(outputs.detach().cpu().numpy())
        # data_y = self.env.scaler.inverse_transform(y.detach().cpu().numpy())
        # for i in range(data_pred.shape[0]):
        #     self.writer.add_scalars('episode/LSTM/vald', 
        #                                     {'data_pred': data_pred[i,0], 
        #                                         'data_y': data_y[i,0]}, 
        #                                     global_step=self.global_step)
        # self.log('episode/Q-Loss', q_loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("episode/Q-performance", {"acc": acc, "recall": recall})
        # return q_loss_total

    # def test_step(self, batch, batch_idx):
    #     x, y = batch

    #     print(f'test_x:{x.shape}\n{x}')
    #     print(f'test_y:{y.shape}\n{y}')
    #     outputs = self.lstm(x)
    #     print(f'test_outpus:{outputs.shape}\n{outputs}')

    #     loss = self.hparams.loss_fn(outputs, y)        
    #     print(f'test_loss:{loss}')
    #     # self.log('episode/Q-Loss', q_loss_total)
    #     # self.log('episode/Reward', rewards.sum(0).flatten())
    #     # batch_dictionary={
    #     # "opt": optimizer_idx,
    #     # "loss": q_loss
    #     # }
    #     self.log('episode/LSTM/test_Loss', loss)
    #     # self.log('episode/Q-Loss', q_loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     # self.log("episode/Q-performance", {"acc": acc, "recall": recall})
    #     # return q_loss_total

    def training_epoch_end(self, training_step_outputs):
        # obs = obs.flatten()
        #run async function as sync
        # self.loop.run_until_complete(self.play_episodes(policy=self.policy))
        # polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
        # polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        # polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)
        if self.current_epoch > 0 and self.current_epoch % 5 == 0:
            self.policy.load_state_dict(self.policy.state_dict())
        # best_model_path = self.trainer.checkpoint_callback.best_model_path
        # best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        # best_tft.predict()
        
        print('END EPOCH.........')
        self.play_episodes()
        print(training_step_outputs)
        # if self.current_epoch % 100 == 0:
        #     print("Epoch: %d, loss: %1.5f" % (self.current_epoch, training_step_outputs['loss'].item()))

        # self.log('episode_return', self.env.return_queue[-1])
        # calculating average loss  

        # if isinstance(training_step_outputs, dict):

        # print(training_step_outputs)
        # obs, _ = self.env.getObs()
        # q_loss = []
        # p_loss = []
        # reward = []
        # entropy = []
        # for output in training_step_outputs:
        #     for item in output:
        #         if item['opt'] == 1:
        #             p_loss.append(item['loss'])
        #             entropy.append(item['entropy'])
        #             # p_mse.append(item['mse'])
        #         else:
        #             q_loss.append(item['loss'])
        #         reward.append(item['reward'])
        # self.log('Policy/Loss', T.stack(p_loss).mean())
        # self.log('Q/Loss', T.stack(q_loss).mean())
        # self.log('Policy/Entropy', T.stack(entropy).mean())
        # self.log('Episode/Reward', T.stack(reward).mean())
        # self.log('Performance/Delay', obs[:,1].mean())
        # self.log('Performance/Goodput', obs[:,2].mean())
        # self.log('Performance/Enrgy_Consumption', obs[:,3].mean())
        # self.log('Policy/Loss', T.stack(p_loss).mean())
        # self.log('Policy/Entropy', T.stack(entropy).mean())
        # self.log('Q/Loss', T.stack(q_loss).mean())
        # self.log('Policy/MSE', T.stack(p_mse).mean())
        # correct=sum([x[0]["correct"] for  x in training_step_outputs])
        # total=sum([x[0]["total"] for  x in training_step_outputs])          

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/')
            # quietRun('rm -r outputs/logs/experiences.csv')
            # quietRun('rm -r outputs/logs/lightning_logs/version_0')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        self.trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            limit_train_batches=10,
            # callbacks=[self.tb_logger, early_stop_callback],
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            gradient_clip_val=0.1,
            # logger=self.tb_logger,
            # reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )        
        T.autograd.detect_anomaly(True)
        # configure the TFT model and timeseries dataset
        obs, nodes = self.env.reset()
        train = obs.iloc[:len(obs)]
        train.drop_duplicates(ignore_index=True)
        train.rename(columns = {'ts':'time_idx'}, inplace = True)
        # train.drop(columns=train.columns[0], axis=1, inplace=True)
        train = train.astype({'time_idx':'int', 'id':'str', 'port':'float', 'intftypeval':'float', 'datatypeval':'float', 'distance':'float', \
            'denisty':'float', 'alinks':'float', 'flinks':'float', 'x':'float', 'y':'float', 'z':'float', 'batt':'float', 'delay':'float', \
            'throughput':'float', 'engcons':'float', 'txpackets_val':'float', 'txbytes_val':'float', 'rxpackets_val':'float', 'rxbytes_val':'float', \
            'drpackets_val':'float', 'txpacketsin_val':'float', 'txbytesin_val':'float', 'rxpacketsout_val':'float', 'rxbytesout_val':'float'})
        max_prediction_length = self.env.prd_time
        max_encoder_length = self.env.obs_time
        training_cutoff = train["time_idx"].max() - max_prediction_length
        train.to_csv('outputs/logs/tftdataset.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/tftdataset.csv'))
        # print(obs)
        # print(train)
        # print(train['time_idx'].max())
        # print(train['time_idx'].dtype.kind)
        # print(train.dtypes)
        self.training_dataset = TimeSeriesDataSet(
            train[lambda x: x.time_idx <= training_cutoff],
            time_idx='time_idx',
            # target=['batt', 'delay', 'throughput', 'engcons', 'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val'],
            target='batt',
            group_ids=['id', 'port', 'intftypeval', 'datatypeval'],
            min_encoder_length=max_encoder_length//2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=['id'],
            static_reals=['port', 'intftypeval', 'datatypeval'],
            time_varying_known_categoricals=[],
            time_varying_unknown_categoricals=[],
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_reals=['distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', 'engcons', 'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val'],
            # target_normalizer=MultiNormalizer([EncoderNormalizer(), TorchNormalizer()]),
            target_normalizer=GroupNormalizer(
                groups=['id', 'port', 'intftypeval', 'datatypeval'], transformation="softplus"
            ),
            categorical_encoders={
                'port':NaNLabelEncoder(add_nan=True), 
                'intftypeval':NaNLabelEncoder(add_nan=True), 
                'datatypeval':NaNLabelEncoder(add_nan=True)
            },
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        self.valid_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, train, predict=True, stop_randomization=True)

        self.policy = TemporalFusionTransformer.from_dataset(
            self.training_dataset, learning_rate=self.hparams.policy_lr, 
            lstm_layers=2, hidden_size=self.hparams.hidden_size, attention_head_size=2,
            dropout=self.hparams.dropout, hidden_continuous_size=self.hparams.hidden_size,
            output_size=1, log_interval=10, reduce_on_plateau_patience=4
        ).to(device)
        res = Tuner(self.trainer).lr_find(
            self.policy,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
            max_lr=10.0,
            min_lr=1e-6,
        )
        # print(f"suggested learning rate: {res.suggestion()}")
        # fig = res.plot(show=True, suggest=True)
        # fig.show()
        self.trainer.fit(self)
              
#*************************************************************
#*************************************************************
#*************************************************************
#*************************************************************
#*************************************************************
class SAC_Agent__(LightningModule):
    """ 
    """
    def __init__(self, env, capacity=100_000, batch_size=256, lr=1e-3, 
                hidden_size=256, gamma=0.99, loss_fn=F.mse_loss, optim=AdamW,
                samples_per_epoch=1_000, tau=0.005, alpha=0.0003, beta=0.0003, epsilon=0.05, reward_scale=2):
                    
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.epoch_cntr = 1
        self.counter = 0
        self.env = env
        self.scale = reward_scale

        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high

        self.actor = ActorNetwork(alpha, self.obs_dims, n_actions=self.action_dims, name='actor', max_action=self.env.max_action_space)
        self.critic_1 = CriticNetwork(beta, self.obs_dims, n_actions=self.action_dims, name='critic_1')
        self.critic_2 = CriticNetwork(beta, self.obs_dims, n_actions=self.action_dims, name='critic_2')
        self.value = ValueNetwork(beta, self.obs_dims, name='value')
        
        self.target_value = copy.deepcopy(self.value)
        self.target_q_net1 = copy.deepcopy(self.critic_1)
        self.target_q_net2 = copy.deepcopy(self.critic_2)
        self.target_policy = copy.deepcopy(self.actor)

        self.buffer = ReplayBuffer(capacity=capacity)

        self.save_hyperparameters('capacity', 'batch_size', 'lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 
            'tau', 'alpha', 'epsilon')

    @T.no_grad()
    def play_episodes(self):
        obs, nds = self.env.getObs()
        done = False
        done = np.zeros((1, 1))
        logger.info('Get predicted action...')
        state = T.tensor([obs]).to(device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        action = actions.cpu().detach().numpy()[0]
        next_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nds))
        # next_obs, reward, info = self.loop.run_until_complete(self.env.step(action))
        done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
        exp = (obs, action, reward, done, next_obs)
        # exp = (obs, action, reward, next_obs)
        self.buffer.append(exp)
        pd.concat([pd.DataFrame(nds, columns=['node']),
                pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                pd.DataFrame(action, columns=self.env.action_cols),
                pd.DataFrame(next_obs, columns=['nxt_obs'+str(i) for i in range(next_obs.shape[1])]), 
                pd.DataFrame(reward, columns=['reward']),
                pd.DataFrame(done, columns=['done']),
                pd.DataFrame(info, columns=['info'])],
                axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
        obs = next_obs
        self.epoch_cntr += 1

    def forward(self, x):
        output = self.actor(x)
        return output
        
    def configure_optimizers(self):
        critic_optimizers = itertools.chain(self.critic_1.parameters(), self.critic_2.parameters())
        critic_optimizer = self.hparams.optim(critic_optimizers, lr=self.hparams.lr)
        value_optimizer = self.hparams.optim(self.value.parameters(), lr=self.hparams.lr)
        actor_optimizer = self.hparams.optim(self.actor.parameters(), lr=self.hparams.lr)
        return [value_optimizer, actor_optimizer, critic_optimizer]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=cpu_count(), #specific to my machine which has 12 CPU cores
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):
        # inputs is batch of experiences exp -> {obs, action, reward, done, next_obs}
        # action_values are the result of trained DQN model when applying action to state (obs)
        obs, action, reward, done, next_obs = batch
        # states, action, rewards, next_states = batch
        batch_dictionary = {}
        # batch_dictionary['delay'] = next_obs[:,:,1].mean()
        # batch_dictionary['goodput'] = next_obs[:,:,2].mean()
        # batch_dictionary['engcons'] = next_obs[:,:,3].mean()
        batch_dictionary['reward'] = reward.sum(0).flatten()


        # self.log('episode/Q-Loss', q_loss_total)
        # print(f'action:\n {action}')
        # print(f'next_obs:\n {next_obs}')
        # print(f'reward:\n {reward}')
        # self.log('episode/Reward', reward.sum(0).flatten())
        self.log('episode/Reward', reward.sum(0).flatten())
        self.log('episode/Performance/Delay', next_obs[:,:,1].mean())
        self.log('episode/Performance/Throughput', next_obs[:,:,2].mean())
        self.log('episode/Performance/Enrgy_Consumption', next_obs[:,:,3].mean())
        self.log('episode/Performance/Txpackets', next_obs[:,:,6].mean())
        self.log('episode/Performance/Txbytes', next_obs[:,:,7].mean())
        self.log('episode/Performance/Rxpackets', next_obs[:,:,8].mean())
        self.log('episode/Performance/Rxbytes', next_obs[:,:,9].mean())
        self.log('episode/Performance/Dropped_Packets', next_obs[:,:,10].mean())
        self.log('episode/Performance/Resedual_Energy_var', next_obs[:,:,13].mean())
        # self.log('Policy/Entropy', T.stack(entropy).mean())
        # self.log('Episode/Reward', T.stack(reward).mean())
        # self.log('Performance/Delay', obs[:,1].mean())
        # self.log('Performance/Goodput', obs[:,2].mean())
        # self.log('Performance/Enrgy_Consumption', obs[:,3].mean())
        # self.log('Performance/Txpackets', obs[:,6].mean())
        # self.log('Performance/Txbytes', obs[:,7].mean())
        # self.log('Performance/Rxpackets', obs[:,8].mean())
        # self.log('Performance/Rxbytes', obs[:,9].mean())
        # self.log('Performance/Energycons', obs[:,10].mean())
        # self.log('Policy/MSE', T.stack(p_mse).mean())


        value = self.value(obs)
        value_ = self.target_value(next_obs)
        print(f'value_:\n {value_}')
        print(f'done:\n {done}')
        value_[done] = 0.0

        if optimizer_idx == 0:
            #train Q-Networks:--------------------------------------------------------
            # (obs, action)              ------> Q1              --> vals1
            # (obs, action)              ------> Q2              --> vals2
            # (nxt_obs)                   ------> TPolicy         --> taction, tprobs

            # (nxt_obs, taction)         ------> TQ1             --> nxt_vals1
            # (nxt_obs, taction)         ------> TQ2             --> nxt_vals2
            # min(nxt_vals1, nxt_vals2)                           --> nxt_vals

            # rewards + gamma * (nxt_vals - alpha * tprobs)       --> exp_vals
            # loss(vals1, exp_vals)                               --> q_loss1
            # loss(vals2, exp_vals)                               --> q_loss2
            # q_loss1 + q_loss2                                   --> q_loss
            #-------------------------------------------------------------------------


            actions, log_probs = self.actor.sample_normal(obs, reparameterize=False)
            t_q_net1 = self.target_q_net1(obs, actions)
            t_q_net2 = self.target_q_net2(obs, actions)
            critic_value = T.min(t_q_net1, t_q_net2)
            print(f'log_probs:\n {log_probs} \n critic_value:\n {critic_value} ')
            value_target = critic_value - log_probs
            value_loss = 0.5 * self.hparams.loss_fn(value, value_target)
            # self.log('episode/Q-Loss', q_loss_total)
            # self.log('episode/Reward', rewards.sum(0).flatten())
            # self.log('Performance/Delay', next_obs[:,:,1].mean())
            # self.log('Performance/Goodput', next_obs[:,:,2].mean())
            # self.log('Performance/Enrgy_Consumption', next_obs[:,:,3].mean())
            # self.log('episode/Reward', reward.mean())
            batch_dictionary['opt'] = optimizer_idx
            batch_dictionary['loss'] = value_loss
            # batch_dictionary={
            # "opt": optimizer_idx,
            # "loss": q_loss_total
            # }
            self.log('episode/DQN/Loss', value_loss)
            # self.log('episode/Q-Loss', q_loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log("episode/Q-performance", {"acc": acc, "recall": recall})
            # return q_loss_total
            return batch_dictionary

        elif optimizer_idx == 1:
            #train the policy:--------------------------------------------------
            # (obs)                 ------> Policy         --> action, probs
            # (obs, action)        ------> Q1             --> vals1
            # (obs, action)        ------> Q2             --> vals2
            # min(vals1, vals2)                            --> vals
            # alpha * probs - vals                         --> p_loss
            #------------------------------------------------------------------
            
            actions, log_probs = self.actor.sample_normal(obs, reparameterize=True)
            t_q_net1 = self.target_q_net1(obs, actions)
            t_q_net2 = self.target_q_net2(obs, actions)
            critic_value = T.min(t_q_net1, t_q_net2)

            actor_loss = log_probs - critic_value
            actor_loss = T.mean(actor_loss)

            batch_dictionary['opt'] = optimizer_idx
            batch_dictionary['loss'] = actor_loss
            # batch_dictionary['entropy'] = entropy.mean()
            # batch_dictionary={
            # "opt": optimizer_idx,
            # "loss": policy_loss,
            # "entropy": entropy.mean()
            # # "mse": policy_mse
            # }
            self.log('episode/Policy/Loss', actor_loss)
            # self.log('episode/Policy/Entropy', entropy.mean())
            # self.log('episode/policy_loss', policy_loss.mean())
            # self.log('episode/policy_mse', policy_mse)

            # self.log('episode/Policy-Loss', policy_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log("episode/Policy-performance", {"acc": acc, "recall": recall})
            # return policy_loss
            return batch_dictionary

        elif optimizer_idx == 2:
            q_hat = self.scale*reward + self.hparams.gamma*value_
            q1_old_policy = self.critic_1.forward(obs, action)
            q2_old_policy = self.critic_2.forward(obs, action)
            critic_1_loss =0.5 * self.hparams.loss_fn(q1_old_policy, q_hat.float())
            critic_2_loss =0.5 * self.hparams.loss_fn(q2_old_policy, q_hat.float())

            critic_loss = critic_1_loss + critic_2_loss

            batch_dictionary['opt'] = optimizer_idx
            batch_dictionary['loss'] = critic_loss

            return batch_dictionary

    def training_epoch_end(self, training_step_outputs):
        self.play_episodes()
        # obs = obs.flatten()
        #run async function as sync
        # self.loop.run_until_complete(self.play_episodes(policy=self.policy))
        polyak_average(self.critic_1, self.target_q_net1, tau=self.hparams.tau)
        polyak_average(self.critic_2, self.target_q_net2, tau=self.hparams.tau)
        polyak_average(self.value, self.target_value, tau=self.hparams.tau)
        polyak_average(self.actor, self.target_policy, tau=self.hparams.tau)
        # self.log('episode_return', self.env.return_queue[-1])
        # calculating average loss  

        # if isinstance(training_step_outputs, dict):

        # print(training_step_outputs)
        # obs, _ = self.env.getObs()
        # q_loss = []
        # p_loss = []
        # reward = []
        # entropy = []
        # for output in training_step_outputs:
        #     for item in output:
        #         if item['opt'] == 1:
        #             p_loss.append(item['loss'])
        #             entropy.append(item['entropy'])
        #             # p_mse.append(item['mse'])
        #         else:
        #             q_loss.append(item['loss'])
        #         reward.append(item['reward'])
        # self.log('Policy/Loss', T.stack(p_loss).mean())
        # self.log('Q/Loss', T.stack(q_loss).mean())
        # self.log('Policy/Entropy', T.stack(entropy).mean())
        # self.log('Episode/Reward', T.stack(reward).sum(0).flatten())
        # self.log('Performance/Delay', obs[:,1].mean())
        # self.log('Performance/Goodput', obs[:,2].mean())
        # self.log('Performance/Enrgy_Consumption', obs[:,3].mean())
        # self.log('Performance/Txpackets', obs[:,6].mean())
        # self.log('Performance/Txbytes', obs[:,7].mean())
        # self.log('Performance/Rxpackets', obs[:,8].mean())
        # self.log('Performance/Rxbytes', obs[:,9].mean())
        # self.log('Performance/Energycons', obs[:,10].mean())
        # self.log('Policy/MSE', T.stack(p_mse).mean())


        # correct=sum([x[0]["correct"] for  x in training_step_outputs])
        # total=sum([x[0]["total"] for  x in training_step_outputs])          

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/experiences.csv')
            quietRun('rm -r outputs/logs/lightning_logs/')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
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
            logger.info(f'{len(self.buffer)} samples in experience buffer. Filling...')
            self.play_episodes()
        trainer.fit(self)
        
        await asyncio.sleep(0)  
        
class SAC_Agent_:
    def __init__(self, env, alpha=0.0003, beta=0.0003, gamma=0.99, max_size=1000000, tau=0.005,
    layer1_size=265, layer2_size=256, batch_size=256, samples_per_epoch=1_000, reward_scale=2):

        self.loop = asyncio.get_running_loop()
        self.epoch_cntr = 1
        self.env = env
        self.input_dims = self.env.observation_space.shape[1]
        self.n_actions = self.env.action_space.shape[1]
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch

        self.memory = ReplayBufferSAC(max_size, self.input_dims, self.n_actions)

        self.actor = ActorNetwork(alpha, self.input_dims, n_actions=self.n_actions, name='actor', max_action=self.env.max_action_space)
        self.critic_1 = CriticNetwork(beta, self.input_dims, n_actions=self.n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, self.input_dims, n_actions=self.n_actions, name='critic_2')
        self.value = ValueNetwork(beta, self.input_dims, name='value')
        self.target_value = ValueNetwork(beta, self.input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation]).to(device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]
        # return actions.detach().cpu().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return np.zeros((1,1)), np.zeros((1,1))

        state, action, reward, state_, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(device).reshape(-1,1)
        done = T.tensor(done).to(device).reshape(-1,1)
        state_ = T.tensor(state_, dtype=T.float).to(device)
        state = T.tensor(state, dtype=T.float).to(device)
        action = T.tensor(action, dtype=T.float).to(device)
        print(f'reward:\n{reward}\ndone:\n{done}\nstate_:\n{state_}\nstate:\n{state}\naction:\n{action}\n')
        value = self.value(state)
        value_ = self.target_value(state_)
        # done[not done.all()] = True if nsamples == self.hparams.samples_per_epoch else done
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action)
        q2_old_policy = self.critic_2.forward(state, action)
        critic_1_loss =0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss =0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
        return actor_loss, critic_loss
        

    async def run(self):
        try:
            quietRun('rm -r outputs/logs/experiences.csv')
            # quietRun('rm -r outputs/logs/lightning_logs/')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        best_score = self.env.best_reward   
        reward_history = []
        delay_history = []
        throughput_history = []
        energycons_history = []
        txpackets_history = []
        txbytes_history = []
        rxpackets_history = []
        rxbytes_history = []
        droppackets_history = []
        resenergyvar_history = []
        policyloss_history = []
        qloss_history = []
        load_checkpoint = False

        if load_checkpoint:
            self.load_models()

        while True:
            obs, nds = self.env.reset()
            done = np.zeros((1, 1))
            action = self.choose_action(obs)
            logger.warn(action)
            obs_, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nds))
            done[not done.all()] = True if self.epoch_cntr % self.samples_per_epoch == 0 else done
            self.remember(obs, action, reward, obs_, done)
            pd.concat([pd.DataFrame(nds, columns=['node']),
            pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
            pd.DataFrame(action, columns=self.env.action_cols),
            pd.DataFrame(obs_, columns=['nxt_obs'+str(i) for i in range(obs_.shape[1])]), 
            pd.DataFrame(reward, columns=['reward']),
            pd.DataFrame(done, columns=['done']),
            pd.DataFrame(info, columns=['info'])],
            axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            if not load_checkpoint:
                policy_loss, q_loss = self.learn()
                if isinstance(policy_loss, T.Tensor):
                    policy_loss = policy_loss.detach().cpu().numpy()
                    q_loss = q_loss.detach().cpu().numpy()
                print(policy_loss)
                print(q_loss)
                policyloss_history.append(*policy_loss.flatten())
                qloss_history.append(*q_loss.flatten())
            obs = obs_
            self.epoch_cntr += 1

            print(obs_)
            print(*obs_[:,2].mean().flatten())
            reward_history.append(*reward.sum(0).flatten())
            delay_history.append(*obs_[:,1].mean().flatten())
            throughput_history.append(*obs_[:,2].mean().flatten())
            energycons_history.append(*obs_[:,3].mean().flatten())
            txpackets_history.append(*obs_[:,6].mean().flatten())
            txbytes_history.append(*obs_[:,7].mean().flatten())
            rxpackets_history.append(*obs_[:,8].mean().flatten())
            rxbytes_history.append(*obs_[:,9].mean().flatten())
            droppackets_history.append(*obs_[:,10].mean().flatten())
            resenergyvar_history.append(*obs_[:,13].mean().flatten())

            avg_score = np.mean(reward_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                self.env.best_reward = avg_score
                if not load_checkpoint:
                    self.save_models()
            
            if not load_checkpoint:
                plot_learning_curve([x for x in range(self.epoch_cntr-1)], reward_history, 'outputs/plots/score.png', 'reward', 1)
                plot_learning_curve([x for x in range(self.epoch_cntr-1)], delay_history, 'outputs/plots/score.png', 'delay', 2)
                plot_learning_curve([x for x in range(self.epoch_cntr-1)], throughput_history, 'outputs/plots/score.png', 'throughput', 3)
                plot_learning_curve([x for x in range(self.epoch_cntr-1)], energycons_history, 'outputs/plots/score.png', 'energy consumption', 4)
                plot_learning_curve([x for x in range(self.epoch_cntr-1)], txpackets_history, 'outputs/plots/score.png', 'tx packets', 5)
                plot_learning_curve([x for x in range(self.epoch_cntr-1)], txbytes_history, 'outputs/plots/score.png', 'tx bytes', 6)
                plot_learning_curve([x for x in range(self.epoch_cntr-1)], rxpackets_history, 'outputs/plots/score.png', 'rx packets', 7)
                plot_learning_curve([x for x in range(self.epoch_cntr-1)], rxbytes_history, 'outputs/plots/score.png', 'rx bytes', 8)
                plot_learning_curve([x for x in range(self.epoch_cntr-1)], droppackets_history, 'outputs/plots/score.png', 'dropped packets', 9)
                plot_learning_curve([x for x in range(self.epoch_cntr-1)], resenergyvar_history, 'outputs/plots/score.png', 'reseducal energy var', 10)
                plot_learning_curve([x for x in range(self.epoch_cntr-1)], policyloss_history, 'outputs/plots/score.png', 'policy loss', 11)
                plot_learning_curve([x for x in range(self.epoch_cntr-1)], qloss_history, 'outputs/plots/score.png', 'q loss', 12)
            
            await asyncio.sleep(0) 

#********************************************************

#***********************************************


class HE(LightningModule):
    """ 
    """
    def __init__(self, env, capacity=100_000, batch_size=256, lr=1e-3, 
                hidden_size=256, gamma=0.99, loss_fn=F.mse_loss, optim=AdamW,
                samples_per_epoch=1_000, tau=0.05, alpha=0.02, epsilon=0.05, her=0.8):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.counter = 0
        self.env = env

        self.obs_dims = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        # self.max_action = self.env.max_action
        # self.obs_dims = self.env.observation_space.shape[0]
        # self.action_dims = self.env.action_space.shape[0]
        # self.max_action = self.env.action_space.high

        self.q_net1 = HE_DQN(hidden_size, self.obs_dims, self.action_dims)
        self.q_net2 = HE_DQN(hidden_size, self.obs_dims, self.action_dims)
        self.policy = HE_GradientPolicy(hidden_size, self.obs_dims, self.action_dims)

        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)
        self.target_policy = copy.deepcopy(self.policy)

        self.buffer = ReplayBuffer(capacity=capacity)

        self.save_hyperparameters('capacity', 'batch_size', 'lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 
            'tau', 'alpha', 'epsilon')

    @T.no_grad()
    def play_episodes(self, policy=None):
        obs, nds = self.env.getObs()
        done = False
        done = np.zeros((1, 1))
        nsamples = 1 
        while not done.all():   
            if policy and random.random() > self.hparams.epsilon:
                logger.info('Get predicted action...')
                action, _, _ = self.policy(obs)
                action = action.detach().cpu().numpy()
                # action = action.detach().numpy()
            else:
                # action = self.env.action_space.sample()
                logger.info('Get random action...')
                action = np.random.uniform(-1, 1, size=(obs.shape[0],self.action_dims))
                # action = np.random.uniform(0, 1, size=(obs.shape[0],self.action_dims))
                # action = np.random.normal(loc=0, scale=1, size=(1,self.action_dims))
                # action = np.random.standard_normal(size=(1,self.action_dims))
                #get zscore of action values                 
            next_obs, reward, done, info = self.loop.run_until_complete(self.env.step(action, obs, nds))
            # next_obs, reward, info = self.loop.run_until_complete(self.env.step(action))
            done[not done.all()] = True if nsamples == self.hparams.samples_per_epoch else done
            exp = (obs, action, reward, done, next_obs)
            # exp = (obs, action, reward, next_obs)
            self.buffer.append(exp)
            pd.concat([pd.DataFrame(nds, columns=['node']),
                    pd.DataFrame(obs, columns=['obs'+str(i) for i in range(obs.shape[1])]), 
                    pd.DataFrame(action, columns=self.env.action_cols),
                    pd.DataFrame(next_obs, columns=['nxt_obs'+str(i) for i in range(next_obs.shape[1])]), 
                    pd.DataFrame(reward, columns=['reward']),
                    pd.DataFrame(done, columns=['done']),
                    pd.DataFrame(info, columns=['info'])],
                    axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            obs = next_obs
            nsamples += 1

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
        # inputs is batch of experiences exp -> {obs, action, reward, done, next_obs}
        # action_values are the result of trained DQN model when applying action to state (obs)
        obs, action, reward, done, next_obs = batch
        # states, action, rewards, next_states = batch
        batch_dictionary = {}
        # batch_dictionary['delay'] = next_obs[:,:,1].mean()
        # batch_dictionary['goodput'] = next_obs[:,:,2].mean()
        # batch_dictionary['engcons'] = next_obs[:,:,3].mean()
        batch_dictionary['reward'] = reward.mean()


        # self.log('episode/Q-Loss', q_loss_total)
        # print(f'action:\n {action}')
        # print(f'next_obs:\n {next_obs}')
        # print(f'reward:\n {reward}')
        # self.log('episode/Reward', reward.sum(0).flatten())
        self.log('episode/Reward', reward.mean())
        self.log('episode/Performance/Delay', next_obs[:,:,1].mean())
        self.log('episode/Performance/Throughput', next_obs[:,:,2].mean())
        self.log('episode/Performance/Enrgy_Consumption', next_obs[:,:,3].mean())
        self.log('episode/Performance/Txpackets', next_obs[:,:,6].mean())
        self.log('episode/Performance/Txbytes', next_obs[:,:,7].mean())
        self.log('episode/Performance/Rxpackets', next_obs[:,:,8].mean())
        self.log('episode/Performance/Rxbytes', next_obs[:,:,9].mean())
        self.log('episode/Performance/Dropped_Packets', next_obs[:,:,10].mean())
        self.log('episode/Performance/Resedual_Energy_var', next_obs[:,:,13].mean())
        # self.log('Policy/Entropy', T.stack(entropy).mean())
        # self.log('Episode/Reward', T.stack(reward).mean())
        # self.log('Performance/Delay', obs[:,1].mean())
        # self.log('Performance/Goodput', obs[:,2].mean())
        # self.log('Performance/Enrgy_Consumption', obs[:,3].mean())
        # self.log('Performance/Txpackets', obs[:,6].mean())
        # self.log('Performance/Txbytes', obs[:,7].mean())
        # self.log('Performance/Rxpackets', obs[:,8].mean())
        # self.log('Performance/Rxbytes', obs[:,9].mean())
        # self.log('Performance/Energycons', obs[:,10].mean())
        # self.log('Policy/MSE', T.stack(p_mse).mean())

        if optimizer_idx == 0:
            #train Q-Networks:--------------------------------------------------------
            # (obs, action)              ------> Q1              --> vals1
            # (obs, action)              ------> Q2              --> vals2
            # (nxt_obs)                   ------> TPolicy         --> taction, tprobs

            # (nxt_obs, taction)         ------> TQ1             --> nxt_vals1
            # (nxt_obs, taction)         ------> TQ2             --> nxt_vals2
            # min(nxt_vals1, nxt_vals2)                           --> nxt_vals

            # rewards + gamma * (nxt_vals - alpha * tprobs)       --> exp_vals
            # loss(vals1, exp_vals)                               --> q_loss1
            # loss(vals2, exp_vals)                               --> q_loss2
            # q_loss1 + q_loss2                                   --> q_loss
            #-------------------------------------------------------------------------

            action_values1 = self.q_net1(obs, action)
            action_values2 = self.q_net2(obs, action)

            target_action, target_log_probs, _ = self.target_policy(next_obs)

            next_action_values = T.min(
                self.target_q_net1(next_obs, target_action),
                self.target_q_net2(next_obs, target_action)
            )
            next_action_values[done] = 0.0

            expected_action_values = reward + self.hparams.gamma * (next_action_values - self.hparams.alpha * target_log_probs)
            q_loss1 = self.hparams.loss_fn(action_values1.float(), expected_action_values.float())
            q_loss2 = self.hparams.loss_fn(action_values2.float(), expected_action_values.float())

            q_loss_total = q_loss1 + q_loss2
            # self.log('episode/Q-Loss', q_loss_total)
            # self.log('episode/Reward', rewards.sum(0).flatten())
            # self.log('Performance/Delay', next_obs[:,:,1].mean())
            # self.log('Performance/Goodput', next_obs[:,:,2].mean())
            # self.log('Performance/Enrgy_Consumption', next_obs[:,:,3].mean())
            # self.log('episode/Reward', reward.mean())
            batch_dictionary['opt'] = optimizer_idx
            batch_dictionary['loss'] = q_loss_total
            # batch_dictionary={
            # "opt": optimizer_idx,
            # "loss": q_loss_total
            # }
            self.log('episode/DQN/Loss', q_loss_total)
            # self.log('episode/Q-Loss', q_loss_total, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log("episode/Q-performance", {"acc": acc, "recall": recall})
            # return q_loss_total
            return batch_dictionary

        elif optimizer_idx == 1:
            #train the policy:--------------------------------------------------
            # (obs)                 ------> Policy         --> action, probs
            # (obs, action)        ------> Q1             --> vals1
            # (obs, action)        ------> Q2             --> vals2
            # min(vals1, vals2)                            --> vals
            # alpha * probs - vals                         --> p_loss
            #------------------------------------------------------------------
            
            action, log_probs, entropy = self.policy(obs)

            action_values = T.min(
                self.q_net1(obs, action),
                self.q_net2(obs, action)
            )
            policy_loss = (self.hparams.alpha * log_probs - action_values).mean()
            # mse = MeanSquaredError().to(device)
            # pred = self.hparams.alpha * log_probs
            # target = action_values
            # policy_mse = mse(pred, target)
            # print(f'pred: {pred}')
            # print(f'labels: {target}')
            # identifying number of correct predections in a given batch
            # correct=pred.argmax(dim=1).eq(target).sum().item()
            # print(f'policy_mse: {policy_mse}')
            # identifying total number of target in a given batch
            # total=len(target.flatten())
            # print(f'total: {total}')
            batch_dictionary['opt'] = optimizer_idx
            batch_dictionary['loss'] = policy_loss
            batch_dictionary['entropy'] = entropy.mean()
            # batch_dictionary={
            # "opt": optimizer_idx,
            # "loss": policy_loss,
            # "entropy": entropy.mean()
            # # "mse": policy_mse
            # }
            self.log('episode/Policy/Loss', policy_loss)
            self.log('episode/Policy/Entropy', entropy.mean())
            # self.log('episode/policy_loss', policy_loss.mean())
            # self.log('episode/policy_mse', policy_mse)

            # self.log('episode/Policy-Loss', policy_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log("episode/Policy-performance", {"acc": acc, "recall": recall})
            # return policy_loss
            return batch_dictionary

    def training_epoch_end(self, training_step_outputs):
        self.play_episodes(policy=self.policy)
        # obs = obs.flatten()
        #run async function as sync
        # self.loop.run_until_complete(self.play_episodes(policy=self.policy))
        polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
        polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)
        # self.log('episode_return', self.env.return_queue[-1])
        # calculating average loss  

        # if isinstance(training_step_outputs, dict):

        # print(training_step_outputs)
        # obs, _ = self.env.getObs()
        # q_loss = []
        # p_loss = []
        # reward = []
        # entropy = []
        # for output in training_step_outputs:
        #     for item in output:
        #         if item['opt'] == 1:
        #             p_loss.append(item['loss'])
        #             entropy.append(item['entropy'])
        #             # p_mse.append(item['mse'])
        #         else:
        #             q_loss.append(item['loss'])
        #         reward.append(item['reward'])
        # self.log('Policy/Loss', T.stack(p_loss).mean())
        # self.log('Q/Loss', T.stack(q_loss).mean())
        # self.log('Policy/Entropy', T.stack(entropy).mean())
        # self.log('Episode/Reward', T.stack(reward).mean())
        # self.log('Performance/Delay', obs[:,1].mean())
        # self.log('Performance/Goodput', obs[:,2].mean())
        # self.log('Performance/Enrgy_Consumption', obs[:,3].mean())
        # self.log('Performance/Txpackets', obs[:,6].mean())
        # self.log('Performance/Txbytes', obs[:,7].mean())
        # self.log('Performance/Rxpackets', obs[:,8].mean())
        # self.log('Performance/Rxbytes', obs[:,9].mean())
        # self.log('Performance/Energycons', obs[:,10].mean())
        # self.log('Policy/MSE', T.stack(p_mse).mean())


        # correct=sum([x[0]["correct"] for  x in training_step_outputs])
        # total=sum([x[0]["total"] for  x in training_step_outputs])          

    async def run(self):
        # Start tensorboard.
        try:
            quietRun('rm -r outputs/logs/experiences.csv')
            quietRun('rm -r outputs/logs/lightning_logs/')
            # quietRun('rm -r outputs/content/videos/')
            # quietRun('tensorboard --logdir outputs/logs/')
        except Exception as ex:
            logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
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
            logger.info(f'{len(self.buffer)} samples in experience buffer. Filling...')
            self.play_episodes()
        trainer.fit(self)
        
        await asyncio.sleep(0)     

