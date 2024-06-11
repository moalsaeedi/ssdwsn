import copy
import itertools
from typing import List, Tuple
from os import cpu_count, path
from enum import Enum
from itertools import count
from collections import namedtuple
import random
import torch as T
import torch.nn.functional as F
import numpy as np
import pandas as pd
import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from ssdwsn.util.utils import quietRun, CustomFormatter
from ssdwsn.data.addr import Addr
from ssdwsn.openflow.action import ForwardUnicastAction, DropAction
from ssdwsn.openflow.entry import Entry
from ssdwsn.openflow.window import Window
from ssdwsn.util.constants import Constants as ct
import networkx as nx
import re
from collections import deque
import torch as T
import torch.nn.functional as F
from ssdwsn.util.utils import quietRun, CustomFormatter
from ssdwsn.app.lossFunction import lossBCE, lossCCE, lossMSE
from app.utilts import polyak_average
from ssdwsn.app.dataset import RLDataset_PPO_shuffle, ExperienceSourceDataset
from ssdwsn.app.network import PPO_GradientPolicy, PPO_Policy, PPO_ValueNet, PPO_Policy_Pred, PPO_ValueNet_Pred, DQN
from torch.optim import Adam, AdamW
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('agg')
import time
from math import ceil, log, sqrt, atan2, pi
from sys import exc_info

#logging----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
#-----------------------------------

seed_everything(33, workers=True)
T.cuda.empty_cache()
num_devices = T.cuda.device_count() if T.cuda.is_available() else cpu_count()
device = f'cuda:{num_devices-1}' if T.cuda.is_available() else 'cpu'
if T.cuda.is_available():
    T.set_float32_matmul_precision('high')

class PPO_ATCP(LightningModule):
    """ Adaptive Traffic Controll On-Policy PPO_DRL Agent
    """
    def __init__(self, ctrl, num_envs=1, batch_size=2, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, lr=1e-3, gamma=0.99, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=Adam):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.automatic_optimization = False
        self.ctrl = ctrl
        self.networkGraph = self.ctrl.networkGraph

        self.obs_cols = ['port', 'intftypeval', 'datatypeval', 'distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', \
        'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val', 'rptti'] 
        self.cal_cols = ['ts']
        self.action_cols = ['isaggr', 'nxhop', 'rptti']
        self.action_space = np.empty((0, len(self.action_cols)))
        self.observation_space = np.empty((0, len(self.obs_cols)+len(self.cal_cols)))

        self.obs_dims = self.observation_space.shape[1]
        self.action_dim = self.action_space.shape[1]
        
        self.obs_nds = None
        self.policy = PPO_Policy(self.obs_dims, hidden_size, self.action_dim)
        self.target_policy = copy.deepcopy(self.policy)
        self.value_net = PPO_ValueNet(self.obs_dims, hidden_size)
        # self.value_net = PPO_ValueNet(self.obs_dims, hidden_size)
        self.target_val_net = copy.deepcopy(self.value_net) 

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.ep_return = []
        self.best_return = 0
        self.ep_step = 0
        self.tb_logger = None

        self.save_hyperparameters('batch_size', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'lr', 'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')

    def reset(self, nds=None):
        """Get network state observation"""
        state, nodes, sink_state, sink_state_nodes = self.networkGraph.getState(nodes=nds, cols=self.obs_cols)  
        obs = np.column_stack((state, np.repeat(int(time.time()), state.shape[0], axis=0).reshape(-1,1)))
        
        data = pd.concat([pd.DataFrame(obs, columns=self.obs_cols+self.cal_cols, dtype=float)], axis=1)
        sink_state = pd.DataFrame(sink_state, columns=self.obs_cols, dtype=float)
        return data, nodes, sink_state, sink_state_nodes
    
    def getReward(self, obs, prv_obs):
        obs_ts = obs['ts'].mean()
        prv_obs_ts = prv_obs['ts'].mean()
        prv_E = prv_obs['batt'].to_numpy().reshape(-1,1)

        TH = obs['throughput'].to_numpy().reshape(-1,1)
        D = obs['delay'].to_numpy().reshape(-1,1)
        E = obs['batt'].to_numpy().reshape(-1,1)
        EC = ((prv_E - E)/(obs_ts - prv_obs_ts)).reshape(-1,1)
        RT = obs['rptti'].to_numpy().reshape(-1,1)
        # R = TH/ct.MAX_BANDWIDTH + 4 - D/ct.MAX_DELAY - EC/ct.MAX_ENRCONS - (EC/ct.MAX_ENRCONS).var() - (RT/ct.MAX_RP_TTI).var()
        R = TH/ct.MAX_BANDWIDTH + 3 - D/ct.MAX_DELAY - EC/ct.MAX_ENRCONS - (EC/ct.MAX_ENRCONS).var()
        
        return np.nan_to_num(R, nan=0)
    
    def step(self, action, obs, obs_nds):
        action = MinMaxScaler((0,1)).fit_transform(action)
        act_nds = dict(zip(obs_nds.flatten(), action))
        
        isaggr_idxs = np.where((action[:,0] > 0.50))
        isaggr_action_nodes = obs_nds[isaggr_idxs,:].flatten().tolist()
        for nd, act in act_nds.items():
            neighbors = [edge[1] for edge in list(self.networkGraph.getGraph().edges(nbunch=nd, data=True, keys=True))]
            nd_pos = self.networkGraph.getPosition(nd)
            src_dist = self.networkGraph.getDistance(nd)
            act_angle = act[1] * pi # angle between 0 and π radians
            act_nxh_node = nd
            vl = pi
            for nr in neighbors:
                dst_dist = self.networkGraph.getDistance(nr)
                nd_nr_edge = self.networkGraph.getEdge(nd, nr)
                nr_pos = self.networkGraph.getPosition(nr)
                y = nr_pos[1] - nd_pos[1]
                x = nr_pos[0] - nd_pos[0]
                angle = atan2(y, x)
                angle = angle if angle > 0 else (angle + (2*pi))
                act_nxh_val = abs(act_angle - angle)
                if act_nxh_val < vl and src_dist > dst_dist:
                    vl = act_nxh_val
                    act_nxh_node = nr

            # action value
            val = int.to_bytes(ct.DRL_AG_INDEX, 1, 'big', signed=False)+int.to_bytes(1 if nd in isaggr_action_nodes else 0, ct.DRL_AG_LEN, 'big', signed=False)+\
                int.to_bytes(ct.DRL_NH_INDEX, 1, 'big', signed=False)+int.to_bytes(Addr(re.sub(r'^.*?.', '', act_nxh_node)[1:]).intValue(), ct.DRL_NH_LEN, 'big', signed=False)+\
                int.to_bytes(ct.DRL_RT_INDEX, 1, 'big', signed=False)+int.to_bytes(int(act[2] * (ct.MAX_RP_TTI-ct.MIN_RP_TTI))+ct.MIN_RP_TTI, ct.DRL_RT_LEN, 'big', signed=False)

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
                self.loop.run_until_complete(self.ctrl.setDRLAction(net=int(sinkId.split('.')[0]), sinkId=sinkId, sinkAddr=route[0], dst=route[-1], newVal=val, path=route))

        time.sleep(15)
        next_obs, _, next_sink_obs, _ = self.reset(obs_nds)
        reward = self.getReward(next_obs, obs)
        done = np.zeros((next_obs.shape[0],1), dtype=bool) #TODO change the logic when to set done to True (since it is a continouse process target optimization are always changing as per the network progress and resource drained)           
        info = np.empty((next_obs.shape[0],1), dtype=str)
        return next_obs, reward, done, info, next_sink_obs

    @T.no_grad()
    def play_episodes(self, policy=None):
        obs, nodes, _, _ = self.reset()
        print('ATCP COLLECTING DATA ....')     
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            log_prob, action = self.policy(obs.to_numpy())
            action = action.detach().cpu().numpy()
            log_prob = log_prob.detach().cpu().numpy()
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info, nxt_sink_obs  = self.step(action, obs, nodes)
            self.buffer.append((obs.to_numpy(), log_prob, action, reward, done, nxt_obs.to_numpy()))
            self.num_samples.append(nxt_obs.shape[0])

            # pd.concat([pd.DataFrame(nodes, columns=['node']),
            #     obs, 
            #     pd.DataFrame(action, columns=self.action_cols),
            #     nxt_obs, 
            #     pd.DataFrame(reward, columns=['reward']),
            #     pd.DataFrame(done, columns=['done']),
            #     pd.DataFrame(info, columns=['info'])],
            #     axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))

            self.pltMetrics(reward, nodes, obs, nxt_obs, nxt_sink_obs)    
            obs = nxt_obs            
            self.ep_step += 1

    def pltMetrics(self, reward, nodes, obs, nxt_obs, nxt_sink_obs):
        if self.ep_step == 0 and self.global_step == 0:
            self.tb_logger.add_scalars('Episode/R', {
                'R': np.zeros(1)
                }, global_step=self.ep_step
            )
        else:
            self.tb_logger.add_scalars('Episode/R', {
                'R': reward.sum()
                }, global_step=self.ep_step
            )
        self.tb_logger.add_scalars('Episode/D', {
            'D': nxt_obs['delay'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/T', {
            'T': nxt_obs['throughput'].mean() 
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/E', {
            'E': nxt_obs['batt'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/DP', {
            'DP': nxt_obs['drpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/TX', {
            'TX': nxt_obs['txpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RX', {
            'RX': nxt_obs['rxpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/TX_in', {
            'TX_in': nxt_sink_obs['txpacketsin_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RX_out', {
            'RX_out': nxt_sink_obs['rxpacketsout_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT', {
            'RT': nxt_obs['rptti'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT_var', {
            'RT_var': nxt_obs['rptti'].var()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/EC', {
            'EC': ((obs['batt'] - nxt_obs['batt'])/(nxt_obs['ts'] - obs['ts'])).mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/EC_var', {
            'EC_var': ((obs['batt'] - nxt_obs['batt'])/(nxt_obs['ts'] - obs['ts'])).var()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/E_nds', 
            dict([(str(nodes[nd].item()), nxt_obs['batt'].to_numpy()[nd]) for nd in range(nodes.shape[0])]),
            global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT_nds', 
            dict([(str(nodes[nd].item()), nxt_obs['rptti'].to_numpy()[nd]) for nd in range(nodes.shape[0])]),
            global_step=self.ep_step
        )
        # self.tb_logger.add_scalars('Episode/Value/Loss', {
        #     'Value_Loss': np.array(self.ep_value_loss).mean()
        #     }, global_step=self.ep_step
        # )
        # self.tb_logger.add_scalars('Episode/Policy/Loss', {
        #     'Policy_Loss': np.array(self.ep_policy_loss).mean()
        #     }, global_step=self.ep_step
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
        # return self.hparams.optim(self.parameters(), lr=self.hparams.lr)

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

    def training_step(self, batch, batch_idx):

        # optimizer = self.optimizers()
        value_opt, policy_opt = self.optimizers()
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = batch        

        # rev_nxt_obs = self.env.scaler.inverse_transform(T.clone(nxt_obs_b.data).detach().cpu().numpy())
        
        state_values = self.value_net(obs_b)

        with T.no_grad():
            _, nxt_action = self.target_policy(nxt_obs_b)
            nxt_state_values = self.target_val_net(nxt_obs_b)
            # nxt_state_values[done_b] = 0.0 
            target = reward_b + self.hparams.gamma * nxt_state_values
        
        advantages = (target - state_values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # value_opt
        value_loss = self.hparams.loss_fn(state_values, target.float())
        value_opt.zero_grad()
        # self.manual_backward(value_loss)
        value_loss.backward()
        T.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
        value_opt.step()
        self.tb_logger.add_scalars('Training/Value/Loss', {'loss': value_loss}, global_step=self.global_step)

        # policy_opt
        log_prob, _ = self.policy(obs_b)
        prv_log_prob = log_prob_b
        
        rho = T.exp(log_prob - prv_log_prob)
        # rho = log_prob / prv_log_prob

        surrogate_1 = rho * advantages
        surrogate_2 = T.clip(rho, 1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

        policy_loss = - T.minimum(surrogate_1, surrogate_2).mean()
        policy_opt.zero_grad()
        # self.manual_backward(value_loss)
        policy_loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        policy_opt.step()

        self.tb_logger.add_scalars('Training/Policy/Loss', {'loss': policy_loss}, global_step=self.global_step)

        # self.log_dict({"value_loss": value_loss.item(), "policy_loss": policy_loss.item(), "return": reward_b.sum()}, prog_bar=True)
                
        # For simplecity:
        #         # Total loss is a combination of critic loss and policy loss
        total_loss = value_loss + policy_loss

        # Perform backpropagation
        # self.manual_backward(total_loss)
        # optimizer.step()
        # optimizer.zero_grad()

        # self.tb_logger.add_scalars('Training/Loss', {'loss': total_loss}, global_step=self.global_step)
        # # Return the total loss for logging purposes
        # self.log_dict({"loss": total_loss, "return": reward_b.sum()}, prog_bar=True)
        return total_loss
                
    def on_train_epoch_end(self):
        # if self.best_return > 0:
        #     self.policy.load_state_dict(T.load('outputs/logs/best_policy'))
        #     self.value_net.load_state_dict(T.load('outputs/logs/best_value'))
        
        self.target_val_net.load_state_dict(self.value_net.state_dict())            
        self.target_policy.load_state_dict(self.policy.state_dict())
        print(f'ATCP END EPOCH: {self.current_epoch}*****************')
        # time.sleep(1)
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        # try:
        #     quietRun('rm -r outputs/logs/')
        #     # quietRun('rm -r outputs/logs/experiences.csv')
        #     # quietRun('rm -r outputs/logs/lightning_logs/version_0')
        #     # quietRun('rm -r outputs/content/videos/')
        #     # quietRun('tensorboard --logdir outputs/logs/')
        # except Exception as ex:
        #     logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('ATCP START TRAINING ...')
        # self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        quietRun('chmod -R 777 outputs/logs')
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        '''
        checkpoint_callback = ModelCheckpoint(
            dirpath='outputs/logs',
            monitor='return',
            save_top_k=3,
            filename='model-{epoch:02d}-{return:.2f}',
            mode='max',
        )
        '''
        # checkpoint_callback = ModelCheckpoint(dirpath='outputs/logs')

        trainer = Trainer(
            accelerator='auto',      
            devices=num_devices,
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            # callbacks=[checkpoint_callback],
            # logger=self.tb_logger,
            reload_dataloaders_every_n_epochs = 1,              
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        T.autograd.detect_anomaly(True)
        self.play_episodes()
        trainer.fit(self)

class PPO_NSFP(LightningModule):
    """ Network State Forcasting On-Policy PPO_DRL Agent
    """
    def __init__(self, ctrl, num_envs=1, batch_size=2, obs_time = 20, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, lr=1e-3, gamma=0.99, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.automatic_optimization = False
        self.ctrl = ctrl
        self.networkGraph = self.ctrl.networkGraph
        self.obs_cols = ['port', 'intftypeval', 'datatypeval', 'distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', \
        'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val', 'rptti'] 
        self.cal_cols = ['ts']
        self.action_cols = ['batt', 'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val']
        self.action_space = np.empty((0, len(self.action_cols)))
        self.observation_space = np.empty((0, len(self.obs_cols)+len(self.cal_cols)))

        self.obs_dims = self.observation_space.shape[1]
        self.action_dims = self.action_space.shape[1]
        self.obs_nds = None
        self.policy = PPO_Policy_Pred(self.obs_dims, hidden_size, self.action_dims)
        self.target_policy = copy.deepcopy(self.policy)
        # self.value_net = PPO_Att_ValueNet(self.obs_dims+self.action_dims, hidden_size)
        self.value_net = PPO_ValueNet_Pred(self.obs_dims, hidden_size)
        self.target_val_net = copy.deepcopy(self.value_net) 

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.ep_step = 0
        self.tb_logger = None
        self.init_ts = time.time()
        
        self.save_hyperparameters('batch_size', 'obs_time', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'lr', 'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')
        
    def reset(self, nds=None):
        """Get network state observation for certain amount of time"""
        obs_time = self.hparams.obs_time
        obs = self.observation_space
        _nodes = np.empty((0,1))
        sink_obs = np.empty((0, len(self.obs_cols)))
        _sink_nodes = np.empty((0,1))
        for _ in range(obs_time):
            state, nodes, sink_state, sink_state_nodes = self.networkGraph.getState(nodes=nds, cols=self.obs_cols)                
            obs = np.vstack((obs, np.column_stack((state, np.repeat(int(time.time()), state.shape[0], axis=0).reshape(-1,1)))))
            _nodes = np.vstack((_nodes, nodes))
            sink_obs = np.vstack((sink_obs, sink_state))
            _sink_nodes = np.vstack((_sink_nodes, sink_state_nodes))
            nds = nodes
            time.sleep(1)

        data = pd.concat([pd.DataFrame(obs, columns=self.obs_cols+self.cal_cols, dtype=float)], axis=1)
        sink_state = pd.DataFrame(sink_obs, columns=self.obs_cols, dtype=float)
        return data, _nodes, sink_state, _sink_nodes
    
    def getReward(self, obs, prv_obs, action):
        R = (((action - prv_obs.get(self.action_cols)) * (obs.get(self.action_cols) - action)) / (prv_obs.get(self.action_cols).max())**2).to_numpy().sum(axis=-1, keepdims=True)
        return np.nan_to_num(R, nan=0)

    def step(self, action, obs, obs_nds):
        act_obs = obs.get(self.action_cols).to_numpy()
        scaler = MinMaxScaler((0,1)).fit(act_obs)
        action =  scaler.transform(act_obs) + scaler.transform(act_obs) * action
        action = scaler.inverse_transform(action)
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
                # try:
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
                self.loop.run_until_complete(self.ctrl.setNodeRule(net=int(sinkId.split('.')[0]), sinkId=sinkId, sinkAddr=route[0], dst=route[-1], newVal=entry, path=route))
                # except Exception as ex:
                #     logger.warn(ex)

        obs_tss = obs['ts'].unique().flatten().tolist()# obervation timestamps
        act_tss = np.hstack((obs['ts'].to_numpy().reshape(-1,1), action))
        for i in range(self.hparams.obs_time):
            pred_act = pd.DataFrame(act_tss[np.where(act_tss[:,0] == obs_tss[i])[0], 1:], columns=self.action_cols)
            state, nodes, _, _ = self.networkGraph.getState(nodes=nds, cols=self.action_cols)                
            actual_act = pd.DataFrame(state, columns=self.action_cols)
            # set predicted value in ctrl graph

            # print pred vs actual
            self.tb_logger.add_scalars('Episode/E_Forcast', {
                'E_pred': pred_act['batt'].mean(),
                'E_actl':actual_act['batt'].mean()
                }, global_step= int(time.time() - self.init_ts)
            )
            self.tb_logger.add_scalars('Episode/TX_Forcast', {
                'TX_pred': pred_act['txpackets_val'].mean(),
                'TX_actl':actual_act['txpackets_val'].mean()
                }, global_step= int(time.time() - self.init_ts)
            )
            self.tb_logger.add_scalars('Episode/TX_bytes_Forcast', {
                'TX_bytes_pred': pred_act['txbytes_val'].mean(),
                'TX_bytes_actl':actual_act['txbytes_val'].mean()
                }, global_step= int(time.time() - self.init_ts)
            )
            self.tb_logger.add_scalars('Episode/RX_Forcast', {
                'RX_pred': pred_act['rxpackets_val'].mean(),
                'RX_actl':actual_act['rxpackets_val'].mean()
                }, global_step= int(time.time() - self.init_ts)
            )
            self.tb_logger.add_scalars('Episode/RX_bytes_Forcast', {
                'RX_bytes_pred': pred_act['rxbytes_val'].mean(),
                'RX_bytes_actl':actual_act['rxbytes_val'].mean()
                }, global_step= int(time.time() - self.init_ts)
            )
            time.sleep(1)
        
        next_obs, _, next_sink_obs, _ = self.reset(nds)            
        reward = self.getReward(next_obs, obs, pd.DataFrame(action, columns=self.action_cols))
        done = np.zeros((next_obs.shape[0],1), dtype=bool) #TODO change the logic when to set done to True (since it is a continouse process target optimization are always changing as per the network progress and resource drained)           
        info = np.empty((next_obs.shape[0],1), dtype=str)
        return next_obs, reward, done, info, next_sink_obs
    
    @T.no_grad()
    def play_episodes(self, policy=None):
        obs, nodes, _, _ = self.reset()
        print('NSFP COLLECTING DATA ....')     
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            log_prob, action = self.policy(obs.to_numpy())
            action = action.detach().cpu().numpy()
            log_prob = log_prob.detach().cpu().numpy()
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info, nxt_sink_obs  = self.step(action, obs, nodes)
            self.buffer.append((obs.to_numpy(), log_prob, action, reward, done, nxt_obs.to_numpy()))
            self.num_samples.append(nxt_obs.shape[0])

            # pd.concat([pd.DataFrame(nodes, columns=['node']),
            #     obs, 
            #     pd.DataFrame(action, columns=self.action_cols),
            #     nxt_obs, 
            #     pd.DataFrame(reward, columns=['reward']),
            #     pd.DataFrame(done, columns=['done']),
            #     pd.DataFrame(info, columns=['info'])],
            #     axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))

            self.pltMetrics(reward, nodes, obs, nxt_obs, nxt_sink_obs)  
            obs = nxt_obs            
            self.ep_step += 1

    def pltMetrics(self, reward, nodes, obs, nxt_obs, nxt_sink_obs):
        self.tb_logger.add_scalars('Episode/R', {
            'R': reward.sum()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/D', {
            'D': nxt_obs['delay'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/T', {
            'T': nxt_obs['throughput'].mean() 
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/E', {
            'E': nxt_obs['batt'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/DP', {
            'DP': nxt_obs['drpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/TX', {
            'TX': nxt_obs['txpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RX', {
            'RX': nxt_obs['rxpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/TX_in', {
            'TX_in': nxt_sink_obs['txpacketsin_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RX_out', {
            'RX_out': nxt_sink_obs['rxpacketsout_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT', {
            'RT': nxt_obs['rptti'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT_var', {
            'RT_var': nxt_obs['rptti'].var()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/EC', {
            'EC': ((obs['batt'] - nxt_obs['batt'])/(nxt_obs['ts'] - obs['ts'])).mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/EC_var', {
            'EC_var': ((obs['batt'] - nxt_obs['batt'])/(nxt_obs['ts'] - obs['ts'])).var()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/E_nds', 
            dict([(str(nodes[nd].item()), nxt_obs['batt'].to_numpy()[nd]) for nd in range(nodes.shape[0])]),
            global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT_nds', 
            dict([(str(nodes[nd].item()), nxt_obs['rptti'].to_numpy()[nd]) for nd in range(nodes.shape[0])]),
            global_step=self.ep_step
        )
        # self.tb_logger.add_scalars('Episode/Value/Loss', {
        #     'Value_Loss': np.array(self.ep_value_loss).mean()
        #     }, global_step=self.ep_step
        # )
        # self.tb_logger.add_scalars('Episode/Policy/Loss', {
        #     'Policy_Loss': np.array(self.ep_policy_loss).mean()
        #     }, global_step=self.ep_step
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
        # return self.hparams.optim(self.parameters(), lr=self.hparams.lr)

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

    def training_step(self, batch, batch_idx):
        
        # optimizer = self.optimizers()
        value_opt, policy_opt = self.optimizers()
        obs_b, log_prob_b, action_b, reward_b, done_b, nxt_obs_b = batch        

        # rev_nxt_obs = self.env.scaler.inverse_transform(T.clone(nxt_obs_b.data).detach().cpu().numpy())
        
        state_values = self.value_net(obs_b)

        with T.no_grad():
            _, nxt_action = self.target_policy(nxt_obs_b)
            nxt_state_values = self.target_val_net(nxt_obs_b)
            # nxt_state_values[done_b] = 0.0 
            target = reward_b + self.hparams.gamma * nxt_state_values
        
        advantages = (target - state_values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # value_opt
        value_loss = self.hparams.loss_fn(state_values, target.float())
        value_opt.zero_grad()
        # self.manual_backward(value_loss)
        value_loss.backward()
        T.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
        value_opt.step()
        self.tb_logger.add_scalars('Training/Value/Loss', {'loss': value_loss}, global_step=self.global_step)

        # policy_opt
        log_prob, _ = self.policy(obs_b)
        prv_log_prob = log_prob_b
        
        rho = T.exp(log_prob - prv_log_prob)
        # rho = log_prob / prv_log_prob

        surrogate_1 = rho * advantages
        surrogate_2 = T.clip(rho, 1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

        policy_loss = - T.minimum(surrogate_1, surrogate_2).mean()
        policy_opt.zero_grad()
        # self.manual_backward(value_loss)
        policy_loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        policy_opt.step()

        self.tb_logger.add_scalars('Training/Policy/Loss', {'loss': policy_loss}, global_step=self.global_step)

        # self.log_dict({"value_loss": value_loss.item(), "policy_loss": policy_loss.item(), "return": reward_b.sum()}, prog_bar=True)
                
        # For simplecity:
        #         # Total loss is a combination of critic loss and policy loss
        total_loss = value_loss + policy_loss

        # Perform backpropagation
        # self.manual_backward(total_loss)
        # optimizer.step()
        # optimizer.zero_grad()

        # self.tb_logger.add_scalars('Training/Loss', {'loss': total_loss}, global_step=self.global_step)
        # # Return the total loss for logging purposes
        # self.log_dict({"loss": total_loss, "return": reward_b.sum()}, prog_bar=True)
        return total_loss

    def on_train_epoch_end(self):
        # if self.best_return > 0:
        #     self.policy.load_state_dict(T.load('outputs/logs/best_policy'))
        #     self.value_net.load_state_dict(T.load('outputs/logs/best_value'))
        
        self.target_val_net.load_state_dict(self.value_net.state_dict())            
        self.target_policy.load_state_dict(self.policy.state_dict())
        print(f'NSFP END EPOCH: {self.current_epoch}*****************')
        # time.sleep(1)
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        # try:
        #     quietRun('rm -r outputs/logs/')
        #     # quietRun('rm -r outputs/logs/experiences.csv')
        #     # quietRun('rm -r outputs/logs/lightning_logs/version_0')
        #     # quietRun('rm -r outputs/content/videos/')
        #     # quietRun('tensorboard --logdir outputs/logs/')
        # except Exception as ex:
        #     logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('NSFP START TRAINING ...')
        # self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        quietRun('chmod -R 777 outputs/logs')
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        '''
        checkpoint_callback = ModelCheckpoint(
            dirpath='outputs/logs',
            monitor='return',
            save_top_k=3,
            filename='model-{epoch:02d}-{return:.2f}',
            mode='max',
        )
        '''
        # checkpoint_callback = ModelCheckpoint(dirpath='outputs/logs')

        trainer = Trainer(
            accelerator='auto',      
            devices=num_devices,
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            # callbacks=[checkpoint_callback],
            # logger=self.tb_logger,
            reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        T.autograd.detect_anomaly(True)
        self.play_episodes()
        trainer.fit(self)

class PPO_MP_ATCNSF(LightningModule):
    """ Network State Forcasting On-Policy PPO_DRL Agent
    """
    def __init__(self, ctrl, num_envs=1, batch_size=2, obs_time = 20, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, lr=1e-3, gamma=0.99, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.automatic_optimization = False
        self.ctrl = ctrl
        self.networkGraph = self.ctrl.networkGraph
        self.obs_cols = ['port', 'intftypeval', 'datatypeval', 'distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', \
        'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val', 'rptti'] 
        self.cal_cols = ['ts']
        self.action_pred_cols = ['batt', 'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val']
        self.action_pred_space = np.empty((0, len(self.action_pred_cols)))
        self.action_optm_cols = ['isaggr', 'nxhop', 'rptti']
        self.action_optm_space = np.empty((0, len(self.action_optm_cols)))

        self.observation_space = np.empty((0, len(self.obs_cols)+len(self.cal_cols)))

        self.obs_dims = self.observation_space.shape[1]
        self.action_pred_dims = self.action_pred_space.shape[1]
        self.action_optm_dims = self.action_optm_space.shape[1]
        self.obs_nds = None
        self.policy_pred = PPO_Policy_Pred(self.obs_dims, hidden_size, self.action_pred_dims)
        self.target_policy_pred = copy.deepcopy(self.policy_pred)
        self.policy_optm = PPO_Policy(self.obs_dims, hidden_size, self.action_optm_dims)
        self.target_policy_optm = copy.deepcopy(self.policy_optm)
        self.value_net_pred = PPO_ValueNet(self.obs_dims, hidden_size)
        self.target_val_net_pred = copy.deepcopy(self.value_net_pred)
        self.value_net_optm = PPO_ValueNet(self.obs_dims, hidden_size)
        self.target_val_net_optm = copy.deepcopy(self.value_net_optm)

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.ep_step = 0
        self.tb_logger = None
        self.init_ts = time.time()
        
        self.save_hyperparameters('batch_size', 'obs_time', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'lr', 'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')
        
    def reset(self, nds=None):
        """Get network state observation for certain amount of time"""
        obs_time = self.hparams.obs_time
        obs = self.observation_space
        _nodes = np.empty((0,1))
        sink_obs = np.empty((0, len(self.obs_cols)))
        _sink_nodes = np.empty((0,1))
        for _ in range(obs_time):
            state, nodes, sink_state, sink_state_nodes = self.networkGraph.getState(nodes=nds, cols=self.obs_cols)                
            obs = np.vstack((obs, np.column_stack((state, np.repeat(int(time.time()), state.shape[0], axis=0).reshape(-1,1)))))
            _nodes = np.vstack((_nodes, nodes))
            sink_obs = np.vstack((sink_obs, sink_state))
            _sink_nodes = np.vstack((_sink_nodes, sink_state_nodes))
            nds = nodes
            time.sleep(1)

        data = pd.concat([pd.DataFrame(obs, columns=self.obs_cols+self.cal_cols, dtype=float)], axis=1)
        sink_state = pd.DataFrame(sink_obs, columns=self.obs_cols, dtype=float)
        return data, _nodes, sink_state, _sink_nodes
    
    def getReward_Pred(self, obs, prv_obs, action):
        R = (((action - prv_obs.get(self.action_pred_cols)) * (obs.get(self.action_pred_cols) - action)) / (prv_obs.get(self.action_pred_cols).max())**2).to_numpy().sum(axis=-1, keepdims=True)
        return np.nan_to_num(R, nan=0)
    
    def getReward_optm(self, obs, prv_obs):
        obs_ts = obs['ts'].mean()
        prv_obs_ts = prv_obs['ts'].mean()
        prv_E = prv_obs['batt'].to_numpy().reshape(-1,1)

        TH = obs['throughput'].to_numpy().reshape(-1,1)
        D = obs['delay'].to_numpy().reshape(-1,1)
        E = obs['batt'].to_numpy().reshape(-1,1)
        EC = ((prv_E - E)/(obs_ts - prv_obs_ts)).reshape(-1,1)
        RT = obs['rptti'].to_numpy().reshape(-1,1)
        # R = TH/ct.MAX_BANDWIDTH + 4 - D/ct.MAX_DELAY - EC/ct.MAX_ENRCONS - (EC/ct.MAX_ENRCONS).var() - (RT/ct.MAX_RP_TTI).var()
        R = TH/ct.MAX_BANDWIDTH + 3 - D/ct.MAX_DELAY - EC/ct.MAX_ENRCONS - (EC/ct.MAX_ENRCONS).var()
        
        return np.nan_to_num(R, nan=0)
    
    def step(self, action_pred, action_optm, obs, obs_nds):
        try:
            # Calculate mean of action_optm per unique value in obs_nds
            unique_nds = np.unique(obs_nds, axis=0)
            mean_actions = np.zeros((len(unique_nds), action_optm.shape[1]))

            for i, nd in enumerate(unique_nds):
                selected_rows = action_optm[np.where(np.all(obs_nds == nd, axis=1))]
                mean_actions[i] = np.mean(selected_rows, axis=0)

            # Normalize mean_actions
            mean_actions = MinMaxScaler((0, 1)).fit_transform(mean_actions)

            # Concatenate mean_actions with unique_nds
            action_nd_optm = np.hstack((unique_nds, mean_actions))

            act_pred_obs = obs.get(self.action_pred_cols).to_numpy()
            scaler = MinMaxScaler((0,1)).fit(act_pred_obs)
            action_pred =  scaler.transform(act_pred_obs) + scaler.transform(act_pred_obs) * action_pred
            action_pred = scaler.inverse_transform(action_pred)

            for nd in unique_nds.flatten().tolist():
                # selected_rows = action_nd_optm[np.where(obs_nds[:, 0] == nd)[0]]
                neighbors = [edge[1] for edge in list(self.networkGraph.getGraph().edges(nbunch=nd, data=True, keys=True))]
                nd_pos = self.networkGraph.getPosition(nd)
                src_dist = self.networkGraph.getDistance(nd)
                # act_angle = np.mean(selected_rows, axis=0)[2] * pi # angle between 0 and π radians 
                act_angle = action_nd_optm[np.where((action_nd_optm[:, 0] == nd))[0], 2].astype(float) * pi # angle between 0 and π radians 
                act_nxh_node = nd
                vl = pi
                for nr in neighbors:
                    dst_dist = self.networkGraph.getDistance(nr)
                    nd_nr_edge = self.networkGraph.getEdge(nd, nr)
                    nr_pos = self.networkGraph.getPosition(nr)
                    y = nr_pos[1] - nd_pos[1]
                    x = nr_pos[0] - nd_pos[0]
                    angle = atan2(y, x)
                    angle = angle if angle > 0 else (angle + (2*pi))
                    act_nxh_val = abs(act_angle - angle)
                    if act_nxh_val < vl and src_dist > dst_dist:
                        vl = act_nxh_val
                        act_nxh_node = nr

                # action value
                val = int.to_bytes(ct.DRL_AG_INDEX, 1, 'big', signed=False)+int.to_bytes(1 if action_nd_optm[np.where((action_nd_optm[:, 0] == nd))[0], 1].astype(float) > 0.50 else 0, ct.DRL_AG_LEN, 'big', signed=False)+\
                    int.to_bytes(ct.DRL_NH_INDEX, 1, 'big', signed=False)+int.to_bytes(Addr(re.sub(r'^.*?.', '', act_nxh_node)[1:]).intValue(), ct.DRL_NH_LEN, 'big', signed=False)+\
                    int.to_bytes(ct.DRL_RT_INDEX, 1, 'big', signed=False)+int.to_bytes(int(action_nd_optm[np.where((action_nd_optm[:, 0] == nd))[0], 3].astype(float) * (ct.MAX_RP_TTI-ct.MIN_RP_TTI))+ct.MIN_RP_TTI, ct.DRL_RT_LEN, 'big', signed=False)+\
                    int.to_bytes(ct.DRL_DR_INDEX, 1, 'big', signed=False)+int.to_bytes(self.hparams.obs_time, ct.DRL_DR_LEN, 'big', signed=False)

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
                    self.loop.run_until_complete(self.ctrl.setDRLAction(net=int(sinkId.split('.')[0]), sinkId=sinkId, sinkAddr=route[0], dst=route[-1], newVal=val, path=route))

            obs_tss = obs['ts'].unique().flatten().tolist()# obervation timestamps
            act_tss = np.hstack((obs['ts'].to_numpy().reshape(-1,1), action_pred))
            for i in range(self.hparams.obs_time):
                pred_act = pd.DataFrame(act_tss[np.where(act_tss[:,0] == obs_tss[i])[0], 1:], columns=self.action_pred_cols)
                state, nodes, _, _ = self.networkGraph.getState(nodes=unique_nds, cols=self.action_pred_cols)                
                actual_act = pd.DataFrame(state, columns=self.action_pred_cols)
                # set predicted value in ctrl graph

                # print pred vs actual
                self.tb_logger.add_scalars('Episode/E_Forcast', {
                    'E_pred': pred_act['batt'].mean(),
                    'E_actl':actual_act['batt'].mean()
                    }, global_step= int(time.time() - self.init_ts)
                )
                self.tb_logger.add_scalars('Episode/TX_Forcast', {
                    'TX_pred': pred_act['txpackets_val'].mean(),
                    'TX_actl':actual_act['txpackets_val'].mean()
                    }, global_step= int(time.time() - self.init_ts)
                )
                self.tb_logger.add_scalars('Episode/TX_bytes_Forcast', {
                    'TX_bytes_pred': pred_act['txbytes_val'].mean(),
                    'TX_bytes_actl':actual_act['txbytes_val'].mean()
                    }, global_step= int(time.time() - self.init_ts)
                )
                self.tb_logger.add_scalars('Episode/RX_Forcast', {
                    'RX_pred': pred_act['rxpackets_val'].mean(),
                    'RX_actl':actual_act['rxpackets_val'].mean()
                    }, global_step= int(time.time() - self.init_ts)
                )
                self.tb_logger.add_scalars('Episode/RX_bytes_Forcast', {
                    'RX_bytes_pred': pred_act['rxbytes_val'].mean(),
                    'RX_bytes_actl':actual_act['rxbytes_val'].mean()
                    }, global_step= int(time.time() - self.init_ts)
                )
                time.sleep(1)
            
            next_obs, _, next_sink_obs, _ = self.reset(unique_nds)            
            reward_pred = self.getReward_Pred(next_obs, obs, pd.DataFrame(action_pred, columns=self.action_pred_cols))
            reward_optm = self.getReward_optm(next_obs, obs)
            done = np.zeros((next_obs.shape[0],1), dtype=bool) #TODO change the logic when to set done to True (since it is a continouse process target optimization are always changing as per the network progress and resource drained)           
            info = np.empty((next_obs.shape[0],1), dtype=str)
        except Exception as ex:
            print(ex)
            exc_type, exc_obj, exc_tb = exc_info()
            fname = path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        return next_obs, reward_pred, reward_optm, done, info, next_sink_obs
    
    @T.no_grad()
    def play_episodes(self, policy=None):
        obs, nodes, _, _ = self.reset()
        print('COLLECTING DATA ....')     
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            log_prob_pred, action_pred = self.policy_pred(obs.to_numpy())
            action_pred = action_pred.detach().cpu().numpy()
            log_prob_pred = log_prob_pred.detach().cpu().numpy()
            log_prob_optm, action_optm = self.policy_optm(obs.to_numpy())
            action_optm = action_optm.detach().cpu().numpy()
            log_prob_optm = log_prob_optm.detach().cpu().numpy()
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward_pred, reward_optm, done, info, nxt_sink_obs  = self.step(action_pred, action_optm, obs, nodes)
            self.buffer.append((obs.to_numpy(), log_prob_pred, log_prob_optm, action_pred, action_optm, reward_pred, reward_optm, done, nxt_obs.to_numpy()))
            self.num_samples.append(nxt_obs.shape[0])

            # pd.concat([pd.DataFrame(nodes, columns=['node']),
            #     obs, 
            #     pd.DataFrame(action, columns=self.action_cols),
            #     nxt_obs, 
            #     pd.DataFrame(reward, columns=['reward']),
            #     pd.DataFrame(done, columns=['done']),
            #     pd.DataFrame(info, columns=['info'])],
            #     axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))

            self.pltMetrics(reward_pred, reward_optm, nodes, obs, nxt_obs, nxt_sink_obs)  
            obs = nxt_obs            
            self.ep_step += 1

    def pltMetrics(self, reward_pred, reward_optm, nodes, obs, nxt_obs, nxt_sink_obs):
        self.tb_logger.add_scalars('Episode/R', {
            'R': reward_pred.sum() + reward_optm.sum()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/D', {
            'D': nxt_obs['delay'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/T', {
            'T': nxt_obs['throughput'].mean() 
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/E', {
            'E': nxt_obs['batt'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/DP', {
            'DP': nxt_obs['drpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/TX', {
            'TX': nxt_obs['txpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RX', {
            'RX': nxt_obs['rxpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/TX_in', {
            'TX_in': nxt_sink_obs['txpacketsin_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RX_out', {
            'RX_out': nxt_sink_obs['rxpacketsout_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT', {
            'RT': nxt_obs['rptti'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT_var', {
            'RT_var': nxt_obs['rptti'].var()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/EC', {
            'EC': ((obs['batt'] - nxt_obs['batt'])/(nxt_obs['ts'] - obs['ts'])).mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/EC_var', {
            'EC_var': ((obs['batt'] - nxt_obs['batt'])/(nxt_obs['ts'] - obs['ts'])).var()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/E_nds', 
            dict([(str(nodes[nd].item()), nxt_obs['batt'].to_numpy()[nd]) for nd in range(nodes.shape[0])]),
            global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT_nds', 
            dict([(str(nodes[nd].item()), nxt_obs['rptti'].to_numpy()[nd]) for nd in range(nodes.shape[0])]),
            global_step=self.ep_step
        )
        # self.tb_logger.add_scalars('Episode/Value/Loss', {
        #     'Value_Loss': np.array(self.ep_value_loss).mean()
        #     }, global_step=self.ep_step
        # )
        # self.tb_logger.add_scalars('Episode/Policy/Loss', {
        #     'Policy_Loss': np.array(self.ep_policy_loss).mean()
        #     }, global_step=self.ep_step
        # )

    def _dataset(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_pred_b, log_prob_optm_b, action_pred_b, action_optm_b, reward_pred_b, reward_optm_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(sum(self.num_samples)):
            yield obs_b[i], log_prob_pred_b[i], log_prob_optm_b[i], action_pred_b[i], action_optm_b[i], reward_pred_b[i], reward_optm_b[i], done_b[i], nxt_obs_b[i]

    def _dataset_shuffle(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, log_prob_pred_b, log_prob_optm_b, action_pred_b, action_optm_b, reward_pred_b, reward_optm_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.hparams.epoch_repeat):
            idx = list(range(sum(self.num_samples)))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], log_prob_pred_b[i], log_prob_optm_b[i], action_pred_b[i], action_optm_b[i], reward_pred_b[i], reward_optm_b[i], done_b[i], nxt_obs_b[i]

    def configure_optimizers(self):
        value_pred_opt = self.hparams.optim(self.value_net_pred.parameters(), lr=self.hparams.value_lr)
        value_optm_opt = self.hparams.optim(self.value_net_optm.parameters(), lr=self.hparams.value_lr)
        policy_pred_opt = self.hparams.optim(self.policy_pred.parameters(), lr=self.hparams.policy_lr)  
        policy_optm_opt = self.hparams.optim(self.policy_optm.parameters(), lr=self.hparams.policy_lr)   
        return value_pred_opt, value_optm_opt, policy_pred_opt, policy_optm_opt
        # return self.hparams.optim(self.parameters(), lr=self.hparams.lr)

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

    def training_step(self, batch, batch_idx):
        
        # optimizer = self.optimizers()
        value_pred_opt, value_optm_opt, policy_pred_opt, policy_optm_opt = self.optimizers()
        obs_b, log_prob_pred_b, log_prob_optm_b, action_pred_b, action_optm_b, reward_pred_b, reward_optm_b, done_b, nxt_obs_b = batch        

        # rev_nxt_obs = self.env.scaler.inverse_transform(T.clone(nxt_obs_b.data).detach().cpu().numpy())
        
        state_values_pred = self.value_net_pred(obs_b)
        state_values_optm = self.value_net_optm(obs_b)

        with T.no_grad():
            _, nxt_action_pred = self.target_policy_pred(nxt_obs_b)
            _, nxt_action_optm = self.target_policy_optm(nxt_obs_b)
            nxt_state_values_pred = self.target_val_net_pred(nxt_obs_b)
            nxt_state_values_optm = self.target_val_net_optm(nxt_obs_b)
            # nxt_state_values[done_b] = 0.0 
            target_pred = reward_pred_b + self.hparams.gamma * nxt_state_values_pred
            target_optm = reward_optm_b + self.hparams.gamma * nxt_state_values_optm
        
        advantages_pred = (target_pred - state_values_pred).detach()
        advantages_pred = (advantages_pred - advantages_pred.mean()) / (advantages_pred.std() + 1e-8)

        advantages_optm = (target_optm - state_values_optm).detach()
        advantages_optm = (advantages_optm - advantages_optm.mean()) / (advantages_optm.std() + 1e-8)

        # value_pred_opt
        value_pred_loss = self.hparams.loss_fn(state_values_pred, target_pred.float())
        value_pred_opt.zero_grad()
        # self.manual_backward(value_loss)
        value_pred_loss.backward()
        T.nn.utils.clip_grad_norm_(self.value_net_pred.parameters(), max_norm=0.5)
        value_pred_opt.step()
        # value_optm_opt
        value_optm_loss = self.hparams.loss_fn(state_values_optm, target_optm.float())
        value_optm_opt.zero_grad()
        # self.manual_backward(value_loss)
        value_optm_loss.backward()
        T.nn.utils.clip_grad_norm_(self.value_net_optm.parameters(), max_norm=0.5)
        value_optm_opt.step()

        self.tb_logger.add_scalars('Training/Value/Loss', {'loss': value_pred_loss+value_optm_loss}, global_step=self.global_step)

        # policy_pred_opt
        log_prob_pred, _ = self.policy_pred(obs_b)
        prv_log_prob_pred = log_prob_pred_b
        
        rho_pred = T.exp(log_prob_pred - prv_log_prob_pred)
        # rho = log_prob / prv_log_prob

        surrogate_1_pred = rho_pred * advantages_pred
        surrogate_2_pred = T.clip(rho_pred, 1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages_pred

        policy_pred_loss = - T.minimum(surrogate_1_pred, surrogate_2_pred).mean()
        policy_pred_opt.zero_grad()
        # self.manual_backward(value_loss)
        policy_pred_loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy_pred.parameters(), max_norm=0.5)
        policy_pred_opt.step()

        # policy_optm_opt
        log_prob_optm, _ = self.policy_optm(obs_b)
        prv_log_prob_optm = log_prob_optm_b
        
        rho_optm = T.exp(log_prob_optm - prv_log_prob_optm)
        # rho = log_prob / prv_log_prob

        surrogate_1_optm = rho_optm * advantages_optm
        surrogate_2_optm = T.clip(rho_optm, 1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages_optm

        policy_optm_loss = - T.minimum(surrogate_1_optm, surrogate_2_optm).mean()
        policy_optm_opt.zero_grad()
        # self.manual_backward(value_loss)
        policy_optm_loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy_optm.parameters(), max_norm=0.5)
        policy_optm_opt.step()

        self.tb_logger.add_scalars('Training/Policy/Loss', {'loss': policy_pred_loss+policy_optm_loss}, global_step=self.global_step)

        # self.log_dict({"value_loss": value_loss.item(), "policy_loss": policy_loss.item(), "return": reward_b.sum()}, prog_bar=True)
                
        # For simplecity:
        #         # Total loss is a combination of critic loss and policy loss
        total_loss = value_pred_loss + value_optm_loss + policy_pred_loss + policy_optm_loss

        # Perform backpropagation
        # self.manual_backward(total_loss)
        # optimizer.step()
        # optimizer.zero_grad()

        # self.tb_logger.add_scalars('Training/Loss', {'loss': total_loss}, global_step=self.global_step)
        # # Return the total loss for logging purposes
        # self.log_dict({"loss": total_loss, "return": reward_b.sum()}, prog_bar=True)
        return total_loss

    def on_train_epoch_end(self):
        # if self.best_return > 0:
        #     self.policy.load_state_dict(T.load('outputs/logs/best_policy'))
        #     self.value_net.load_state_dict(T.load('outputs/logs/best_value'))
        
        self.target_val_net_pred.load_state_dict(self.value_net_pred.state_dict())            
        self.target_val_net_optm.load_state_dict(self.value_net_optm.state_dict())            
        self.target_policy_pred.load_state_dict(self.policy_pred.state_dict())
        self.target_policy_optm.load_state_dict(self.policy_optm.state_dict())
        print(f'NSFP END EPOCH: {self.current_epoch}*****************')
        # time.sleep(1)
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        # try:
        #     quietRun('rm -r outputs/logs/')
        #     # quietRun('rm -r outputs/logs/experiences.csv')
        #     # quietRun('rm -r outputs/logs/lightning_logs/version_0')
        #     # quietRun('rm -r outputs/content/videos/')
        #     # quietRun('tensorboard --logdir outputs/logs/')
        # except Exception as ex:
        #     logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('NSFP START TRAINING ...')
        # self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        quietRun('chmod -R 777 outputs/logs')
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        '''
        checkpoint_callback = ModelCheckpoint(
            dirpath='outputs/logs',
            monitor='return',
            save_top_k=3,
            filename='model-{epoch:02d}-{return:.2f}',
            mode='max',
        )
        '''
        # checkpoint_callback = ModelCheckpoint(dirpath='outputs/logs')

        trainer = Trainer(
            accelerator='auto',      
            devices=num_devices,
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            # callbacks=[checkpoint_callback],
            # logger=self.tb_logger,
            reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        T.autograd.detect_anomaly(True)
        self.play_episodes()
        try:
            trainer.fit(self)
        except Exception as ex:
            print(ex)
            exc_type, exc_obj, exc_tb = exc_info()
            fname = path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

class RPLS(LightningModule):
    """ Adaptive Traffic Controll Off-Policy RPLS Agent
    """
    def __init__(self, ctrl, num_envs=1, batch_size=2, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, lr=1e-3, gamma=0.99, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=Adam):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.automatic_optimization = False
        self.ctrl = ctrl
        self.networkGraph = self.ctrl.networkGraph

        self.obs_cols = ['port', 'intftypeval', 'datatypeval', 'distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', \
        'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val', 'rptti'] 
        self.cal_cols = ['ts']
        self.action_cols = ['isagg', 'nxhop', 'rptti']
        self.action_space = np.empty((0, len(self.action_cols)))
        self.observation_space = np.empty((0, len(self.obs_cols)+len(self.cal_cols)))

        self.obs_dims = self.observation_space.shape[1]
        self.action_dim = self.action_space.shape[1]
        
        self.obs_nds = None
        # DQN model and target model
        self.q_network = DQN(self.obs_dims, hidden_size, self.action_dim)
        self.target_q_network = DQN(self.obs_dims, hidden_size, self.action_dim)

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.ep_return = []
        self.best_return = 0
        self.ep_step = 0

        self.save_hyperparameters('batch_size', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'lr', 'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')

    def reset(self, nds=None):
        """Get network state observation"""
        state, nodes, sink_state, sink_state_nodes = self.networkGraph.getState(nodes=nds, cols=self.obs_cols)  
        obs = np.column_stack((state, np.repeat(int(time.time()), state.shape[0], axis=0).reshape(-1,1)))
        
        data = pd.concat([pd.DataFrame(obs, columns=self.obs_cols+self.cal_cols, dtype=float)], axis=1)
        sink_state = pd.DataFrame(sink_state, columns=self.obs_cols, dtype=float)
        return data, nodes, sink_state, sink_state_nodes
    
    def getReward(self, obs, prv_obs):
        DS = 1 - (obs['distance'].to_numpy().reshape(-1,1)/ct.DIST_MAX)
        DE = 1 - (obs['denisty'].to_numpy().reshape(-1,1)/ct.MAX_NEIG)
        E = obs['batt'].to_numpy().reshape(-1,1)/ct.BATT_LEVEL
        WL = obs['throughput'].to_numpy().reshape(-1,1)/ct.MAX_BANDWIDTH
        R = DS+E+DE+WL
        return np.nan_to_num(R, nan=0)
    
    def step(self, action, obs, obs_nds):
        action = MinMaxScaler((0,1)).fit_transform(action)
        act_nds = dict(zip(obs_nds.flatten(), action))

        isaggr_idxs = np.where((action[:,0] > 0.50))
        isaggr_action_nodes = obs_nds[isaggr_idxs,:].flatten().tolist()
        clustering_action = {}
        routing_action = {}
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

                #TODO
                if nd in isaggr_action_nodes:
                    entry = Entry()
                    entry.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1)
                        .setLhsOperandType(ct.PACKET).setLhs(ct.SRC_INDEX).setRhsOperandType(ct.CONST)
                        .setRhs(Addr(re.sub(r'^.*?.', '', nr)[1:]).intValue()))
                    # entry.addWindow(Window.fromString("P.TYP == 2"))
                    entry.addAction(ForwardUnicastAction(nxtHop=Addr(re.sub(r'^.*?.', '', nd)[1:])))
                    entry.getStats().setTtl(int(time.time()))
                    entry.getStats().setIdle(20)
                    clustering_action[nd] = entry

            # action value
            val = int.to_bytes(ct.DRL_AG_INDEX, 1, 'big', signed=False)+int.to_bytes(1 if nd in isaggr_action_nodes else 0, ct.DRL_AG_LEN, 'big', signed=False)+\
                int.to_bytes(ct.DRL_NH_INDEX, 1, 'big', signed=False)+int.to_bytes(Addr(re.sub(r'^.*?.', '', act_nxh_node)[1:]).intValue(), ct.DRL_NH_LEN, 'big', signed=False)+\
                int.to_bytes(ct.DRL_RT_INDEX, 1, 'big', signed=False)+int.to_bytes(int(act[2] * (ct.MAX_RP_TTI-ct.MIN_RP_TTI))+ct.MIN_RP_TTI, ct.DRL_RT_LEN, 'big', signed=False)
            routing_action[nd] = val

        # send the action to the data-plane
        for nd, _ in act_nds.items():
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
                #routing
                if routing_action.get(nd):
                    self.loop.run_until_complete(self.ctrl.setDRLAction(net=int(sinkId.split('.')[0]), sinkId=sinkId, sinkAddr=route[0], dst=route[-1], newVal=routing_action[nd], path=route))
                #clustering
                if clustering_action.get(nd):
                    self.loop.run_until_complete(self.ctrl.setNodeRule(net=int(sinkId.split('.')[0]), sinkId=sinkId, sinkAddr=route[0], dst=route[-1], newVal=clustering_action[nd], path=route))

        time.sleep(15)
        next_obs, _, next_sink_obs, _ = self.reset(obs_nds)
        reward = self.getReward(next_obs, obs)
        done = np.zeros((next_obs.shape[0],1), dtype=bool) #TODO change the logic when to set done to True (since it is a continouse process target optimization are always changing as per the network progress and resource drained)           
        info = np.empty((next_obs.shape[0],1), dtype=str)
        
        return next_obs, reward, done, info, next_sink_obs

    @T.no_grad()
    def play_episodes(self, policy=None):
        obs, nodes, _, _ = self.reset()
        print('COLLECTING DATA ....')     
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            _, action = self.q_network(obs.to_numpy())
            action = action.detach().cpu().numpy()
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info, nxt_sink_obs  = self.step(action, obs, nodes)
            print('reach here ..')
            self.buffer.append((obs.to_numpy(), action, reward, done, nxt_obs.to_numpy()))
            self.num_samples.append(nxt_obs.shape[0])

            # pd.concat([pd.DataFrame(nodes, columns=['node']),
            #     obs, 
            #     pd.DataFrame(action, columns=self.action_cols),
            #     nxt_obs, 
            #     pd.DataFrame(reward, columns=['reward']),
            #     pd.DataFrame(done, columns=['done']),
            #     pd.DataFrame(info, columns=['info'])],
            #     axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            self.pltMetrics(reward, nodes, obs, nxt_obs, nxt_sink_obs)    
            obs = nxt_obs            
            self.ep_step += 1

    def pltMetrics(self, reward, nodes, obs, nxt_obs, nxt_sink_obs):
        if self.ep_step == 0 and self.global_step == 0:
            self.tb_logger.add_scalars('Episode/R', {
                'R': np.zeros(1)
                }, global_step=self.ep_step
            )
        else:
            self.tb_logger.add_scalars('Episode/R', {
                'R': reward.sum()
                }, global_step=self.ep_step
            )
        self.tb_logger.add_scalars('Episode/D', {
            'D': nxt_obs['delay'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/T', {
            'T': nxt_obs['throughput'].mean() 
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/E', {
            'E': nxt_obs['batt'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/DP', {
            'DP': nxt_obs['drpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/TX', {
            'TX': nxt_obs['txpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RX', {
            'RX': nxt_obs['rxpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/TX_in', {
            'TX_in': nxt_sink_obs['txpacketsin_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RX_out', {
            'RX_out': nxt_sink_obs['rxpacketsout_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT', {
            'RT': nxt_obs['rptti'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT_var', {
            'RT_var': nxt_obs['rptti'].var()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/EC', {
            'EC': ((obs['batt'] - nxt_obs['batt'])/(nxt_obs['ts'] - obs['ts'])).mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/EC_var', {
            'EC_var': ((obs['batt'] - nxt_obs['batt'])/(nxt_obs['ts'] - obs['ts'])).var()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/E_nds', 
            dict([(str(nodes[nd].item()), nxt_obs['batt'].to_numpy()[nd]) for nd in range(nodes.shape[0])]),
            global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT_nds', 
            dict([(str(nodes[nd].item()), nxt_obs['rptti'].to_numpy()[nd]) for nd in range(nodes.shape[0])]),
            global_step=self.ep_step
        )
        # self.tb_logger.add_scalars('Episode/Value/Loss', {
        #     'Value_Loss': np.array(self.ep_value_loss).mean()
        #     }, global_step=self.ep_step
        # )
        # self.tb_logger.add_scalars('Episode/Policy/Loss', {
        #     'Policy_Loss': np.array(self.ep_policy_loss).mean()
        #     }, global_step=self.ep_step
        # )

    def _dataset(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(sum(self.num_samples)):
            yield obs_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def _dataset_shuffle(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.hparams.epoch_repeat):
            idx = list(range(sum(self.num_samples)))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def configure_optimizers(self):
        # value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        # policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)   
        # return value_opt, policy_opt
        return self.hparams.optim(self.q_network.parameters(), lr=self.hparams.lr)

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

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        states, actions, rewards, dones, next_states = batch 

        # Q-values for current and next states
        q_values, _ = self.q_network(states)
        with T.no_grad():
            target_value, _ = self.target_q_network(next_states)
            # Compute the target Q-values using the Q-learning update rule
            target_q_values = rewards + self.hparams.gamma * target_value

        # Compute the loss using mean squared error
        loss = self.hparams.loss_fn(q_values, target_q_values.float())

        # Perform backpropagation
        optimizer.zero_grad()
        # self.manual_backward(loss)
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)
        optimizer.step()       
        self.tb_logger.add_scalars('Training/Loss', {'loss': loss}, global_step=self.global_step)
        # self.log_dict({"loss": loss, "return": rewards.sum()}, prog_bar=True)
        return loss
                
    def on_train_epoch_end(self):
        # if self.best_return > 0:
        #     self.policy.load_state_dict(T.load('outputs/logs/best_policy'))
        #     self.value_net.load_state_dict(T.load('outputs/logs/best_value'))
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        # try:
        #     quietRun('rm -r outputs/logs/')
        #     # quietRun('rm -r outputs/logs/experiences.csv')
        #     # quietRun('rm -r outputs/logs/lightning_logs/version_0')
        #     # quietRun('rm -r outputs/content/videos/')
        #     # quietRun('tensorboard --logdir outputs/logs/')
        # except Exception as ex:
        #     logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        # self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        quietRun('chmod -R 777 outputs/logs')
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        '''
        checkpoint_callback = ModelCheckpoint(
            dirpath='outputs/logs',
            monitor='return',
            save_top_k=3,
            filename='model-{epoch:02d}-{return:.2f}',
            mode='max',
        )
        '''
        # checkpoint_callback = ModelCheckpoint(dirpath='outputs/logs')

        trainer = Trainer(
            accelerator='auto',      
            devices=num_devices,
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            # callbacks=[checkpoint_callback],
            # logger=self.tb_logger,
            reload_dataloaders_every_n_epochs = 1,              
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        T.autograd.detect_anomaly(True)
        self.play_episodes()
        trainer.fit(self)

class DRLIR(LightningModule):
    """ Adaptive Traffic Controll Off-Policy DRLIR Agent
    """
    def __init__(self, ctrl, num_envs=1, batch_size=2, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, lr=1e-3, gamma=0.99, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=Adam):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.automatic_optimization = False
        self.ctrl = ctrl
        self.networkGraph = self.ctrl.networkGraph

        self.obs_cols = ['port', 'intftypeval', 'datatypeval', 'distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', \
        'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val', 'rptti'] 
        self.cal_cols = ['ts']
        self.action_cols = ['isagg', 'nxhop', 'rptti']
        self.action_space = np.empty((0, len(self.action_cols)))
        self.observation_space = np.empty((0, len(self.obs_cols)+len(self.cal_cols)))

        self.obs_dims = self.observation_space.shape[1]
        self.action_dim = self.action_space.shape[1]
        
        self.obs_nds = None
        # DQN model and target model
        self.q_network = DQN(self.obs_dims, hidden_size, self.action_dim)
        self.target_q_network = DQN(self.obs_dims, hidden_size, self.action_dim)

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.ep_return = []
        self.best_return = 0
        self.ep_step = 0

        self.save_hyperparameters('batch_size', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'lr', 'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')

    def reset(self, nds=None):
        """Get network state observation"""
        state, nodes, sink_state, sink_state_nodes = self.networkGraph.getState(nodes=nds, cols=self.obs_cols)  
        obs = np.column_stack((state, np.repeat(int(time.time()), state.shape[0], axis=0).reshape(-1,1)))
        
        data = pd.concat([pd.DataFrame(obs, columns=self.obs_cols+self.cal_cols, dtype=float)], axis=1)
        sink_state = pd.DataFrame(sink_state, columns=self.obs_cols, dtype=float)
        return data, nodes, sink_state, sink_state_nodes
    
    def getReward(self, obs, prv_obs):
        # L_max = (obs['batt'].to_numpy().reshape(-1,1)/ ((prv_obs['batt'].to_numpy().reshape(-1,1) - obs['batt'].to_numpy().reshape(-1,1))/(obs['ts'].to_numpy().reshape(-1,1) - prv_obs['ts'].to_numpy().reshape(-1,1)))).max() # remining lifetime Li​=E_residual/E(t), Lmax​=min(L1​,L2​,…,Ln​)
        L_max = ((prv_obs['batt'].to_numpy().reshape(-1,1) - obs['batt'].to_numpy().reshape(-1,1))/(obs['ts'].to_numpy().reshape(-1,1) - prv_obs['ts'].to_numpy().reshape(-1,1))).max()
        D_min = obs['delay'].to_numpy().reshape(-1,1).min()
        T_max = obs['throughput'].to_numpy().reshape(-1,1).max()
        R = L_max+D_min+T_max
        R = np.repeat(R, obs.shape[0], axis=0).reshape(-1,1)
        
        # obs_ts = obs['ts'].mean()
        # prv_obs_ts = prv_obs['ts'].mean()
        # prv_E = prv_obs['batt'].to_numpy().reshape(-1,1)

        # TH = obs['throughput'].to_numpy().reshape(-1,1)
        # D = obs['delay'].to_numpy().reshape(-1,1)
        # E = obs['batt'].to_numpy().reshape(-1,1)
        # EC = ((prv_E - E)/(obs_ts - prv_obs_ts)).reshape(-1,1)
        # R = TH/ct.MAX_BANDWIDTH - D/ct.MAX_DELAY - EC/ct.MAX_ENRCONS
        return np.nan_to_num(R, nan=0)
    
    def step(self, action, obs, obs_nds):
        action = MinMaxScaler((0,1)).fit_transform(action)
        act_nds = dict(zip(obs_nds.flatten(), action))

        isaggr_idxs = np.where((action[:,0] > 0.50))
        isaggr_action_nodes = obs_nds[isaggr_idxs,:].flatten().tolist()
        clustering_action = {}
        routing_action = {}
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

                #TODO
                if nd in isaggr_action_nodes:
                    entry = Entry()
                    entry.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1)
                        .setLhsOperandType(ct.PACKET).setLhs(ct.SRC_INDEX).setRhsOperandType(ct.CONST)
                        .setRhs(Addr(re.sub(r'^.*?.', '', nr)[1:]).intValue()))
                    # entry.addWindow(Window.fromString("P.TYP == 2"))
                    entry.addAction(ForwardUnicastAction(nxtHop=Addr(re.sub(r'^.*?.', '', nd)[1:])))
                    entry.getStats().setTtl(int(time.time()))
                    entry.getStats().setIdle(20)
                    clustering_action[nd] = entry

            # action value
            val = int.to_bytes(ct.DRL_AG_INDEX, 1, 'big', signed=False)+int.to_bytes(1 if nd in isaggr_action_nodes else 0, ct.DRL_AG_LEN, 'big', signed=False)+\
                int.to_bytes(ct.DRL_NH_INDEX, 1, 'big', signed=False)+int.to_bytes(Addr(re.sub(r'^.*?.', '', act_nxh_node)[1:]).intValue(), ct.DRL_NH_LEN, 'big', signed=False)+\
                int.to_bytes(ct.DRL_RT_INDEX, 1, 'big', signed=False)+int.to_bytes(int(act[2] * (ct.MAX_RP_TTI-ct.MIN_RP_TTI))+ct.MIN_RP_TTI, ct.DRL_RT_LEN, 'big', signed=False)
            routing_action[nd] = val

        # send the action to the data-plane
        for nd, _ in act_nds.items():
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
                #routing
                if routing_action.get(nd):
                    self.loop.run_until_complete(self.ctrl.setDRLAction(net=int(sinkId.split('.')[0]), sinkId=sinkId, sinkAddr=route[0], dst=route[-1], newVal=routing_action[nd], path=route))
                #clustering
                if clustering_action.get(nd):
                    self.loop.run_until_complete(self.ctrl.setNodeRule(net=int(sinkId.split('.')[0]), sinkId=sinkId, sinkAddr=route[0], dst=route[-1], newVal=clustering_action[nd], path=route))

        time.sleep(15)
        next_obs, _, next_sink_obs, _ = self.reset(obs_nds)
        reward = self.getReward(next_obs, obs)
        done = np.zeros((next_obs.shape[0],1), dtype=bool) #TODO change the logic when to set done to True (since it is a continouse process target optimization are always changing as per the network progress and resource drained)           
        info = np.empty((next_obs.shape[0],1), dtype=str)
        
        return next_obs, reward, done, info, next_sink_obs

    @T.no_grad()
    def play_episodes(self, policy=None):
        obs, nodes, _, _ = self.reset()
        print('COLLECTING DATA ....')     
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            _, action = self.q_network(obs.to_numpy())
            action = action.detach().cpu().numpy()
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info, nxt_sink_obs  = self.step(action, obs, nodes)
            self.buffer.append((obs.to_numpy(), action, reward, done, nxt_obs.to_numpy()))
            self.num_samples.append(nxt_obs.shape[0])

            # pd.concat([pd.DataFrame(nodes, columns=['node']),
            #     obs, 
            #     pd.DataFrame(action, columns=self.action_cols),
            #     nxt_obs, 
            #     pd.DataFrame(reward, columns=['reward']),
            #     pd.DataFrame(done, columns=['done']),
            #     pd.DataFrame(info, columns=['info'])],
            #     axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            self.pltMetrics(reward, nodes, obs, nxt_obs, nxt_sink_obs)    
            obs = nxt_obs            
            self.ep_step += 1

    def pltMetrics(self, reward, nodes, obs, nxt_obs, nxt_sink_obs):
        if self.ep_step == 0 and self.global_step == 0:
            self.tb_logger.add_scalars('Episode/R', {
                'R': np.zeros(1)
                }, global_step=self.ep_step
            )
        else:
            self.tb_logger.add_scalars('Episode/R', {
                'R': reward.sum()
                }, global_step=self.ep_step
            )
        self.tb_logger.add_scalars('Episode/D', {
            'D': nxt_obs['delay'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/T', {
            'T': nxt_obs['throughput'].mean() 
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/E', {
            'E': nxt_obs['batt'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/DP', {
            'DP': nxt_obs['drpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/TX', {
            'TX': nxt_obs['txpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RX', {
            'RX': nxt_obs['rxpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/TX_in', {
            'TX_in': nxt_sink_obs['txpacketsin_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RX_out', {
            'RX_out': nxt_sink_obs['rxpacketsout_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT', {
            'RT': nxt_obs['rptti'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT_var', {
            'RT_var': nxt_obs['rptti'].var()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/EC', {
            'EC': ((obs['batt'] - nxt_obs['batt'])/(nxt_obs['ts'] - obs['ts'])).mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/EC_var', {
            'EC_var': ((obs['batt'] - nxt_obs['batt'])/(nxt_obs['ts'] - obs['ts'])).var()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/E_nds', 
            dict([(str(nodes[nd].item()), nxt_obs['batt'].to_numpy()[nd]) for nd in range(nodes.shape[0])]),
            global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT_nds', 
            dict([(str(nodes[nd].item()), nxt_obs['rptti'].to_numpy()[nd]) for nd in range(nodes.shape[0])]),
            global_step=self.ep_step
        )
        # self.tb_logger.add_scalars('Episode/Value/Loss', {
        #     'Value_Loss': np.array(self.ep_value_loss).mean()
        #     }, global_step=self.ep_step
        # )
        # self.tb_logger.add_scalars('Episode/Policy/Loss', {
        #     'Policy_Loss': np.array(self.ep_policy_loss).mean()
        #     }, global_step=self.ep_step
        # )

    def _dataset(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(sum(self.num_samples)):
            yield obs_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def _dataset_shuffle(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.hparams.epoch_repeat):
            idx = list(range(sum(self.num_samples)))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def configure_optimizers(self):
        # value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        # policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)   
        # return value_opt, policy_opt
        return self.hparams.optim(self.q_network.parameters(), lr=self.hparams.lr)

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

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        states, actions, rewards, dones, next_states = batch 

        # Q-values for current and next states
        q_values, _ = self.q_network(states)
        with T.no_grad():
            target_value, _ = self.target_q_network(next_states)
            # Compute the target Q-values using the Q-learning update rule
            target_q_values = rewards + self.hparams.gamma * target_value

        # Compute the loss using mean squared error
        loss = self.hparams.loss_fn(q_values, target_q_values.float())

        # Perform backpropagation
        optimizer.zero_grad()
        # self.manual_backward(loss)
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)
        optimizer.step()  
        self.tb_logger.add_scalars('Training/Loss', {'loss': loss}, global_step=self.global_step)
        # self.log_dict({"loss": loss, "return": rewards.sum()}, prog_bar=True)
        return loss
                
    def on_train_epoch_end(self):
        # if self.best_return > 0:
        #     self.policy.load_state_dict(T.load('outputs/logs/best_policy'))
        #     self.value_net.load_state_dict(T.load('outputs/logs/best_value'))
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        # try:
        #     quietRun('rm -r outputs/logs/')
        #     # quietRun('rm -r outputs/logs/experiences.csv')
        #     # quietRun('rm -r outputs/logs/lightning_logs/version_0')
        #     # quietRun('rm -r outputs/content/videos/')
        #     # quietRun('tensorboard --logdir outputs/logs/')
        # except Exception as ex:
        #     logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        # self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        quietRun('chmod -R 777 outputs/logs')
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        '''
        checkpoint_callback = ModelCheckpoint(
            dirpath='outputs/logs',
            monitor='return',
            save_top_k=3,
            filename='model-{epoch:02d}-{return:.2f}',
            mode='max',
        )
        '''
        # checkpoint_callback = ModelCheckpoint(dirpath='outputs/logs')

        trainer = Trainer(
            accelerator='auto',      
            devices=num_devices,
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            # callbacks=[checkpoint_callback],
            # logger=self.tb_logger,
            reload_dataloaders_every_n_epochs = 1,              
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        T.autograd.detect_anomaly(True)
        self.play_episodes()
        trainer.fit(self)

class RLSDWSN(LightningModule):
    """ Adaptive Traffic Controll Off-Policy RLSDWSN Agent
    """
    def __init__(self, ctrl, num_envs=1, batch_size=2, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, lr=1e-3, gamma=0.99, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=Adam):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.automatic_optimization = False
        self.ctrl = ctrl
        self.networkGraph = self.ctrl.networkGraph

        self.obs_cols = ['port', 'intftypeval', 'datatypeval', 'distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', \
        'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val', 'rptti'] 
        self.cal_cols = ['ts']
        self.action_cols = ['isagg', 'nxhop', 'rptti']
        self.action_space = np.empty((0, len(self.action_cols)))
        self.observation_space = np.empty((0, len(self.obs_cols)+len(self.cal_cols)))

        self.obs_dims = self.observation_space.shape[1]
        self.action_dim = self.action_space.shape[1]
        
        self.obs_nds = None
        # DQN model and target model
        self.q_network = DQN(self.obs_dims, hidden_size, self.action_dim)
        self.target_q_network = DQN(self.obs_dims, hidden_size, self.action_dim)

        self.buffer = deque(maxlen=samples_per_epoch)
        self.num_samples = deque(maxlen=samples_per_epoch)
        self.ep_value_loss = []
        self.ep_policy_loss = []
        self.ep_entropy = []
        self.ep_return = []
        self.best_return = 0
        self.ep_step = 0

        self.save_hyperparameters('batch_size', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'lr', 'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
            'epoch_repeat', 'epsilon')

    def reset(self, nds=None):
        """Get network state observation"""
        state, nodes, sink_state, sink_state_nodes = self.networkGraph.getState(nodes=nds, cols=self.obs_cols)  
        obs = np.column_stack((state, np.repeat(int(time.time()), state.shape[0], axis=0).reshape(-1,1)))
        
        data = pd.concat([pd.DataFrame(obs, columns=self.obs_cols+self.cal_cols, dtype=float)], axis=1)
        sink_state = pd.DataFrame(sink_state, columns=self.obs_cols, dtype=float)
        return data, nodes, sink_state, sink_state_nodes
    
    def getReward(self, obs, prv_obs):
        D = obs['distance'].to_numpy().reshape(-1,1)/ct.DIST_MAX
        E = obs['batt'].to_numpy().reshape(-1,1)/ct.BATT_LEVEL
        T = obs['throughput'].to_numpy().reshape(-1,1)/ct.MAX_BANDWIDTH

        R = D+E+T
        
        # obs_ts = obs['ts'].mean()
        # prv_obs_ts = prv_obs['ts'].mean()
        # prv_E = prv_obs['batt'].to_numpy().reshape(-1,1)

        # TH = obs['throughput'].to_numpy().reshape(-1,1)
        # D = obs['delay'].to_numpy().reshape(-1,1)
        # E = obs['batt'].to_numpy().reshape(-1,1)
        # EC = ((prv_E - E)/(obs_ts - prv_obs_ts)).reshape(-1,1)
        # R = TH/ct.MAX_BANDWIDTH - D/ct.MAX_DELAY - EC/ct.MAX_ENRCONS
        return np.nan_to_num(R, nan=0)
    
    def step(self, action, obs, obs_nds):
        action = MinMaxScaler((0,1)).fit_transform(action)
        act_nds = dict(zip(obs_nds.flatten(), action))

        isaggr_idxs = np.where((action[:,0] > 0.50))
        isaggr_action_nodes = obs_nds[isaggr_idxs,:].flatten().tolist()
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
            val = int.to_bytes(ct.DRL_AG_INDEX, 1, 'big', signed=False)+int.to_bytes(1 if nd in isaggr_action_nodes else 0, ct.DRL_AG_LEN, 'big', signed=False)+\
                int.to_bytes(ct.DRL_NH_INDEX, 1, 'big', signed=False)+int.to_bytes(Addr(re.sub(r'^.*?.', '', act_nxh_node)[1:]).intValue(), ct.DRL_NH_LEN, 'big', signed=False)+\
                int.to_bytes(ct.DRL_RT_INDEX, 1, 'big', signed=False)+int.to_bytes(int(act[2] * (ct.MAX_RP_TTI-ct.MIN_RP_TTI))+ct.MIN_RP_TTI, ct.DRL_RT_LEN, 'big', signed=False)

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
                self.loop.run_until_complete(self.ctrl.setDRLAction(net=int(sinkId.split('.')[0]), sinkId=sinkId, sinkAddr=route[0], dst=route[-1], newVal=val, path=route))

        time.sleep(15)
        next_obs, _, next_sink_obs, _ = self.reset(obs_nds)
        reward = self.getReward(next_obs, obs)
        done = np.zeros((next_obs.shape[0],1), dtype=bool) #TODO change the logic when to set done to True (since it is a continouse process target optimization are always changing as per the network progress and resource drained)           
        info = np.empty((next_obs.shape[0],1), dtype=str)
        return next_obs, reward, done, info, next_sink_obs

    @T.no_grad()
    def play_episodes(self, policy=None):
        obs, nodes, _, _ = self.reset()
        print('COLLECTING DATA ....')     
        for step in range(self.hparams.samples_per_epoch):
            done = False
            done = np.zeros((1, 1))
            _, action = self.q_network(obs.to_numpy())
            action = action.detach().cpu().numpy()
            # done[not done.all()] = True if self.epoch_cntr % self.hparams.samples_per_epoch == 0 else done
            nxt_obs, reward, done, info, nxt_sink_obs  = self.step(action, obs, nodes)
            self.buffer.append((obs.to_numpy(), action, reward, done, nxt_obs.to_numpy()))
            self.num_samples.append(nxt_obs.shape[0])

            # pd.concat([pd.DataFrame(nodes, columns=['node']),
            #     obs, 
            #     pd.DataFrame(action, columns=self.action_cols),
            #     nxt_obs, 
            #     pd.DataFrame(reward, columns=['reward']),
            #     pd.DataFrame(done, columns=['done']),
            #     pd.DataFrame(info, columns=['info'])],
            #     axis=1).to_csv('outputs/logs/experiences.csv', mode='a', sep='\t', index=False, header=not path.exists('outputs/logs/experiences.csv'))
            self.pltMetrics(reward, nodes, obs, nxt_obs, nxt_sink_obs)    
            obs = nxt_obs            
            self.ep_step += 1

    def pltMetrics(self, reward, nodes, obs, nxt_obs, nxt_sink_obs):
        if self.ep_step == 0 and self.global_step == 0:
            self.tb_logger.add_scalars('Episode/R', {
                'R': np.zeros(1)
                }, global_step=self.ep_step
            )
        else:
            self.tb_logger.add_scalars('Episode/R', {
                'R': reward.sum()
                }, global_step=self.ep_step
            )
        self.tb_logger.add_scalars('Episode/D', {
            'D': nxt_obs['delay'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/T', {
            'T': nxt_obs['throughput'].mean() 
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/E', {
            'E': nxt_obs['batt'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/DP', {
            'DP': nxt_obs['drpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/TX', {
            'TX': nxt_obs['txpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RX', {
            'RX': nxt_obs['rxpackets_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/TX_in', {
            'TX_in': nxt_sink_obs['txpacketsin_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RX_out', {
            'RX_out': nxt_sink_obs['rxpacketsout_val'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT', {
            'RT': nxt_obs['rptti'].mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT_var', {
            'RT_var': nxt_obs['rptti'].var()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/EC', {
            'EC': ((obs['batt'] - nxt_obs['batt'])/(nxt_obs['ts'] - obs['ts'])).mean()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/EC_var', {
            'EC_var': ((obs['batt'] - nxt_obs['batt'])/(nxt_obs['ts'] - obs['ts'])).var()
            }, global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/E_nds', 
            dict([(str(nodes[nd].item()), nxt_obs['batt'].to_numpy()[nd]) for nd in range(nodes.shape[0])]),
            global_step=self.ep_step
        )
        self.tb_logger.add_scalars('Episode/RT_nds', 
            dict([(str(nodes[nd].item()), nxt_obs['rptti'].to_numpy()[nd]) for nd in range(nodes.shape[0])]),
            global_step=self.ep_step
        )
        # self.tb_logger.add_scalars('Episode/Value/Loss', {
        #     'Value_Loss': np.array(self.ep_value_loss).mean()
        #     }, global_step=self.ep_step
        # )
        # self.tb_logger.add_scalars('Episode/Policy/Loss', {
        #     'Policy_Loss': np.array(self.ep_policy_loss).mean()
        #     }, global_step=self.ep_step
        # )

    def _dataset(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(sum(self.num_samples)):
            yield obs_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def _dataset_shuffle(self):
        reshape_fn = lambda x: x.reshape(sum(self.num_samples), -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.hparams.epoch_repeat):
            idx = list(range(sum(self.num_samples)))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

    def configure_optimizers(self):
        # value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        # policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)   
        # return value_opt, policy_opt
        return self.hparams.optim(self.q_network.parameters(), lr=self.hparams.lr)

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

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        states, actions, rewards, dones, next_states = batch 

        # Q-values for current and next states
        q_values, _ = self.q_network(states)
        with T.no_grad():
            target_value, _ = self.target_q_network(next_states)
            # Compute the target Q-values using the Q-learning update rule
            target_q_values = rewards + self.hparams.gamma * target_value

        # Compute the loss using mean squared error
        loss = self.hparams.loss_fn(q_values, target_q_values.float())

        # Perform backpropagation
        optimizer.zero_grad()
        # self.manual_backward(loss)
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)
        optimizer.step()
        self.tb_logger.add_scalars('Training/Loss', {'loss': loss}, global_step=self.global_step)
        # self.log_dict({"loss": loss, "return": rewards.sum()}, prog_bar=True)
        return loss
                
    def on_train_epoch_end(self):
        # if self.best_return > 0:
        #     self.policy.load_state_dict(T.load('outputs/logs/best_policy'))
        #     self.value_net.load_state_dict(T.load('outputs/logs/best_value'))
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        print(f'END EPOCH: {self.current_epoch}*****************')
        self.play_episodes()

    async def run(self):
        # Start tensorboard.
        # try:
        #     quietRun('rm -r outputs/logs/')
        #     # quietRun('rm -r outputs/logs/experiences.csv')
        #     # quietRun('rm -r outputs/logs/lightning_logs/version_0')
        #     # quietRun('rm -r outputs/content/videos/')
        #     # quietRun('tensorboard --logdir outputs/logs/')
        # except Exception as ex:
        #     logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        # self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        quietRun('chmod -R 777 outputs/logs')
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        '''
        checkpoint_callback = ModelCheckpoint(
            dirpath='outputs/logs',
            monitor='return',
            save_top_k=3,
            filename='model-{epoch:02d}-{return:.2f}',
            mode='max',
        )
        '''
        # checkpoint_callback = ModelCheckpoint(dirpath='outputs/logs')

        trainer = Trainer(
            accelerator='auto',      
            devices=num_devices,
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            # callbacks=[checkpoint_callback],
            # logger=self.tb_logger,
            reload_dataloaders_every_n_epochs = 1,              
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        T.autograd.detect_anomaly(True)
        self.play_episodes()
        trainer.fit(self)