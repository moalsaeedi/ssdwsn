import copy
import itertools
from typing import List, Tuple
from os import cpu_count, path
import random
import torch as T
import torch.nn.functional as F
import numpy as np
import pandas as pd
import asyncio
import logging
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
from ssdwsn.app.network import PPO_GradientPolicy, PPO_Policy, PPO_ValueNet, PPO_Policy_Pred, PPO_ValueNet_Pred
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
num_gpus = T.cuda.device_count() if T.cuda.is_available() else 0
device = f'cuda:{num_gpus-1}' if T.cuda.is_available() else 'cpu'

class PPO_ATCP(LightningModule):
    """ Adaptive Traffic Controll On-Policy PPO_DRL Agent
    """
    def __init__(self, ctrl, num_envs=50, batch_size=2, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, gamma=0.99, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=Adam):
        super().__init__()
        self.loop = asyncio.get_running_loop()
        self.ctrl = ctrl
        self.networkGraph = self.ctrl.networkGraph

        self.obs_cols = ['port', 'intftypeval', 'datatypeval', 'distance', 'denisty', 'alinks', 'flinks', 'x', 'y', 'z', 'batt', 'delay', 'throughput', \
        'txpackets_val', 'txbytes_val', 'rxpackets_val', 'rxbytes_val', 'drpackets_val', 'txpacketsin_val', 'txbytesin_val', 'rxpacketsout_val', 'rxbytesout_val', 'rptti'] 
        self.cal_cols = ['ts']
        self.action_cols = ['clhop', 'nxhop', 'rptti']
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

        self.save_hyperparameters('batch_size', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
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

        state_values = self.value_net(obs_b)

        with T.no_grad():
            _, nxt_action = self.target_policy(nxt_obs_b)
            nxt_state_values = self.target_val_net(nxt_obs_b)
            # nxt_state_values[done_b] = 0.0 
            target = reward_b + self.hparams.gamma * nxt_state_values
        
        advantages = (target - state_values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if optimizer_idx == 0:
            loss = self.hparams.loss_fn(target.float(), state_values.float())
            self.tb_logger.add_scalars('Training/Value/Loss', {
                'loss': loss
                }, global_step=self.global_step
            )    
            self.log('value_loss', loss)
            return loss

        elif optimizer_idx == 1:
            log_prob, _ = self.policy(obs_b)
            prv_log_prob = log_prob_b
            
            rho = T.exp(log_prob - prv_log_prob)
            # rho = log_prob / prv_log_prob

            surrogate_1 = rho * advantages
            surrogate_2 = T.clip(rho, 1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            loss = - T.minimum(surrogate_1, surrogate_2).mean()
            # self.ep_policy_loss.append(loss.item())
            # entropy = -T.sum(action_b*log_prob_b, dim=-1, keepdim=True)
            # loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()
            self.tb_logger.add_scalars('Training/Policy/Loss', {
                'loss': loss
                }, global_step=self.global_step
            )
            # self.tb_logger.add_scalars('Training/Policy/Entropy', {
            #     'entropy': entropy.mean()
            #     }, global_step=self.global_step
            # )
            self.log('policy_loss', loss)
            self.log('return', reward_b.sum())
            return loss
                
    def training_epoch_end(self, training_step_outputs):
        # if self.best_return > 0:
        #     self.policy.load_state_dict(T.load('outputs/logs/best_policy'))
        #     self.value_net.load_state_dict(T.load('outputs/logs/best_value'))
        
        self.target_val_net.load_state_dict(self.value_net.state_dict())            
        self.target_policy.load_state_dict(self.policy.state_dict())
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
            gpus=num_gpus, 
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
    def __init__(self, ctrl, num_envs=50, batch_size=2, obs_time = 20, nb_optim_iters=4, hidden_size=256, samples_per_epoch=2,
                epoch_repeat=4, policy_lr=1e-4, value_lr=1e-3, gamma=0.99, epsilon=0.3, entropy_coef=0.1,
                loss_fn=F.mse_loss, optim=AdamW):
        super().__init__()
        self.loop = asyncio.get_running_loop()
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
        self.init_ts = time.time()
        
        self.save_hyperparameters('batch_size', 'obs_time', 'nb_optim_iters', 'policy_lr', 'value_lr', 
            'hidden_size', 'gamma', 'loss_fn', 'optim', 'samples_per_epoch', 'entropy_coef',
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
        print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee4')
        obs, nodes, _, _ = self.reset()
        print('COLLECTING DATA ....')     
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
            print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee5')   
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
        
        print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee7')
        state_values = self.value_net(obs_b)

        with T.no_grad():
            _, nxt_action = self.target_policy(nxt_obs_b)
            nxt_state_values = self.target_val_net(nxt_obs_b)
            # nxt_state_values[done_b] = 0.0 
            target = reward_b + self.hparams.gamma * nxt_state_values
        
        advantages = (target - state_values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if optimizer_idx == 0:
            loss = self.hparams.loss_fn(target.float(), state_values.float())
            self.tb_logger.add_scalars('Training/Value/Loss', {
                'loss': loss
                }, global_step=self.global_step
            )    
            self.log('value_loss', loss)
            return loss

        elif optimizer_idx == 1:
            log_prob, _ = self.policy(obs_b)
            prv_log_prob = log_prob_b
            
            rho = T.exp(log_prob - prv_log_prob)
            # rho = log_prob / prv_log_prob

            surrogate_1 = rho * advantages
            surrogate_2 = T.clip(rho, 1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

            loss = - T.minimum(surrogate_1, surrogate_2).mean()
            # self.ep_policy_loss.append(loss.item())
            # entropy = -T.sum(action_b*log_prob_b, dim=-1, keepdim=True)
            # loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()
            self.tb_logger.add_scalars('Training/Policy/Loss', {
                'loss': loss
                }, global_step=self.global_step
            )
            # self.tb_logger.add_scalars('Training/Policy/Entropy', {
            #     'entropy': entropy.mean()
            #     }, global_step=self.global_step
            # )
            self.log('policy_loss', loss)
            self.log('return', reward_b.sum())
            return loss
                
    def training_epoch_end(self, training_step_outputs):
        # if self.best_return > 0:
        #     self.policy.load_state_dict(T.load('outputs/logs/best_policy'))
        #     self.value_net.load_state_dict(T.load('outputs/logs/best_value'))
        
        self.target_val_net.load_state_dict(self.value_net.state_dict())            
        self.target_policy.load_state_dict(self.policy.state_dict())
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

        print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee1')
        trainer = Trainer(devices="auto", accelerator="auto",num_nodes=num_gpus,
            # gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            # callbacks=[checkpoint_callback],
            # logger=self.tb_logger,
            reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee2')
        T.autograd.detect_anomaly(True)
        print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee3')
        self.play_episodes()
        print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee6')
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
        print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee3')
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
        # try:
        #     quietRun(f'rm -r outputs/logs/{self.id}_experiences.csv')
        #     quietRun('rm -r outputs/logs/lightning_logs/')
        #     # quietRun('rm -r outputs/content/videos/')
        #     # quietRun('tensorboard --logdir outputs/logs/')
        # except Exception as ex:
        #     logger.warn(ex)

        # algo = SAC('SAC_CTRL_SSDWSN', lr=1e-3, alpha=0.002, tau=0.1)
        logger.info('START TRAINING ...')
        self.tb_logger = SummaryWriter(log_dir="outputs/logs")
        quietRun('chmod -R 777 outputs/logs')
        # tb_logger = CSVLogger(save_dir="outputs/logs")
        print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee1')
        trainer = Trainer(
            gpus=num_gpus, 
            max_epochs=-1, #infinite training
            log_every_n_steps=1,
            callbacks=[TQDMProgressBar(refresh_rate=2)],
            logger=self.tb_logger,
            reload_dataloaders_every_n_epochs = 1,
            # callbacks=[EarlyStopping(monitor='outputs/Q-Loss', mode='min', patience=1000)]
        )
        
        print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee2')
        self.play_episodes()
        trainer.fit(self) 