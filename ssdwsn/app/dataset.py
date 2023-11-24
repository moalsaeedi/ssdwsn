from torch.utils.data import Dataset, DataLoader, IterableDataset
from pytorch_lightning import LightningDataModule
from collections import deque, namedtuple
import itertools
import torch as T
import random
import numpy as np
from typing import Iterable, Callable

class ReplayBuffer:
    """Dataset replay buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class TSReplayBuffer:
    """Timeseries Dataset replay buffer"""
    def __init__(self, capacity, seq_len, y_seq_len=1, test_size=0.10):
        self.seq_len = seq_len
        self.y_seq_len = y_seq_len
        self.test_size = test_size
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)-(self.seq_len-self.y_seq_len)
    
    def append(self, sequence):
        self.buffer.append(sequence)
    
    def trainsample(self, sample_size):
        samples = []
        ln = len(self.buffer) - abs(int(len(self.buffer)*self.test_size))
        for _ in range(sample_size):
            random_idx = random.randint(0, ln-(self.seq_len-self.y_seq_len))
            _x = np.array(list(itertools.islice(self.buffer, random_idx, random_idx+self.seq_len))).squeeze()
            _y = np.array(list(itertools.islice(self.buffer, random_idx+self.seq_len, random_idx+self.seq_len+self.y_seq_len))).squeeze()
            samples.append((_x, _y))
        return samples
    
    def testsample(self, sample_size):
        samples = []
        ln = len(self.buffer) - abs(int(len(self.buffer)*self.test_size))
        for _ in range(sample_size):
            random_idx = random.randint(ln, len(self))
            _x = np.array(list(itertools.islice(self.buffer, random_idx, random_idx+self.seq_len))).squeeze()
            _y = np.array(list(itertools.islice(self.buffer, random_idx+self.seq_len, random_idx+self.seq_len+self.y_seq_len))).squeeze()
        samples.append((_x, _y))
        return samples

class ReplayBufferHindsight:
    """Dataset replay buffer"""
    def __init__(self, capacity, her_prop=0.8):
        self.her_prop = her_prop
        self.buffer = deque(maxlen=capacity//2)
        self.her_buffer = deque(maxlen=capacity//2)

    def __len__(self):
        return len(self.buffer) + len(self.her_buffer)
    
    def append(self, experience, her=False):
        if her:
            self.her_buffer.append(experience)
        else:
            self.buffer.append(experience)
    
    def sample(self, batch_size):
        her_batch_size = int(batch_size * self.her_prop)
        regular_batch_size = batch_size - her_batch_size
        batch = random.sample(self.buffer, regular_batch_size)
        her_batch = random.sample(self.her_buffer, her_batch_size)
        full_batch = list(batch+her_batch)
        random.shuffle(full_batch)
        return full_batch

class ReplayBufferSAC:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class RLDataset_(IterableDataset):
    """Reinforcement Learning dataset

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, buffer, num_samples, samples_per_epoch, gamma):        
        self.buffer = buffer
        self.num_samples = num_samples
        self.samples_per_epoch = samples_per_epoch
        self.gamma = gamma

    def __iter__(self):
        reshape_fn = lambda x: x.reshape(self.num_samples, -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        running_return = np.zeros(self.samples_per_epoch, dtype=float)
        return_b = np.zeros_like(reward_b)

        for row in range(self.samples_per_epoch -1, -1, -1):
            running_return = reward_b[row] + -done_b[row].astype(int) * self.gamma * running_return
            return_b[row] = running_return
        return_b.reshape(self.num_samples, -1)
        
        idx = list(range(self.num_samples))
        random.shuffle(idx)
        for i in idx:
            yield obs_b[i], action_b[i], return_b[i], nxt_obs_b[i]

class RLDataset_SAC(IterableDataset):
    """Reinforcement Learning dataset

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, buffer, num_samples):
        self.buffer = buffer
        self.num_samples = num_samples

    def __iter__(self):
        reshape_fn = lambda x: x.reshape(self.num_samples, -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(self.num_samples):
            yield obs_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

class RLDataset_SAC_Shuffle(IterableDataset):
    """Reinforcement Learning dataset

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, buffer, num_samples, epoch_repeat=4):
        self.buffer = buffer
        self.num_samples = num_samples
        self.epoch_repeat = epoch_repeat

    def __iter__(self):
        reshape_fn = lambda x: x.reshape(self.num_samples, -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.epoch_repeat):
            idx = list(range(self.num_samples))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

class RLDataset_A2C(IterableDataset):
    """Reinforcement Learning dataset

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, buffer, num_samples):
        self.buffer = buffer
        self.num_samples = num_samples

    def __iter__(self):
        reshape_fn = lambda x: x.reshape(self.num_samples, -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(self.num_samples):
            yield obs_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

class RLDataset_A2C_Shuffle(IterableDataset):
    """Reinforcement Learning dataset

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, buffer, num_samples, epoch_repeat=4):
        self.buffer = buffer
        self.num_samples = num_samples
        self.epoch_repeat = epoch_repeat

    def __iter__(self):
        reshape_fn = lambda x: x.reshape(self.num_samples, -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.epoch_repeat):
            idx = list(range(self.num_samples))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

class RLDataset_PPO(IterableDataset):
    """Reinforcement Learning dataset with shuffle

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, buffer, num_samples, epoch_repeat):
        self.buffer = buffer
        self.num_samples = num_samples
        self.epoch_repeat = epoch_repeat

    def __iter__(self):
        reshape_fn = lambda x: x.reshape(self.num_samples, -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, loc_b, scale_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(self.num_samples):
            yield obs_b[i], loc_b[i], scale_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

class RLDataset_PPO_shuffle(IterableDataset):
    """Reinforcement Learning dataset with shuffle

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, buffer, num_samples, epoch_repeat=4):
        self.buffer = buffer
        self.num_samples = num_samples
        self.epoch_repeat = epoch_repeat

    def __iter__(self):
        reshape_fn = lambda x: x.reshape(self.num_samples, -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, loc_b, scale_b, action_b, reward_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.epoch_repeat):
            idx = list(range(self.num_samples))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], loc_b[i], scale_b[i], action_b[i], reward_b[i], done_b[i], nxt_obs_b[i]

class ExperienceSourceDataset(IterableDataset):
    """
    Implementation from PyTorch Lightning Bolts:
    https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/datamodules/experience_source.py

    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterable:
        iterator = self.generate_batch()
        return iterator
    
class RLDataset_GAE(IterableDataset):
    """Reinforcement Learning dataset with shuffle

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, buffer, num_samples, epoch_repeat):
        self.buffer = buffer
        self.num_samples = num_samples
        self.epoch_repeat = epoch_repeat

    def __iter__(self):
        reshape_fn = lambda x: x.reshape(self.num_samples, -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, loc_b, scale_b, action_b, reward_b, gae_b, target_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for i in range(self.num_samples):
            yield obs_b[i], loc_b[i], scale_b[i], action_b[i], reward_b[i], gae_b[i], target_b[i], done_b[i], nxt_obs_b[i]

class RLDataset_GAE_shuffle(IterableDataset):
    """Reinforcement Learning dataset with shuffle

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, buffer, num_samples, epoch_repeat=4):
        self.buffer = buffer
        self.num_samples = num_samples
        self.epoch_repeat = epoch_repeat

    def __iter__(self):
        reshape_fn = lambda x: x.reshape(self.num_samples, -1)
        batch = map(np.vstack, zip(*self.buffer))
        obs_b, loc_b, scale_b, action_b, reward_b, gae_b, target_b, done_b, nxt_obs_b = map(reshape_fn, batch)
        for _ in range(self.epoch_repeat):
            idx = list(range(self.num_samples))
            random.shuffle(idx)
            for i in idx:
                yield obs_b[i], loc_b[i], scale_b[i], action_b[i], reward_b[i], gae_b[i], target_b[i], done_b[i], nxt_obs_b[i]

class TSDataset(IterableDataset):
    """Time Series Learning dataset

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, buffer, sample_size=400):
        self.buffer = buffer
        self.sample_size = sample_size
    
    def __iter__(self):
        for sequance in self.buffer.trainsample(self.sample_size):
            yield sequance

class TSTestDataset(IterableDataset):
    """Time Series Testing dataset

    Args:
        IterableDataset (_type_): _description_
    """
    def __init__(self, buffer, sample_size=400):
        self.buffer = buffer
        self.sample_size = sample_size
    
    def __iter__(self):
        for sequance in self.buffer.testsample(self.sample_size):
            yield sequance

class TimeseriesDataset(Dataset):   
    """Timeseries Dataset"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = T.tensor(X).float()
        self.y = T.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len], self.y[index+self.seq_len-1])
    
class TabularDataset(Dataset):
    """Tabular Dataset"""
    def __init__(self, cats, cons, targets, device):
        self.cats = cats
        self.cons = cons
        self.targets = targets
        self.device = device
        
    def __len__(self):
        return len(self.cats)+len(self.cons)

    def __getitem__(self, index):
        cat = self.cats[index]
        con = self.cons[index]
        target = self.targets[index]
        return dict(
            cat = T.tensor(cat, dtype=T.long).to(self.device),
            con = T.tensor(con, dtype=T.float).to(self.device),
            target = T.tensor(target, dtype=T.long).to(self.device)    
        )
        
class _TSDataset(Dataset):
    """Time Series Dataset"""
    def __init__(self, sequences, labels, device):
        self.sequences = sequences
        self.labels = labels
        self.device = device
        
    def __len__(self):
        return(len(self.sequences))

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        return dict(
            sequence=T.tensor(sequence, dtype=T.float).to(self.device),
            label=T.tensor(label, dtype=T.float).to(self.device)
        )

class _TabularDataset(Dataset):
    """Tabular Dataset"""
    def __init__(self, cats, cons, targets, device):
        self.cats = cats
        self.cons = cons
        self.targets = targets
        self.device = device
        
    def __len__(self):
        return len(self.cats)+len(self.cons)

    def __getitem__(self, index):
        cat = self.cats[index]
        con = self.cons[index]
        target = self.targets[index]
        return dict(
            cat = T.tensor(cat, dtype=T.long).to(self.device),
            con = T.tensor(con, dtype=T.float).to(self.device),
            target = T.tensor(target, dtype=T.long).to(self.device)    
        )           
        
# `MultiCategoryDataset()` for multi-head multi-category (2 or more) model
class MultiCategoryDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        features = self.x[index, :]
        labels = self.y[index, :]
        
        features = T.tensor(features, dtype=T.float32)
        label1 = T.tensor(labels[0], dtype=T.long)
        label2 = T.tensor(labels[1], dtype=T.long)
        label3 = T.tensor(labels[2], dtype=T.long)
        label4 = T.tensor(labels[3], dtype=T.long)
        label5 = T.tensor(labels[4], dtype=T.long)
        return {
            'features': features,
            'label1': label1,
            'label2': label2,
            'label3': label3,
            'label4': label4,
            'label5': label5,
        }
        
class TabularDataModule(LightningDataModule):
    """Flow-rules Setup Data Module"""
    def __init__(self, cat_train, cat_test, con_train, con_test, y_train, y_test, batch_sz, device):
        super().__init__()
        self.cat_train = cat_train
        self.con_train = con_train
        self.y_train = y_train
        self.cat_test = cat_test
        self.con_test = con_test
        self.y_test = y_test        
        self.batch_sz = batch_sz
        self.device = device
        
    def setup(self):
        self.train_dataset = _TabularDataset(self.cat_train, self.con_train, self.y_train, self.device)
        self.test_dataset = _TabularDataset(self.cat_test, self.con_test, self.y_test, self.device)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_sz, shuffle=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_sz, shuffle=True)    
    
class TSDataModule(LightningDataModule):
    """Flow-rules Setup Data Module"""
    def __init__(self, x_train, x_test, y_train, y_test, batch_sz, device):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test        
        self.batch_sz = batch_sz
        self.device = device
        
    def setup(self, stage):
        self.train_dataset = _TSDataset(self.x_train, self.y_train, self.device)
        self.test_dataset = _TSDataset(self.x_test, self.y_test, self.device)
        self.val_dataset = _TSDataset(self.x_test, self.y_test, self.device)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_sz, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_sz, num_workers=4)   
     
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_sz, num_workers=4)   