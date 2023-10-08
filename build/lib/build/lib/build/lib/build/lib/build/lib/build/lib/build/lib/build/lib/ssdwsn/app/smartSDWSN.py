
import asyncio
from typing import Dict
import copy
import random
import itertools
from math import floor, sqrt
import numpy as np
import pandas as pd
from tqdm import tqdm
# ssdwsn libraries
from ssdwsn.util.utils import quietRun
from ssdwsn.app.dataset import TSDataModule, TabularDataModule, TSDataset, TabularDataset, RLDataset, ReplayBuffer
from ssdwsn.app.network import NxtHop_Model, LSTM_Model, LSTM_Seq2Seq, DQN, GradientPolicy
from ssdwsn.app.lossFunction import lossBCE, lossCCE, lossMSE
from app.utilts import polyak_average
# Pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

"""
%matplotlib inline
%config InlineBackend.figure_format='retina'        
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ['#01BEFE', '#FFDD00', '#FF7D00', '#FF006D', '#ADFF02', '#8F00FF']
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
tqdm.pandas()
"""

        
        
######################################################################################
    
class LearningModel:
        #
    BETA=2
    RETURN_CMATRIX=True
    INVALID_ZERO_DIVISON=False
    VALID_ZERO_DIVISON=1.0

    def __init__(self, cat_train, cat_test, con_train, con_test, y_train, y_test, emb_szs, epochs=100, lr=0.001, p=0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {self.device} device')   
        try:            
            self.train_dataset = TensorDataset(torch.tensor(cat_train, dtype=torch.long).to(self.device), 
                                            torch.tensor(con_train, dtype=torch.float).to(self.device), 
                                            torch.tensor(y_train, dtype=torch.float).to(self.device))
            self.test_dataset = TensorDataset(torch.tensor(cat_test, dtype=torch.long).to(self.device), 
                                            torch.tensor(con_test, dtype=torch.float).to(self.device), 
                                            torch.tensor(y_test, dtype=torch.float).to(self.device))

            # self.train_dataset = train_dataset
            # self.test_dataset = test_dataset
            self.emb_szs = emb_szs
            self.cat_dim = cat_train.shape[1]
            self.con_dim = con_train.shape[1]
            self.target_dim = y_train.shape[1]
            self.epochs = epochs
            self.lr = lr
            self.p = p
            self.train_batch_size = int(sqrt(con_train.shape[0]))
        
        except Exception as e:
            error(f'ERROR: {e}')
            from tkinter import messagebox
            messagebox.showerror('ERROR', str(y_train.dtype))    
        # print(f'emb_szs:\n{self.emb_szs}')
        # print(f'cat_dim:\n{self.cat_dim}')
        # print(f'con_dim:\n{self.con_dim}')
        # print(f'target_dim:\n{self.target_dim}')
        # print(f'train_batch_size:\n{self.train_batch_size}')         
        # self.input_dim = conts.shape[1] + emb_szs
        # self.output_dim = len(targets) 
    
    def lossBCE(self, y_pred, y_train):
        """Binary Cross Entropy Loss"""
        if y_train is None:
            return None
        # class_weight = torch.FloatTensor([1.0, 2.0, 1.0]) #second label has 2x penality (double loss)
        return nn.BCEWithLogitsLoss()(y_pred, y_train)
    
    def lossCCE(self, y_pred, y_train):
        """Categorical Cross Entropy Loss"""
        if y_train is None:
            return None
        # class_weight = torch.FloatTensor([1.0, 2.0, 1.0]) #second label has 2x penality (double loss)
        return nn.CrossEntropyLoss()(y_pred, y_train)      
    
    def lossMSE(self, y_pred, y_train):
        """Mean Square Error Loss"""
        if y_train is None:
            return None
        # class_weight = torch.FloatTensor([1.0, 2.0, 1.0]) #second label has 2x penality (double loss)
        return torch.sqrt(nn.MSELoss()(y_pred, y_train))
    
    def one_hot_ce_loss(self, y_pred, y_train):
        criterion = nn.CrossEntropyLoss()
        _, labels = torch.max(y_train, dim=1)
        print(labels)
        return criterion(y_pred, labels)
    
    # training function
    async def _train(self, model, train_dataloader, optimizer, loss_fn, train_dataset):
        
        import time
        start_time = time.time()
        counter = 0
        train_running_loss = 0.0
        losses = []
        for i, (cat_train, con_train, y_train) in tqdm(enumerate(train_dataloader), total=int(len(train_dataset)/train_dataloader.batch_size)):
            try:
                counter += 1
                # print(f'cat_train:\n{cat_train}')
                # print(f'con_train:\n{con_train}')
                # print(f'y_train:\n{y_train}')
                
                # extract the features and labels
                # cat_train = data['cats'].to(device)
                # con_train = data['conts'].to(device)
                # y_train = data['targets'].to(device)
                
                # zero-out the optimizer gradients
                optimizer.zero_grad()
                y_pred = model(cat_train, con_train)
                print(y_pred)
                print(y_train)
                i_loss = loss_fn(y_pred, y_train)
                train_running_loss += i_loss.item()
                
                # losses.append(i_loss)
                if i%10 == 0:
                    print(f'epoch {i} loss is {i_loss.item()}')
                     
                # backpropagation
                i_loss.backward()
                # update optimizer parameters
                optimizer.step()
            except Exception as e:
                error(f'ERROR: {e}')
                from tkinter import messagebox
                messagebox.showerror('ERROR', e)              
            
        train_loss = train_running_loss / counter
        
        duration = time.time() - start_time
        
        print(f'Training took {duration/60} minutes')
        await asyncio.sleep(0)
        return train_loss
        
    def print_confusion_matrix(self, confusion_matrix, axes, class_label, classes, fontsize=14):

        df_cm = pd.DataFrame(
            confusion_matrix, index=classes, columns=classes,
        )

        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_ylabel('True label')
        axes.set_xlabel('Predicted label')
        axes.set_title("Confusion Matrix for the class - " + class_label)

    async def nextHopPredModel(self, layers:list=[100], classes:dict=None):
        """Async function to train and evaluate the Next-Hope Prediction Model
        result: a trained model to predict next-hop route of arriving packet"""
        # torch.manual_seed(33)
        if self.train_batch_size == 1:
            await asyncio.sleep(0)
            return
        model = NxtHop_Model(self.emb_szs, self.con_dim, self.target_dim, layers, p=0.4)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = self.lossBCE
        train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size)
        test_dataloader = DataLoader(self.test_dataset, shuffle=True, batch_size=1)
        
        epochs = self.epochs
        # load the model on to the computation device
        model.to(self.device)

        # start the training
        model.train()
        train_loss = []
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = await self._train(model, train_dataloader, optimizer, loss_fn, self.train_dataset)
            train_loss.append(train_epoch_loss)
            print(f"Train Loss: {train_epoch_loss:.4f}")
            
        torch.save(model.state_dict(), 'outputs/multi_head_binary.pth')

        # plot and save the train loss graph
        plt.figure(figsize=(10, 7))
        plt.plot(list(range(epochs)), train_loss, color='orange', label='train loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('outputs/multi_head_binary_loss.png')
        plt.show()

        # TESTING ..  
        model.eval()
        y_pred_labels = []
        y_actl_labels = []
        n_correct = 0
        n_wrong = 0
        try:
            with torch.no_grad(): 
                for i, (cat_test, con_test, y_test) in enumerate(test_dataloader):
                    print(f"SAMPLE {i}")        
                    y_pred = model(cat_test, con_test)
                    loss = loss_fn(y_pred, y_test)
                
                    print(f'{loss_fn.__name__}: {loss:.8f}') 
                    if torch.argmax(y_test) == torch.argmax(y_pred):
                        n_correct += 1
                    else: n_wrong += 1                                  
                    pred = classes[torch.argmax(y_pred).item()]
                    y_pred_labels.append(torch.argmax(y_pred).item())
                    actl = classes[torch.argmax(y_test).item()]
                    y_actl_labels.append(torch.argmax(y_test).item())
                    
                    print(f'{"PREDICTED":>12} {"ACTUAL":>8}')
                    print(f'{pred:>12} {actl:>8}')                                
            
            acc = (n_correct * 1.0) / (n_correct + n_wrong)
            print(f'{"ACCURACY":>12}{acc}')
            labels = np.array(classes)
            cr = classification_report(labels[y_actl_labels], labels[y_pred_labels], output_dict=True, zero_division=0)
            report = pd.DataFrame(cr).T
            report.to_csv('outputs/classification_report.csv', mode='w+', sep='\t',)
            mcm = multilabel_confusion_matrix(y_actl_labels, y_pred_labels)
            outputs = pd.DataFrame({'actual': [classes[val] for val in y_actl_labels],
                                    'pred': [classes[val] for val in y_pred_labels]})
            outputs.to_csv('outputs/y_test_vs_y_pred.csv', mode='w+', sep='\t',)
            fig, ax = plt.subplots(4, 4, figsize=(12, 7))
    
            for axes, cfs_matrix, label in zip(ax.flatten(), mcm, classes):
                self.print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
            
            fig.tight_layout()
            plt.show()
            
        except Exception as e:
            error(f'ERROR: {e}')
            from tkinter import messagebox
            messagebox.showerror('ERROR', e)
        # print(confusion_matrix.diag()/confusion_matrix.sum(1))
        # visualization accuracy 
        # plt.plot(list(int(len(self.test_dataset)/test_dataloader.batch_size)),accuracy_list,color = "red")
        # plt.xlabel("Number of iteration")
        # plt.ylabel("Accuracy")
        # plt.title("RNN: Accuracy vs Number of iteration")
        # plt.savefig('graph.png')
        # plt.show()
        
        await asyncio.sleep(0)
        
    async def fullRoutePredModel(self, layers=[100], classes:dict=None):
        """Async function to train and evaluate the Next-Hope Prediction Model
        result: a trained model to predict (full-path) source route of arriving packet"""
        # torch.manual_seed(33)
        if self.train_batch_size == 1:
            await asyncio.sleep(0)
            return  
        model = NxtHop_Model(self.emb_szs, self.con_dim, self.target_dim, layers, p=0.4)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = self.lossMSE
        train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size)
        test_dataloader = DataLoader(self.test_dataset, shuffle=True, batch_size=1)
        
        epochs = self.epochs
        # load the model on to the computation device
        model.to(self.device)

        # start the training
        model.train()
        train_loss = []
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = await self._train(model, train_dataloader, optimizer, loss_fn, self.train_dataset)
            train_loss.append(train_epoch_loss)
            print(f"Train Loss: {train_epoch_loss:.4f}")
            
        torch.save(model.state_dict(), 'outputs/multi_head_binary.pth')

        # plot and save the train loss graph
        plt.figure(figsize=(10, 7))
        plt.plot(list(range(epochs)), train_loss, color='orange', label='train loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('outputs/multi_head_binary_loss.png')
        plt.show()

        # TESTING ..  
        model.eval()
        y_pred_labels = []
        y_actl_labels = []
        n_correct = 0
        n_wrong = 0
        try:
            with torch.no_grad(): 
                for i, (cat_test, con_test, y_test) in enumerate(test_dataloader):
                    print(f"SAMPLE {i}")        
                    y_pred = model(cat_test, con_test)
                    loss = loss_fn(y_pred, y_test)
                
                    print(f'{loss_fn.__name__}: {loss:.8f}') 
                    # get all the labels
                    y_pred = torch.round(y_pred).abs()
                    pred_labels = []
                    for out in y_pred.squeeze(0):
                        if out < 0:
                            pred_labels.append(-1)
                        else:
                            pred_labels.append(out)
                                                        
                    # pred = mbzr.inverse_transform(np.array([pred_labels]))
                    y_pred_labels.append(pred_labels)
                    # actl = mbzr.inverse_transform(y_test.numpy())
                    y_actl_labels.append(y_test.squeeze(0).tolist())
                    print(f'{"PREDICTED":>12} {"ACTUAL":>8}')
                    # print(f'{pred:>12} {actl:>8}')                    
            
            # acc = (n_correct * 1.0) / (n_correct + n_wrong)
            # print(f'{"ACCURACY":>12}{acc}')
            y_actl_labels = np.array(y_actl_labels)
            y_pred_labels = np.array(y_pred_labels)
            outputs = pd.DataFrame()
            for i in range(y_actl_labels.shape[1]):
                cr = classification_report(y_actl_labels[:,i], y_pred_labels[:,i], output_dict=True, zero_division=0)
                report = pd.DataFrame(cr).T
                if i == 0:
                    report.to_csv('outputs/classification_report.csv', mode='w+', sep='\t',)
                    outputs = pd.concat([pd.DataFrame(y_actl_labels[:,i], columns=[str(i)]), pd.DataFrame(y_pred_labels[:,i], columns=[str(i)])], axis=1)
                else: 
                    report.to_csv('outputs/classification_report.csv', mode='a', sep='\t',)
                    outputs = pd.concat([outputs, pd.DataFrame(y_actl_labels[:,i], columns=[str(i)]), pd.DataFrame(y_pred_labels[:,i], columns=[str(i)])], axis=1)
                mcm = multilabel_confusion_matrix(y_actl_labels[:,i], y_pred_labels[:,i])
                # fig, ax = plt.subplots(4, 4, figsize=(12, 7))
        
                # for axes, cfs_matrix, label in zip(ax.flatten(), mcm, list(classes.values())):
                #     self.print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
                
                # fig.tight_layout()
                # plt.show()
            outputs.to_csv('outputs/y_test_vs_y_pred.csv', mode='w+', sep='\t',)
            
        except Exception as e:
            error(f'ERROR: {e}')
            from tkinter import messagebox
            messagebox.showerror('ERROR', e)
                  
        await asyncio.sleep(0)

class TSLearningModel:
        #
    BETA=2
    RETURN_CMATRIX=True
    INVALID_ZERO_DIVISON=False
    VALID_ZERO_DIVISON=1.0

    def __init__(self, x_train, x_test, y_train, y_test, y_seq_len=1, epochs=100, lr=0.001, p=0.5):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {self.device} device')          
        self.batch_sz = int(sqrt(x_train.shape[0]))
        
        self.train_dataset = TSDataset(x_train, y_train, self.device)
        self.test_dataset = TSDataset(x_test, y_test, self.device)
        # self.data_module.setup()
        
        self.input_dim = x_train.shape[-1]
        self.output_dim = y_train.shape[1]
        self.y_seq_len = y_seq_len
        self.epochs = epochs
        self.lr = lr
        self.p = p
        # print(f'emb_szs:\n{self.emb_szs}')
        # print(f'cat_dim:\n{self.cat_dim}')
        # print(f'con_dim:\n{self.con_dim}')
        # print(f'target_dim:\n{self.target_dim}')
        # print(f'train_batch_size:\n{self.train_batch_size}')         
        # self.input_dim = conts.shape[1] + emb_szs
        # self.output_dim = len(targets)  
           
    # LSTM training function
    async def _train(self, model, train_dataloader, optimizer, loss_fn, train_dataset):
        
        import time
        start_time = time.time()
        counter = 0
        train_running_loss = 0.0
        losses = []
        for i, item in tqdm(enumerate(train_dataloader), total=int(len(train_dataset)/train_dataloader.batch_size)):
            try:
                counter += 1
                # print(f'cat_train:\n{cat_train}')
                # print(f'con_train:\n{con_train}')
                # print(f'label:\n{label}')
                
                # extract the features and labels
                # cat_train = data['cats'].to(device)
                # con_train = data['conts'].to(device)
                # label = data['targets'].to(device)
                
                # zero-out the optimizer gradients
                optimizer.zero_grad()
                y_pred = model(item['sequence'])
                i_loss = loss_fn(y_pred, item['label'])
                train_running_loss += i_loss.item()
                
                # losses.append(i_loss)
                if i%10 == 0:
                    print(f'epoch {i} loss is {i_loss.item()}')
                     
                # backpropagation
                i_loss.backward()
                # update optimizer parameters
                optimizer.step()
            except Exception as e:
                error(f'ERROR: {e}')
                from tkinter import messagebox
                messagebox.showerror('ERROR', e)              
            
        train_loss = train_running_loss / counter
        
        duration = time.time() - start_time
        
        print(f'Training took {duration/60} minutes')
        await asyncio.sleep(0)
        return train_loss
   
    # Seq2Seq LSTM training function
    def _seq2seq_train(self, model, train_dataloader, optimizer, loss_fn, train_dataset):
        
        import time
        start_time = time.time()
        counter = 0
        train_running_loss = 0.0
        losses = []
        for i, item in tqdm(enumerate(train_dataloader), total=int(len(train_dataset)/train_dataloader.batch_size)):
            try:
                counter += 1
                # print(f'cat_train:\n{cat_train}')
                # print(f'con_train:\n{con_train}')
                # print(f'label:\n{label}')
                
                # extract the features and labels
                # cat_train = data['cats'].to(device)
                # con_train = data['conts'].to(device)
                # label = data['targets'].to(device)
                # outputs tensor
                outputs = torch.zeros(self.y_seq_len, item['sequence'].shape[1], item['sequence'].shape[2])
                # initialize hidden state
                encoder_hidden = model.encoder.init_hidden(item['sequence'].shape[1])
                # zero-out the optimizer gradients                
                optimizer.zero_grad()
                # encoder outputs
                encoder_output, encoder_hidden = model.encoder(item['sequence'])
                # decoder with teacher forcing
                decoder_input = item['sequence']   # shape: (batch_size, input_size)
                decoder_hidden = encoder_hidden
                # predict recursively
                for t in range(self.y_seq_len): 
                    decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output
                    decoder_input = decoder_output
                
                print(f'outputs: {outputs}')
                i_loss = loss_fn(outputs, item['label'])
                train_running_loss += i_loss.item()
                
                # losses.append(i_loss)
                if i%10 == 0:
                    print(f'epoch {i} loss is {i_loss.item()}')
                     
                # backpropagation
                i_loss.backward()
                # update optimizer parameters
                optimizer.step()
            except Exception as e:
                error(f'ERROR: {e}')
                from tkinter import messagebox
                messagebox.showerror('ERROR', e)              
            
        train_loss = train_running_loss / counter
        
        duration = time.time() - start_time
        
        print(f'Training took {duration/60} minutes')
        # await asyncio.sleep(0)
        return train_loss
     
    def one_hot_ce_loss(self, y_pred, y_train):
        criterion = nn.CrossEntropyLoss()
        _, labels = torch.max(y_train, dim=1)
        return criterion(y_pred, labels)
    
    def descale(self, descaler, values):
        values_2d = np.array(values)[:, np.newaxis]
        return descaler.inverse_transform(values_2d).flatten()
    
    async def lstm(self, layers, scaler=None):
        # torch.manual_seed(33)
        try:
            model = LSTM_Model(self.input_dim, layers[0], len(layers), 0.2)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = lossMSE
            train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_sz)
            test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=1)
            
            epochs = self.epochs
            # load the model on to the computation device
            model.to(self.device)

            # start the training
            model.train()
            train_loss = []
            for epoch in range(epochs):
                print(f"Epoch {epoch+1} of {epochs}")
                train_epoch_loss = await self._train(model, train_dataloader, optimizer, loss_fn, self.train_dataset)
                train_loss.append(train_epoch_loss)
                print(f"Train Loss: {train_epoch_loss:.4f}")
                
            torch.save(model.state_dict(), 'outputs/multi_head_binary.pth')

            # plot and save the train loss graph
            plt.figure(figsize=(10, 7))
            plt.plot(list(range(epochs)), train_loss, color='orange', label='train loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('outputs/multi_head_binary_loss.png')
            plt.show()
            
            # TESTING ..  
            model.eval()  
            
            predictions = []
            labels = []                        
            
            for item in tqdm(test_dataloader):
                sequence = item['sequence']
                label = item['label']
                output = model(sequence)
                predictions.append(output.item())
                labels.append(label.item())
                
            descaler = MinMaxScaler()
            descaler.min_, descaler.scale_ = scaler.min_[0], scaler.scale_[1]
            data_predict = self.descale(descaler, predictions)
            dataY_plot = self.descale(descaler, labels)

            plt.axvline(x=data_predict.shape[0], c='r', linestyle='--')

            plt.plot(data_predict, '-', label='predicted')
            plt.plot(dataY_plot, '-', label='actual')
            plt.xticks(rotation=45)
            plt.legend()
            plt.suptitle('Time-Series Prediction')
            plt.savefig('outputs/lstmpred.png')        
            plt.show()
            
        except Exception as e:
            error(f'ERROR: {e}')
            from tkinter import messagebox
            messagebox.showerror('ERROR', e)  
    
    def seq2seq_lstm(self, layers, scaler=None):
        # torch.manual_seed(33)
        try:
            
            model = LSTM_Seq2Seq(self.input_dim, layers[0], len(layers), 0.5)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = lossMSE
            train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_sz)
            test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=1)
            
            epochs = self.epochs
            # load the model on to the computation device
            model.to(self.device)

            # start the training
            model.train()
            train_loss = []
            for epoch in range(epochs):
                print(f"Epoch {epoch+1} of {epochs}")
                train_epoch_loss = self._seq2seq_train(model, train_dataloader, optimizer, loss_fn, self.train_dataset)
                train_loss.append(train_epoch_loss)
                print(f"Train Loss: {train_epoch_loss:.4f}")
                
            torch.save(model.state_dict(), 'outputs/multi_head_binary.pth')

            # plot and save the train loss graph
            plt.figure(figsize=(10, 7))
            plt.plot(list(range(epochs)), train_loss, color='orange', label='train loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('outputs/multi_head_binary_loss.png')
            plt.show()
            
            # TESTING ..  
            model.eval()  
            
            predictions = []
            labels = []                        
            
            for item in tqdm(test_dataloader):
                sequence = item['sequence']
                label = item['label']
                output = model(sequence)
                predictions.append(output.item())
                labels.append(label.item())
                
            descaler = MinMaxScaler()
            descaler.min_, descaler.scale_ = scaler.min_[0], scaler.scale_[1]
            data_predict = self.descale(descaler, predictions)
            dataY_plot = self.descale(descaler, labels)

            plt.axvline(x=data_predict.shape[0], c='r', linestyle='--')

            plt.plot(data_predict, '-', label='predicted')
            plt.plot(dataY_plot, '-', label='actual')
            plt.xticks(rotation=45)
            plt.legend()
            plt.suptitle('Time-Series Prediction')
            plt.savefig('outputs/lstmpred.png')        
            plt.show()
            
        except Exception as e:
            error(f'ERROR: {e}')
            from tkinter import messagebox
            messagebox.showerror('ERROR', e)              
############################################################################################################        

class TabularLearningModel:
        #
    BETA=2
    RETURN_CMATRIX=True
    INVALID_ZERO_DIVISON=False
    VALID_ZERO_DIVISON=1.0

    def __init__(self, cat_train, cat_test, con_train, con_test, y_train, y_test, emb_szs, epochs=100, p=0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {self.device} device')          
        self.batch_sz = int(sqrt(con_train.shape[0]))
        
        self.data_module = TabularDataModule(cat_train, cat_test, con_train, con_test, y_train, y_test, self.batch_sz, self.device)       
        
        self.emb_szs = emb_szs
        n_emb = sum([nf for ni,nf in emb_szs]) 
        self.n_cont = con_train.shape[-1]
        self.input_dim = n_emb + con_train.shape[-1]
        self.output_dim = y_train.shape[1]
        self.epochs = epochs
        self.p = p
            
        # print(f'emb_szs:\n{self.emb_szs}')
        # print(f'cat_dim:\n{self.cat_dim}')
        # print(f'con_dim:\n{self.con_dim}')
        # print(f'target_dim:\n{self.target_dim}')
        # print(f'train_batch_size:\n{self.train_batch_size}')         
        # self.input_dim = conts.shape[1] + emb_szs
        # self.output_dim = len(targets) 
    
    def one_hot_ce_loss(self, y_pred, y_train):
        criterion = nn.CrossEntropyLoss()
        _, labels = torch.max(y_train, dim=1)
        print(labels)
        return criterion(y_pred, labels)
    
class TSLearningModell:
        #
    BETA=2
    RETURN_CMATRIX=True
    INVALID_ZERO_DIVISON=False
    VALID_ZERO_DIVISON=1.0

    def __init__(self, x_train, x_test, y_train, y_test, epochs=100, p=0.5):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {self.device} device')          
        self.batch_sz = 1 #int(sqrt(x_train.shape[0]))
        
        self.data_module = TSDataModule(x_train, x_test, y_train, y_test, self.batch_sz, self.device)       
        # self.data_module.setup()
        
        self.input_dim = x_train.shape[-1]
        self.output_dim = y_train.shape[1]
        self.epochs = epochs
        self.p = p
        # print(f'emb_szs:\n{self.emb_szs}')
        # print(f'cat_dim:\n{self.cat_dim}')
        # print(f'con_dim:\n{self.con_dim}')
        # print(f'target_dim:\n{self.target_dim}')
        # print(f'train_batch_size:\n{self.train_batch_size}')         
        # self.input_dim = conts.shape[1] + emb_szs
        # self.output_dim = len(targets)     
    
    def one_hot_ce_loss(self, y_pred, y_train):
        criterion = nn.CrossEntropyLoss()
        _, labels = torch.max(y_train, dim=1)
        return criterion(y_pred, labels)
    
    def descale(self, descaler, values):
        values_2d = np.array(values)[:, np.newaxis]
        return descaler.inverse_transform(values_2d).flatten()
    
    async def flowSetupPred(self, layers=[100], scaler=None):
        await asyncio.sleep(0)
        
        try:
            model = FlowSetupPredictor(self.input_dim, self.output_dim, layers, p=0.2)
            
            early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)
            checkpoint_callback = ModelCheckpoint(
                dirpath = "outputs/checkpoints",
                filename = 'best-checkpoint',
                save_top_k = 1,
                verbose = True,
                monitor = "val_loss",
                mode = "min"
            )
            logger = TensorBoardLogger("lightning_logs", name="flowrules-setup")

            trainer = pl.Trainer(
                logger = logger,
                log_every_n_steps=5,
                callbacks = [checkpoint_callback, early_stopping_callback],
                max_epochs = self.epochs,                
                enable_progress_bar = False
            )
            
            trainer.fit(model, self.data_module)
            
            trained_model = FlowSetupPredictor.load_from_checkpoint("outputs/checkpoints/best-checkpoint.ckpt", input_dim=self.input_dim, output_dim=self.output_dim, layers=layers)
            trained_model.eval()
            trained_model.freeze()
            
            predictions = []
            labels = []                        
            
            for item in tqdm(self.data_module.test_dataset):
                sequence = item['sequence']
                label = item['label']
                print(sequence.unsqueeze(dim=0))
                _, output = trained_model(sequence.unsqueeze(dim=0))
                predictions.append(output.item())
                labels.append(label.item())
            print(predictions)
            print(labels)
            descaler = MinMaxScaler()
            descaler.min_, descaler.scale_ = scaler.min_[0], scaler.scale_[1]
            data_predict = self.descale(descaler, predictions)
            dataY_plot = self.descale(descaler, labels)

            #plt.axvline(x=train_size, c='r', linestyle='--')

            plt.plot(dataY_plot)
            plt.plot(data_predict)
            plt.suptitle('Time-Series Prediction')
            plt.show()
        
        except Exception as e:
            print(e)
            from tkinter import messagebox
            messagebox.showerror('ERROR', e)
        await asyncio.sleep(0)
    """
    async def RNNModel(self, laysrs=[100]):
        # separate features from targets (labels)
        # targets_numpy = self.dataset.label.values
        targets_numpy = self.targets
        # features_numpy = self.dataset.loc[:,self.dataset.columns != 'label'].values
        features_numpy = self.features
        # split data into 80% (train data) and 20% (test data)
        features_train, features_test, targets_train, targets_test = train_test_split(features_numpy, targets_numpy, test_size = 0.2, random_state = 42)
        # Training set tensor
        featuresTrain = torch.from_numpy(features_train)
        targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
        # Test set Tensor
        featuresTest = torch.from_numpy(features_test)
        targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)
        # batch_size, epoch and iteration
        batch_size = floor(len(features_train)/4) #100
        n_iters = 100 #10000
        num_epochs = n_iters / (len(features_train) / batch_size)
        num_epochs = int(num_epochs)
        # Pytorch train and test sets
        train = TensorDataset(featuresTrain,targetsTrain)
        test = TensorDataset(featuresTest,targetsTest)
        # data loader
        train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
        test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)
        # Create RNN
        model = _RNNModel(input_dim, hidden_dim, layer_dim, output_dim).to(self.device)
        # Cross Entropy Loss 
        error = nn.CrossEntropyLoss()
        # SGD Optimizer
        learning_rate = 0.05
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        seq_dim = self.input_dim  
        loss_list = []
        iteration_list = []
        accuracy_list = []
        count = 0
        for epoch in range(num_epochs):
            for i, (instances, labels) in enumerate(train_loader):
                try:
                    print('1 Here we are >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    train  = Variable(instances.view(-1, seq_dim, input_dim))
                    labels = Variable(labels)
                    print('2 Here we are >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    # Clear gradients
                    optimizer.zero_grad()
                    print('train Variable ',train)
                    # Forward propagation
                    outputs = model(train)
                    
                    # Calculate softmax and ross entropy loss
                    loss = error(outputs, labels)
                    
                    # Calculating gradients
                    loss.backward()
                    
                    # Update parameters
                    optimizer.step()
                    
                    count += 1
                    print('3 Here we are >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')                    
                    if count % 25 == 0:
                        # Calculate Accuracy         
                        correct = 0
                        total = 0
                        # Iterate through test dataset
                        for instances, labels in test_loader:
                            instances = Variable(instances.view(-1, seq_dim, input_dim))
                            
                            # Forward propagation
                            outputs = model(instances)
                            
                            # Get predictions from the maximum value
                            predicted = torch.max(outputs.data, 1)[1]
                            
                            # Total number of labels
                            total += labels.size(0)
                            
                            correct += (predicted == labels).sum()
                        
                        accuracy = 100 * correct / float(total)
                        
                        # store loss and iteration
                        loss_list.append(loss.data)
                        iteration_list.append(count)
                        accuracy_list.append(accuracy)
                        if count % 50 == 0:
                            # Print Loss
                            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0], accuracy))
                except Exception as e:
                    print(e)
        print('4 Here we are >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')            
        # visualization loss 
        plt.plot(iteration_list,loss_list)
        plt.xlabel("Number of iteration")
        plt.ylabel("Loss")
        plt.title("RNN: Loss vs Number of iteration")
        plt.show()

        # visualization accuracy 
        plt.plot(iteration_list,accuracy_list,color = "red")
        plt.xlabel("Number of iteration")
        plt.ylabel("Accuracy")
        plt.title("RNN: Accuracy vs Number of iteration")
        plt.savefig('graph.png')
        plt.show()
        await asyncio.sleep(0)
        """   