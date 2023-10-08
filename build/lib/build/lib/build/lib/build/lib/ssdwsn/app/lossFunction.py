import torch
import torch.nn as nn

def lossBCE(y_pred, y_train):
    """Binary Cross Entropy Loss"""
    if y_train is None:
        return None
    # class_weight = torch.FloatTensor([1.0, 2.0, 1.0]) #second label has 2x penality (double loss)
    return nn.BCEWithLogitsLoss()(y_pred, y_train)

def lossCCE(y_pred, y_train):
    """Categorical Cross Entropy Loss"""
    if y_train is None:
        return None
    # class_weight = torch.FloatTensor([1.0, 2.0, 1.0]) #second label has 2x penality (double loss)
    return nn.CrossEntropyLoss()(y_pred, y_train)      

def lossMSE(y_pred, y_train):
    """Mean Square Error Loss"""
    if y_train is None:
        return None
    # class_weight = torch.FloatTensor([1.0, 2.0, 1.0]) #second label has 2x penality (double loss)
    return torch.sqrt(nn.MSELoss()(y_pred, y_train))