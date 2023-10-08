#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
 * Copyright (C) 2022 ssdwsn
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from enum import Enum
from time import sleep
import math
from random import gauss

LSPEED = 299792458.0  # speed of light in vacuum m/sec

class PropModel(Enum):
    """Radio Propagation Models

        lognormalshadowing: Log-Distance or Log Normal Shadowing (Indoors)
        friis: Free-Space (Indoors)
        itu: International Telecommunication Union (ITU) (Indoors)
        towrayground: Two-Ray-Ground (Outdoors)

    Args:
        Enum (PropModel): a propagation model type

    Returns:
        value: int value of the propagation model
        name: name of the propagation model
    """
    friis = 0
    lognormalshadowing = 1
    itu = 2
    towrayground = 3
    
    def getValue(self):
        return self.value
    
    @classmethod
    def fromString(cls, str):
        return bytes(cls(str).value)
    
    @classmethod
    def fromByte(cls, value):
        return cls(value).name
    
    @classmethod
    def fromValue(cls, value):
        return cls(value).name
    
    @classmethod
    def getParams(cls, val):
        params = {}
        if val == 0 or val == 'friis':
            params = {
            'rssi': -60,
            'range': 0,
            'txpower': 0,
            'ple': 3,
            'l': 1,
            'noise': -88,
            'cca': -91           
            }
        if val == 1 or val == 'lognormalshadowing':
            params = {
            'rssi': -60,
            'range': 0,
            'txpower': 0,
            'ple': 4,
            'l': 1,
            'mu': 0,
            'variance': 2,
            'noise': -88,
            'cca': -91            
            }
        if val == 2 or val == 'itu':
            params = {
            'rssi': -60,
            'range': 0,
            'txpower': 0,
            'ple': 3,
            'l': 1,
            'pf': 0, # floor penetration loss
            'pl': 0, # power loss
            'nf': 0, # number of floors
            'noise': -88,
            'cca': -91            
            }
        if val == 3 or val == 'towrayground':
            params = {
            'rssi': -60,
            'range': 0,
            'txpower': 0,
            'ple': 3,
            'l': 1,
            'noise': -88,
            'cca': -91          
            }
                           
        return params   

class _Prop(object):
    """Abstract Propagation object
    
        rx: receiver, 
        tx: transmitter
        Pr: transmission power at transmitter tx
        Gr: antenna gain at the receiver rx
        Gt: antenna gain at the transmitter tx
        rssi: receiver signal strength indecator (dBm) 
            RSSI=Pt-PL(d). In this formula, Pt indicated the signal transmission power, 
            PL(d) indicated the path loss when the distance is d, and they are both in dBm
        range: antenna transmission range at the receiver
        d0: the reference distance of calculating the reference path loss
        PLd0: path loss at reference distance d0
        PLd: path loss at arbitrary distance d
        PLE: is the path loss exponent (PLE) for the given environment (e.g., 2:vacuum, infinit space)
        ____________________________________________
        |Environment                    |PLE        |        
        |_______________________________|___________|
        |free space                     |2          |
        |urban area                     |2.7 to 3.5 |
        |shadowed urban                 |3 to 5     |
        |inside bldgs Line-of-Sight     |1.6 to 1.8 |
        |obstructed in bldgs            |4 to 6     |
        |obstructed in factory          |2 to 3     |
        ---------------------------------------------
        cons: constant variable (speed of light in vacuum (m))
        d: distance between the receiver (rx) and transmitter (tx)            
        L: system loss (represents other losses that is not associated with the propagation loss) L=1 for no such system losses
        x: normal random variable
        lambda_: wavelength of carrier in meters
        Pf: floor penetration loss
        Pl: power loss
        Nf: number of floors between the transmitter and receiver. 
        N: power loss coefficient, 28 Indoor at 2.4 GHz Re: Chrysikos et al.
        variance: variance
        noise: noise threshold
        cca: Clear Channel Assessment (CCA) threshold
    """
    
    def __init__(self, propModel: PropModel):
        self.model = propModel.getValue()
        self.params = {}
    
    def getRssi(self):
        return self.params['rssi']
    
    def setRssi(self, val:int):
        self.params['rssi'] = val
    
    def getTxPower(self):
        return self.params['txpower']
    
    def setTxPower(self, val:int):
        self.params['txpower'] = val
    
    def getRange(self):
        return self.params['range']
    
    def setRange(self, val:int):
        self.params['range'] = val
        
    def getSysLoss(self):
        return self.params['l']

    def setSysLoss(self, val:int):
        self.params['l'] = val
        
    def getNoiseThreshold(self):
        return self.params['noise']
    
    def setNoiseThreshold(self, val:int):
        self.params['noise'] = val
        
    def getCCAThreshold(self):
        return self.params['cca']

    def setCCAThreshold(self, val:int):
        self.params['cca'] = val
    
    def pathLoss(self, rx, d:int):
        """Path Loss
            freq: signal frequencey transmitted (Hz)
            d: distance between the transmitter and receiver (m)
            cons: constant variable (speed of light in vacuum (m))
            loss: loss is equal to the system loss
            
        Args:
            rx: receiver node
            distance (int): distance between the transmitter and receiver (m)

        Returns:
            int: return the path loss between the transmitter and receiver for distance d
        """
        
        if d == 0: d = 0.1
        if isinstance(rx, dict):
            frx = rx['freq']
        else:
            frx = rx.wintf.getFreq()
        freq = frx * 10 ** 9  # Convert Ghz to Hz
        cons = LSPEED
        loss = self.params['l']
        lambda_ = cons / freq
        PLd = -10 * math.log10(lambda_ ** 2 / (4 * math.pi * d) ** 2 * loss)

        return int(PLd)
    
    
    def friisPathLoss(self, rx, d:int):
        """Path Loss
            freq: signal frequencey transmitted (Hz)
            d: distance between the transmitter and receiver (m)
            cons: constant variable (speed of light in vacuum (m))
            lambda_: wavelength of carrier in meters
            
        Args:
            rx: receiver node
            distance (int): distance between the transmitter and receiver (m)

        Returns:    
            int: return the path loss between the transmitter and receiver for distance d
        """
        
        if d == 0: d = 0.1
        if isinstance(rx, dict):
            frx = rx['freq']
        else:
            frx = rx.wintf.getFreq()
        freq = frx * 10 ** 9  # Convert Ghz to Hz
        cons = LSPEED
        lambda_ = cons / freq 
        PLd = -10 * math.log10(lambda_ ** 2 / (4 * math.pi * d) ** 2)

        return int(PLd)    

class Friis(_Prop):
    """Friis Free Space Propagation Model
    Friis free space propagation model is used to model the line-of-sight (LOS) path loss 
    incurred in a free space environment
    Ref:https://www.gaussianwaves.com/2013/09/friss-free-space-propagation-model/"""

    def __init__(self):
        super().__init__(PropModel.friis)
        self.params.update(PropModel.getParams(self.model))
        
    def rssi(self, rx, tx, d: int = 0):
        if isinstance(tx, dict):
            Gr = rx['antGain']
        else:
            Gr = rx.wintf.getAntGain()
        Pt = tx['antGain']
        Gt = tx['txPower']
        
        PLd = self.friisPathLoss(rx, d)
        self.params['rssi'] = (Pt + Gt + Gr) - PLd

        return self.params['rssi']
    
    def range(self, rx):
        """Range given the transmission power"""
        f = rx.wintf.getFreq() * 10 ** 9
        Gr = int(rx.wintf.getAntGain())
        Pr = int(rx.wintf.getTxPower())
        cons =  LSPEED
        loss = self.params['l']
    
        lambda_ = cons / f
        self.params['range'] = math.pow(10, ((-self.params['noise'] + (Pr + (Gr * 2)) +
                               10 * math.log10(lambda_ ** 2)) /
                              10 - math.log10((4 * math.pi) ** 2 * loss)) / 2)
        return self.params['range']
    
    def txPower(self, rx):
        """Range given the transmission power"""
        rx_range = rx.wintf.getRange()
        f = rx.wintf.getFreq() * 10 ** 9
        Gr = rx.wintf.getAntGain()
        cons =  LSPEED
        lambda_ = cons / f

        self.params['txpower'] = 10 * (
            math.log10((4 * math.pi) ** 2 * self.params['l'] * rx_range ** 2)) + self.params['noise'] \
                       - 10 * math.log10(lambda_ ** 2) - (Gr * 2)
        if self.params['txpower'] < 0: self.params['txpower'] = 1

        return self.params['txpower']
 
class LogNormalShadowing(_Prop):
    """Log-Distance or Log Normal Shadowing Propagation Model
    Is a radio propagation model that predicts the path loss signal 
    encounters inside a building or densely populated areas over distance
    Ref:https://www.gaussianwaves.com/2013/09/log-distance-path-loss-or-log-normal-shadowing-model/"""
    
    def __init__(self):
        super().__init__(PropModel.lognormalshadowing)
        self.params.update(PropModel.getParams(self.model))
        
    def rssi(self, rx, tx, d: int = 0):
        if d == 0: d = 0.1
        if isinstance(tx, dict):
            Gr = rx['antGain']
        else:
            Gr = rx.wintf.getAntGain()
        Pt = tx['txPower']
        Gt = tx['antGain']
        x = round(gauss(self.params['mu'], self.params['variance']), 2)
        d0 = 1 # close-in reference distance (m)
        PLd0 = self.pathLoss(rx, d0) # reference path loss (dB)
        PLd = int(PLd0) + int(10 * self.params['ple'] * math.log10(d / d0) + x) # path loss at distance d
        self.params['rssi'] = (Pt + Gt + Gr) - PLd

        return self.params['rssi']  
    
    def range(self, rx):
        """Range given the transmission power"""
        Pr = int(rx.wintf.getTxPower())
        Gr = int(rx.wintf.getAntGain())
        x = round(gauss(self.params['mu'], self.params['variance']), 2)
        d0 = 1 # close-in reference distance (m)
        PLd0 = self.pathLoss(rx, d0) # reference path loss (dB)
        loss = PLd0 - x
        self.params['range'] = math.pow(10, (-self.params['noise'] - loss + (Pr + (Gr * 2)) / (10 * self.params['ple']))) * d0
        
        return self.params['range']
    
    def txPower(self, rx):
        """Range given the transmission power"""
        rx_range = rx.wintf.getRange()
        Gr = rx.wintf.getAntGain()
        x = round(gauss(self.params['mu'], self.params['variance']), 2)
        d0 = 1 # close-in reference distance (m)
        PLd0 = self.pathLoss(rx, d0) # reference path loss (dB)
        loss = PLd0 - x
        self.params['txpower'] = 10 * self.params['ple'] * math.log10(rx_range / d0) + \
                       self.params['noise'] + loss - (Gr * 2)
        if self.params['txpower'] < 0: self.params['txpower'] = 1

        return self.params['txpower']

class ITU(_Prop):    
    """International Telecommunication Union (ITU) Propagation Loss Model:
    is a radio propagation model that estimates the path loss inside a room or 
    a closed area inside a building delimited by walls of any form"""

    def __init__(self):
        super().__init__(PropModel.itu)
        self.params.update(PropModel.itu)
        
    def rssi(self, rx, tx, d: int = 0):
        if d == 0: d = 0.1
        if isinstance(tx, dict):
            Gr = rx['antGain']
        else:
            Gr = rx.wintf.getAntGain()
        Pt = tx['txPower']
        Gt = tx['antGain']
        f = tx['freq'] * 10 ** 9
        nf = self.params['nf']
        Pl = self.params['pl']
        Pf = self.params['pf']
        N = 28

        if d > 16: N = 38
        if Pl != 0: N = Pl

        PLd = 20 * math.log10(f) + N * math.log10(d) + Pf * nf - 28
        self.params['rssi'] = (Pt + Gt + Gr) - int(PLd)

        return self.params['rssi']
    
    def range(self, rx):
        """Range given the transmission power"""
        f = rx.wintf.getFreq() * 10 ** 9
        Pr = int(rx.wintf.getTxPower())
        Gr = int(rx.wintf.getAntGain())
        nf = self.params['nf']
        Pf = self.params['pf']
        N = 28

        self.params['range'] = math.pow(10, ((-self.params['noise'] + (Pr + (Gr * 2)) -
                                    20 * math.log10(f) - Pf * nf + 28)/N))
        return self.params['range']
    
    def txPower(self, rx):
        """Transmission power given the range"""
        rx_range = rx.wintf.getRange()
        f = rx.wintf.getFreq() * 10 ** 9
        Gr = rx.wintf.getAntGain()
        Pf = self.params['pf']
        nf = self.params['nf']
        N = 28

        self.params['txpower'] = N * math.log10(rx_range) + self.params['noise'] + \
                       20 * math.log10(f) + Pf * nf - 28 - (Gr * 2)
        if self.params['txpower'] < 0: self.params['txpower'] = 1

        return self.params['txpower']
        
class TowRayGround(_Prop):
    
    """Two Ray Ground Propagation Loss Model 
    is a multipath radio propagation model which predicts the path losses between a transmitting antenna 
    and a receiving antenna when they are in line of sight (LOS)
    model doesn't work well for short distances
    Ref:https://www.gaussianwaves.com/2019/03/two-ray-ground-reflection-model/"""
    
    def __init__(self):
        super().__init__(PropModel.towrayground)
        self.params.update(PropModel.getParams(self.model))
    
    def _friis(self, rx, tx, d: int = 0):
        if isinstance(tx, dict):
            Gr = rx['antGain']
        else:
            Gr = rx.wintf.getAntGain()
        Pt = tx['txPower']
        Gt = tx['antGain']

        PLd = self.friisPathLoss(rx, d)
        self.params['rssi'] = (Pt + Gt + Gr) - PLd

        return self.params['rssi']
       
    def rssi(self, rx, tx, d: int = 0):
        Gr = int(rx.wintf.getAntGain())
        Hr = int(rx.wintf.getAntHeight())
        Pt = int(tx['txPower'])
        Gt = int(tx['antGain'])
        Ht = int(tx['antHeigth'])
        cons =  LSPEED
        f = rx.wintf.getFreq() * 10 ** 9

        if d == 0: d = 0.1
        dCross = (4 * math.pi * Ht * Hr) / ((cons / f))
        if d < dCross:
            self.params['rssi'] = self._friis(rx, tx, d)
        else:
            PLd = int((Pt * Gt * Gr * Ht ** 2 * Hr ** 2) / (d ** 4))
            self.params['rssi'] = (Pt + Gt + Gr) - PLd
            
        return self.params['rssi']
    
    def range(self, rx):
        """Range given the transmission power"""
        Gr = int(rx.wintf.getAntGain())
        Hr = float(rx.wintf.getAntHeight())
        Pr = int(rx.wintf.getTxPower())
        cons =  LSPEED
        f = rx.wintf.getFreq() * 10 ** 9

        dCross = (4 * math.pi * Hr * Hr) / ((cons / f))
        self.params['range'] = ((Pr * Gr * Gr * Hr ** 2 * Hr ** 2) / ((Pr + Gr) - self.params['noise'])/self.params['l']) * dCross
        
        return self.params['range']
    
    def txPower(self, rx):
        """Transmission power given the range"""
        rx_range = rx.wintf.getRange()
        Gr = rx.wintf.getAntGain()
        Hr = rx.wintf.getAntHeight()
        Pr = rx.wintf.getTxPower()
        cons =  LSPEED
        f = rx.wintf.getFreq() * 10 ** 9

        dCross = ((4 * math.pi * Hr) / (cons / f)) * self.params['l']
        self.params['txpower'] = (dCross * ((Pr + Hr)-self.params['rssi']))/(Gr * Hr ** 2)
        if self.params['txpower'] < 0: self.params['txpower'] = 1

        return self.params['txpower']
