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

from math import ceil
from ssdwsn.util.constants import Constants as ct

class Battery(object):    
    """Battery Abstract Object"""
    def __init__(self, level:int=None):
        if level:
            self.level = level
            self.initLevel = level
        else: 
            self.level = ct.MAX_LEVEL
            self.initLevel = ct.MAX_LEVEL

    def getInitLevel(self):
        return int(self.initLevel)

    def getLevel(self):
        return int(self.level)
    
    def setLevel(self, batteryLevel):
        if batteryLevel >= 0:
            self.level = batteryLevel
        else: self.level = 0 

    def transmitRadio(self, t):
        newLevel = self.level - ct.RADIO_TX * t
        self.setLevel(newLevel)
    
    def receivedRadio(self, t):
        # newLevel = self.level - ct.RADIO_RX * t
        newLevel = self.level - ct.RADIO_RX
        self.setLevel(newLevel)
    
    def keepAlive(self, t):
        newLevel = self.level - ct.KEEP_ALIVE * t
        self.setLevel(newLevel)
        
class Dischargable(Battery):
    """Dischargable Battery"""
    def __init__(self, level:int=None):
        """Initiate a dischargable battery

        Args:
            level (int, optional): charging level. Defaults to None.
        """
        super().__init__(level)
    
class Chargable(Battery):
    """Chargable Battery"""
    def __init__(self):
        """Initiate a chargable battery
        """
        super().__init__()
        
    def transmitRadio(self, t):
        pass
    
    def receivedRadio(self, t):
        pass
    
    def keepAlive(self, t):
        pass