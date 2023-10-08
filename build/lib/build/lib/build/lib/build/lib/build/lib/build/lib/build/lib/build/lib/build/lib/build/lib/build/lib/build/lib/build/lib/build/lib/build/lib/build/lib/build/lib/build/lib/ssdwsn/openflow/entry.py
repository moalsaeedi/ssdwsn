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
"""This code is a refactoring of the logic written in SDN-Wise implementation of OpenFlow protocol.
Ref:https://github.com/sdnwiselab/sdn-wise-java
@article{Anadiotis:2019,
    author    = {{Angelos-Christos} Anadiotis and Laura Galluccio and Sebastiano Milardo and Giacomo Morabito and Sergio Palazzo},
    title     = {{SD-WISE: A Software-Defined WIreless SEnsor network}},
    journal   = {Computer Networks},
    volume    = {159},
    pages     = {84 - 95},
    year      = {2019},
    doi       = {10.1016/j.comnet.2019.04.029},
    url       = {http://www.sciencedirect.com/science/article/pii/S1389128618312192},
}
"""

from ssdwsn.util.constants import Constants as ct
from ssdwsn.openflow.window import Window
from ssdwsn.openflow.stat import Stat
from ssdwsn.openflow.action import AbstractAction
from itertools import groupby

class Entry(object):
    """Flow Table Entry"""
    def __init__(self, entry:bytearray=None, windows:list=None, actions:list=None, **params):
        """Initiate a flow table entry (rule)

        Args:
            entry (bytearray, optional): array of bytes contains the match, action, stats of an entry. Defaults to None.
            windows (list, optional): first part of an entry for matching an arriving packet (can be one or more matching windows). Defaults to None.
            actions (list, optional): second part of an entry for performing instructions (can be one or more actions). Defaults to None.
        """
        self.windows = []
        self.actions = []
        # the third part of an entry which contains the entry's statistics
        self.stats = Stat()
        if entry:
            i = 0        
            nWindows = entry[i]
            i += 1       
            for _ in range(nWindows):
                self.windows.append(Window(entry[i:i+ct.W_SIZE]))
                i += ct.W_SIZE
            while i < (len(entry) - ct.ST_SIZE - 1):                
                # FlowEntry is [no. of windows]+[windows[]]+[size+Action+size+Action+..]+[stats[]]
                size = entry[i]
                i += 1
                self.actions.append(AbstractAction.build(entry[i:i+size]))    
                i += size
            if i < len(entry): 
                self.stats = Stat(entry[i:-1])
            else: self.stats = Stat()
            
        elif windows and actions:
            self.windows = windows
            self.actions = actions
            self.stats = Stat()
    
    @staticmethod
    def fromString(val:str):
        val = val.upper().strip()
        res = Entry()

        strWindows = val[val.find('(')+1:val.find(')')].strip().split('&&')
        for w in strWindows:
            res.addWindow(Window.fromString(w))            
        
        strActions = val[val.find('{')+1:val.find('}')].strip().split(';')
        for a in strActions:
            res.addAction(AbstractAction.build(a))
        
        strStats = val[val.find('[')+1:val.find(']')].strip()
        res.addStats(Stat.fromString(strStats))
        
        return res
    
    @staticmethod
    def getEntryWCKey(entry):
        """translate an entry matching windows to wildcard index"""
        src, dst, typ = '.*', '.*', '.*'
        for window in entry.getWindows():                 
            if window.getLhs() == ct.SRC_INDEX:
                src = 'src:'+str(window.getRhs())
            if window.getLhs() == ct.DST_INDEX:
                dst = 'dst:'+str(window.getRhs())
            if window.getLhs() == ct.TYP_INDEX:
                typ = 'typ:'+str(window.getRhs())
        key = [src, dst, typ]
        return ''.join(str(x) for x, _ in groupby(key))

    def toByteArray(self):
        target = bytearray()
        
        target.append(len(self.windows))
        
        for fw in self.windows:
            target.extend(fw.toByteArray())
            
        for a in self.actions:
            target.append(len(a.action))
            target.extend(a.toByteArray())
            
        if self.stats: 
            target.extend(self.stats.toByteArray())

        return target
        
    def getWindows(self):
        return self.windows
    
    def setWindows(self, windows:list):
        self.windows = windows

    def addWindow(self, window:Window):
        self.windows.append(window)
    
    def getActions(self):
        return self.actions

    def setActions(self, actions:list):
        self.actions = actions
            
    def addAction(self, action:AbstractAction):
        self.actions.append(action)

    def getStats(self):
        return self.stats
    
    def setStats(self, stats:Stat):
        self.stats = stats
        
    def addStats(self, stats:Stat):
        self.stats = stats
    
    def hashCode(self):
        return hash(self.windows) + hash(self.actions) + hash(self.stats.stats)
    
    def __eq__(self, other):
        return (self.windows == other.windows)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return (self.windows < other.windows)
            
    def __str__(self) -> str:
        return "IF (%s) {%s} [%s]" %('&&'.join(str(w) for w in self.windows), ';'.join(str(a) for a in self.actions), self.stats)
       