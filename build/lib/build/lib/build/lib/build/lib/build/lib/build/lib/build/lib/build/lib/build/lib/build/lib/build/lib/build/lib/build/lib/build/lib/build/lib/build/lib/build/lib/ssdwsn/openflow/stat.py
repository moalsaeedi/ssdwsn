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

import struct
import time
from ssdwsn.util.constants import Constants as ct

class Stat(object):
    """Flow Table Entry's Statistics"""
    def __init__(self, stats:bytearray=None):
        """Initiate the statistical part of an entry

        Args:
            stats (bytearray, optional): array of bytes contains (TTL, IDLE, PCOUNT, BCOUNT). Defaults to None.
                TTL : elapsed time the flow entry has been installed
                IDLE: elapsed time for which the flow entry has not matched any packet. values from 0 to 254. IDLE_TIME = 255 means that the entry is install permanently
                PCOUNT: number of packets matched by a flow entry
                BCOUNT: number of bytes in packets matched by a flow entry
        """
        self.stats = bytearray(ct.ST_SIZE)
        if stats:
            size = len(stats)
            if size == 12:
                self.stats[ct.ST_TTL_INDEX:ct.ST_TTL_INDEX+ct.ST_TTL_LEN] = stats[ct.ST_TTL_INDEX:ct.ST_TTL_INDEX+ct.ST_TTL_LEN]
                self.stats[ct.ST_IDLE_INDEX] = stats[ct.ST_IDLE_INDEX]    
                self.stats[ct.ST_PCOUNT_INDEX:ct.ST_PCOUNT_INDEX+ct.ST_PCOUNT_LEN] = stats[ct.ST_PCOUNT_INDEX:ct.ST_PCOUNT_INDEX+ct.ST_PCOUNT_LEN]
                self.stats[ct.ST_BCOUNT_INDEX:ct.ST_BCOUNT_INDEX+ct.ST_BCOUNT_LEN] = stats[ct.ST_BCOUNT_INDEX:ct.ST_BCOUNT_INDEX+ct.ST_BCOUNT_LEN]
            elif size == 5:
                self.stats[ct.ST_TTL_INDEX:ct.ST_TTL_INDEX+ct.ST_TTL_LEN] = stats[ct.ST_TTL_INDEX:ct.ST_TTL_INDEX+ct.ST_TTL_LEN]
                self.stats[ct.ST_IDLE_INDEX] = stats[ct.ST_IDLE_INDEX]  
                self.stats[ct.ST_PCOUNT_INDEX:ct.ST_PCOUNT_INDEX+ct.ST_PCOUNT_LEN] = int.to_bytes(0, 4,'big')
                self.stats[ct.ST_BCOUNT_INDEX:ct.ST_BCOUNT_INDEX+ct.ST_BCOUNT_LEN] = int.to_bytes(0, 4,'big')
            else:
                self.stats[ct.ST_TTL_INDEX:ct.ST_TTL_INDEX+ct.ST_TTL_LEN] = struct.pack(">i", int(time.time()))
                self.stats[ct.ST_IDLE_INDEX] = ct.RL_IDLE 
                self.stats[ct.ST_PCOUNT_INDEX:ct.ST_PCOUNT_INDEX+ct.ST_PCOUNT_LEN] = int.to_bytes(0, 4,'big')
                self.stats[ct.ST_BCOUNT_INDEX:ct.ST_BCOUNT_INDEX+ct.ST_BCOUNT_LEN] = int.to_bytes(0, 4,'big')
        else:
            self.stats[ct.ST_TTL_INDEX:ct.ST_TTL_INDEX+ct.ST_TTL_LEN] = struct.pack(">i", int(time.time()))
            self.stats[ct.ST_IDLE_INDEX] = ct.RL_IDLE 
            self.stats[ct.ST_PCOUNT_INDEX:ct.ST_PCOUNT_INDEX+ct.ST_PCOUNT_LEN] = int.to_bytes(0, 4,'big')
            self.stats[ct.ST_BCOUNT_INDEX:ct.ST_BCOUNT_INDEX+ct.ST_BCOUNT_LEN] = int.to_bytes(0, 4,'big')
    
    @classmethod
    def fromString(cls, val:str):
        frm = Stat()
        stats = val.split(",")
        for stat in stats:
            tmp = stat.split(":")
            lhs = tmp[0].strip()
            rhs = "255" if tmp[1].strip() == "PERM" else tmp[1].strip()        
            {
                "TTL": lambda: frm.setTtl(int(rhs)),
                "IDLE": lambda : frm.setIdle(int(rhs)),
                "PCOUNT": lambda : frm.setPCounter(int(rhs)),
                "BCOUNT": lambda : frm.setBCounter(int(rhs))
            }.get(lhs)()

        return frm        
    
    def getPCounter(self):
        return self.stats[ct.ST_PCOUNT_INDEX:ct.ST_PCOUNT_INDEX+ct.ST_PCOUNT_LEN]
    
    def setPCounter(self, count:int):
        self.stats[ct.ST_PCOUNT_INDEX:ct.ST_PCOUNT_INDEX+ct.ST_PCOUNT_LEN] = int.to_bytes(count, 4,'big')
        
    def getBCounter(self):
        return self.stats[ct.ST_BCOUNT_INDEX:ct.ST_BCOUNT_INDEX+ct.ST_BCOUNT_LEN]
    
    def setBCounter(self, count:int):
        self.stats[ct.ST_BCOUNT_INDEX:ct.ST_BCOUNT_INDEX+ct.ST_BCOUNT_LEN] = int.to_bytes(count, 4,'big')
        
    def increasePCounter(self):
        try:
            self.stats[ct.ST_PCOUNT_INDEX:ct.ST_PCOUNT_INDEX+ct.ST_PCOUNT_LEN] = (int.from_bytes(self.stats[ct.ST_PCOUNT_INDEX:ct.ST_PCOUNT_INDEX+ct.ST_PCOUNT_LEN], 'big') + 1).to_bytes(4, 'big')
        except:
            print('Flow entry reached its max counter usage.')
            
    def increaseBCounter(self, count:int):
        try:
            self.stats[ct.ST_BCOUNT_INDEX:ct.ST_BCOUNT_INDEX+ct.ST_BCOUNT_LEN] = (int.from_bytes(self.stats[ct.ST_BCOUNT_INDEX:ct.ST_BCOUNT_INDEX+ct.ST_BCOUNT_LEN], 'big') + count).to_bytes(4, 'big')
        except:
            print('Flow entry reached its max counter usage.')
    
    def getTtl(self):
        return int.from_bytes(self.stats[ct.ST_TTL_INDEX:ct.ST_TTL_INDEX+ct.ST_TTL_LEN], 'big')
    
    def setTtl(self, ttl:int):
        self.stats[ct.ST_TTL_INDEX:ct.ST_TTL_INDEX+ct.ST_TTL_LEN] = struct.pack(">i", ttl)
    
    def getIdle(self):
        return self.stats[ct.ST_IDLE_INDEX]
        
    def setIdle(self, idle_timeout:int):
        self.stats[ct.ST_IDLE_INDEX] = idle_timeout
        
    def setPermanent(self):
        self.setIdle(ct.RL_IDLE_PERM)

    def isPermanent(self):
        return True if self.getIdle() == ct.RL_IDLE_PERM else False
        
    def restoreIdle(self):
        if not self.isPermanent():
            self.setIdle(ct.RL_IDLE)

    def decrementIdle(self, val:int):
        self.setIdle(self.getIdle() - val)
        
    def toByteArray(self):
        return self.stats
         
    def __str__(self):
        if self.getIdle() == ct.RL_IDLE_PERM:
            return f"IDLE: PERM, PCOUNT: {int.from_bytes(self.getPCounter(), 'big')}, BCOUNT: {int.from_bytes(self.getBCounter(), 'big')}"
        else:
            return f"IDLE: {self.getIdle()}, PCOUNT: {int.from_bytes(self.getPCounter(), 'big')}, BCOUNT: {int.from_bytes(self.getBCounter(), 'big')}"
        
        
    