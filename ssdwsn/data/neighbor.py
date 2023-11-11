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

from ctypes import DEFAULT_MODE
from typing import Tuple
from ssdwsn.data.addr import Addr
from ssdwsn.util.constants import Constants as ct
from ssdwsn.util.utils import getColorInt, getColorVal

class Neighbor:
    """Neigbor Node Information Object"""
    # def __init__(self, addr:Addr=None, rssi:int=None, batt:int=None, port:int=None, color:str=None):
    def __init__(self, dist:float, addr:Addr=None,  rssi:int=None, port:int=None, tosinkdist:int=None):
        """Generate neighbor node information object

        Args:
            addr (Addr, optional): ip address of the neigbor node. Defaults to None.
            rssi (int, optional): received signal strength of the neigbor node. Defaults to None.
            batt (int, optional): battery charging level of the neigbor node. Defaults to None.
            port (int, optional): port address of the neigbor node. Defaults to None.
            tosinkdist (int, optional): Distance (number of hops) to the neariest sink. Defaults to None.
        """
        self.dist = dist
        self.addr = Addr(ct.BROADCAST_ADDR) if addr is None else addr
        self.rssi = int(ct.DEFAULT) if rssi is None else rssi
        self.port = port # tuple (Address, port)
        self.tosinkdist = ct.DIST_MAX + 1 if tosinkdist is None else tosinkdist
        # self.color = color # default is black
    
    def getDist(self):
        return self.dist
        
    def getAddr(self):
        return self.addr
    
    def getRssi(self):
        return self.rssi
    
    def getPort(self):
        return self.port
    
    def getToSinkDist(self):
        return self.tosinkdist

    def __str__(self) -> str:
        return f'Dist: {self.dist} Addr: {self.addr.__str__()} RSSI: {self.rssi} Port: {self.port} ToSinkDist: {self.tosinkdist}'
    # def getColorInt(self):
    #     return getColorInt(self.color)

    # def getColorVal(self):
    #     return self.color
