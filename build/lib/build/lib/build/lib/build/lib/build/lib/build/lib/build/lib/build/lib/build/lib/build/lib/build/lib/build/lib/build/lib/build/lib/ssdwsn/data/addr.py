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

from ssdwsn.util.constants import Constants as ct
from ssdwsn.util.utils import mergeBytes
from ctypes import ArgumentError, c_uint32 as unsigned_int32

class Addr:
    """Node Address Object"""
    def __init__(self, addr):
        """Generate a node address (address is two bytes )
            since all nodes running in the same virtual/physical machine
            the address of a node is two bytes with a maximum broadcasting address 255.255
            if adding the subnet id the address is three bytes <subnet>.<addr0>.<addr1>
        Args:
            addr (bytearray): one byte of subset of the node address
        """
        self.addr = bytearray(2)
        if type(addr) is int:
            self.addr[0] = addr >> 8
            self.addr[1] = addr & 0xff
        elif type(addr) is bytearray or type(addr) is bytes:
            self.addr[0] = addr[0]
            self.addr[1] = addr[1]
        elif type(addr) is str:
            tmp = addr.split('.')
            if len(tmp) == 2:
                self.addr[0] = int(tmp[0])
                self.addr[1] = int(tmp[1])
            else: 
                self.addr[0] = int(tmp[0]) >> 8
                self.addr[1] = int(tmp[0]) & 0xff
    
    def getArray(self):
        return self.addr
    
    def getHigh(self):
        return self.addr[0]
    
    def getLow(self):
        return self.addr[1]
    
    def hashCode(self):
        return hash(self.intValue())

    def intValue(self):
        return mergeBytes(self.addr[0], self.addr[1])

    def isBroadcast(self):
        return self.__eq__(Addr(ct.BROADCAST_ADDR))
    
    def __eq__(self, other):
        return (self.intValue() == other.intValue())

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return (self.intValue() < other.intValue())
            
    def __str__(self) -> str: return "{}.{}".format(self.addr[0],self.addr[1])
