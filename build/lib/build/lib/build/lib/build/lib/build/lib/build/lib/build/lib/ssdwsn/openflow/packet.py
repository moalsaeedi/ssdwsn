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

from ctypes import ArgumentError, c_uint32 as unsigned_int32
from enum import Enum
from socket import inet_pton
from math import ceil, floor, modf
from collections import OrderedDict
import ipaddress
import copy
import struct
import time

from ctypes import Array
from logging import setLogRecordFactory, warn
from ssdwsn.openflow.entry import Entry
from ssdwsn.openflow.window import Window
from ssdwsn.util.utils import mergeBytes, ipStr, ip6Str, macColonHex, _colonHex, long_to_bytes, byteToStr, bitsToBytes, getBitRange, setBitRange
from ssdwsn.data.addr import Addr
from ssdwsn.util.constants import Constants as ct

class Packet(object):
    """Packet abstract object"""
    def __init__(self, data:bytearray=None, net:int=None, src:Addr=None, dst:Addr=None):
        """Initiate a packet

        Args:
            data (bytearray, optional): array of bytes contains the packet header and payload. Defaults to None.
            net (int, optional): network subnet id. Defaults to None.
            src (Addr, optional): source node address. Defaults to None.
            dst (Addr, optional): destination node address. Defaults to None.
        """
        self.data = bytearray(ct.DFLT_HDR_LEN + ct.DFLT_PAYLOAD_LEN)
        if not(data is None):
            self.setArray(data)
        else:
            self.setTtl(ct.TTL_MAX)
            self.setLen(ct.DFLT_HDR_LEN)
            self.setNet(net)
            self.setSrc(src)
            self.setDst(dst)
            self.setTS(round(time.time(), 4))
    
    def setArray(self, data):
        if self.isssdwsnPacket():
            if(len(data) <= ct.MTU and len(data) >= ct.DFLT_HDR_LEN):
                self.setNet(data[ct.NET_INDEX])
                self.setLen(data[ct.LEN_INDEX])
                self.setSrc(Addr(data[ct.SRC_INDEX:ct.SRC_INDEX + ct.SRC_LEN])) #source addr
                self.setDst(Addr(data[ct.DST_INDEX:ct.DST_INDEX + ct.DST_LEN])) #destination addr
                self.setType(data[ct.TYP_INDEX])
                self.setTtl(data[ct.TTL_INDEX])
                self.setNxh(Addr(data[ct.NXH_INDEX:ct.NXH_INDEX + ct.NXH_LEN])) #next-hop addr
                self.setPrh(Addr(data[ct.PRH_INDEX:ct.PRH_INDEX + ct.PRH_LEN])) #prev-hop addr
                self.setTS(float(int.from_bytes(data[ct.TS_INDEX:ct.TS_INDEX+ct.TS_LEN-2], 'big', signed=False)) + 
                           ((int.from_bytes(data[ct.TS_INDEX+ct.TS_LEN-2:ct.TS_INDEX+ct.TS_LEN], 'big', signed=False))/10000))
                self.setPayload(data[ct.DFLT_HDR_LEN:])
            else:
                warn('Packet length (%s) is greater/lower than expected'%len(data))
        else:
            self.data[0:len(data)] = data[0:len(data)]
        return self

    def getLen(self):
        return self.data[ct.LEN_INDEX]

    def setLen(self, val):
        if val <= ct.MTU and val > 0:
            self.data[ct.LEN_INDEX] = val
        else: warn("Invalid packet length: ", val)
        return self

    def getNet(self):
        return self.data[ct.NET_INDEX]
    
    def setNet(self, val):
        self.data[ct.NET_INDEX] = val
        return self
        
    def getSrc(self):
        return Addr(self.data[ct.SRC_INDEX:ct.SRC_INDEX + ct.SRC_LEN]) #1
    
    def setSrc(self, val:Addr):
        self.data[ct.SRC_INDEX] = val.getHigh()
        self.data[ct.SRC_INDEX + 1] = val.getLow()
        return self
    
    def getDst(self):
        return Addr(self.data[ct.DST_INDEX:ct.DST_INDEX + ct.DST_LEN]) #1

    def setDst(self, val:Addr):
        self.data[ct.DST_INDEX] = val.getHigh()
        self.data[ct.DST_INDEX + 1] = val.getLow()
        return self
            
    def getType(self):
        return self.data[ct.TYP_INDEX]
    
    def getTypeName(self):
        return {
            0: 'DATA',
            1: 'BEACON',
            2: 'REPORT',
            3: 'REQUEST',
            4: 'RESPONSE',
            5: 'OPEN_PATH',
            6: 'CONFIG',
            7: 'REG_PROXY',
            8: 'AGRR'
        }.get(self.data[ct.TYP_INDEX], "")
    
    def setType(self, val):
        self.data[ct.TYP_INDEX] = val
        return self

    def getTtl(self):
        return self.data[ct.TTL_INDEX]
    
    def setTtl(self, val):
        self.data[ct.TTL_INDEX] = val
        return self
        
    def decrementTtl(self):
        if self.data[ct.TTL_INDEX] > 0:
            self.data[ct.TTL_INDEX] -= 1

    def getNxh(self):
        return Addr(self.data[ct.NXH_INDEX:ct.NXH_INDEX + ct.NXH_LEN]) #1

    def setNxh(self, val:Addr):
        self.data[ct.NXH_INDEX] = val.getHigh()
        self.data[ct.NXH_INDEX + 1] = val.getLow()
        return self

    def getPrh(self):
        return Addr(self.data[ct.PRH_INDEX:ct.PRH_INDEX + ct.PRH_LEN]) #1

    def setPrh(self, val:Addr):
        self.data[ct.PRH_INDEX] = val.getHigh()
        self.data[ct.PRH_INDEX + 1] = val.getLow()
        return self
    
    def getTS(self):
        realnum = int.from_bytes(self.data[ct.TS_INDEX:ct.TS_INDEX+ct.TS_LEN-2], 'big', signed=False)
        floatnum = int.from_bytes(self.data[ct.TS_INDEX+ct.TS_LEN-2:ct.TS_INDEX+ct.TS_LEN], 'big', signed=False)
        return realnum + (floatnum/10000)
    
    def setTS(self, val:float):        
        floatnum, realnum = modf(val)
        self.data[ct.TS_INDEX:ct.TS_INDEX+ct.TS_LEN-2] = int(realnum).to_bytes(4, 'big')
        self.data[ct.TS_INDEX+ct.TS_LEN-2:ct.TS_INDEX+ct.TS_LEN] = int(floatnum*10000).to_bytes(2, 'big')
        return self
    
    def getPayloadSize(self):
        return self.getLen() - ct.DFLT_HDR_LEN
    
    def setPayloadSize(self, size):
        if size <= ct.DFLT_PAYLOAD_LEN:
            self.setLen(ct.DFLT_HDR_LEN + size)
        else: warn("Cannot be greater than the maximum payload size: ", ct.DFLT_PAYLOAD_LEN)
        return self

    def getPayload(self):
        return self.data[ct.DFLT_HDR_LEN:self.getLen()]

    def setPayload(self, payload):
        if len(payload) <= ct.DFLT_PAYLOAD_LEN:
            self.data[ct.DFLT_HDR_LEN: len(payload)] = payload
            self.setLen(len(payload) + ct.DFLT_HDR_LEN)
        else: warn("Payload exceeds packet size")
        return self

    def getPayloadValue(self, index):
        if index < ct.DFLT_PAYLOAD_LEN:
            return self.data[ct.DFLT_HDR_LEN + index]
        else: warn("Cannot be greater than the payload size: ", ct.DFLT_PAYLOAD_LEN)
        
    def setPayloadValue(self, index, val):
        if index < ct.DFLT_PAYLOAD_LEN:
            self.data[ct.DFLT_HDR_LEN + index] = val
            if index+1 > (self.getLen() - ct.DFLT_HDR_LEN):
                self.setLen(self.getLen() + 1)
        else: warn("Cannot be greater than the maximum payload size: ", ct.DFLT_PAYLOAD_LEN)
        return self
    
    def getPayloadFromTo(self, start, to):
        if to < 0: 
            return warn('Stop must be greater than 0')
        if (ct.DFLT_HDR_LEN + to) > self.getLen():
            return warn('Stop is greater than packet size')
        return self.data[ct.DFLT_HDR_LEN+start:ct.DFLT_HDR_LEN+to]
            
    def setPayloadFromTo(self, src, src_start, payload_start, length):
        if src_start < 0 or payload_start < 0 or length < 0:
            return warn("Invalide slice")
        if payload_start + (length-src_start) <= ct.DFLT_PAYLOAD_LEN:
            if ct.DFLT_HDR_LEN + payload_start + (length-src_start) <= self.getLen():
                self.data[ct.DFLT_HDR_LEN+payload_start: ct.DFLT_HDR_LEN+payload_start+(length-src_start)] = src[src_start:length]
            else:
                self.setLen(ct.DFLT_HDR_LEN+payload_start+(length-src_start))
                self.data[ct.DFLT_HDR_LEN+payload_start: ct.DFLT_HDR_LEN+payload_start+(length-src_start)] = src[src_start:length]
        else: warn("Cannot be greater than the maximum payload size: ", ct.DFLT_PAYLOAD_LEN)
        # for i in range(length):
        #     if i < len(src):
        #         self.setPayloadValue(payload_start+i, src[src_start+i])
        #     elif payload_start+i >= (self.getLen() - ct.DFLT_HDR_LEN):
        #         self.setLen(self.getLen() + 1)
        return self

    def toIntArray(self):
        return [int(x) for x in self.data]
    
    def toByteArray(self):
        return self.data[:self.getLen()]
    
    def clone(self):
        return copy.deepcopy(self)    
        
    def isssdwsnPacket(self):
        return self.data[ct.NET_INDEX] < ct.THRES
        
"""
data: can be int[], bytes or object of Packet
"""        
class BeaconPacket(Packet):
    """Beacon Pakcet"""
    def __init__(self, data:bytearray=None, net:int=None, src:Addr=None, dst:Addr=None, distance:int=None, battery:int=None,
                    pos:tuple=None, intfType:int=None, sensorType:int=None, port:int=None, aggrpayload:bytearray=None):
        """Initiate a beacon packet

        Args:
            data (bytearray, optional): array of bytes contains the packet header and payload. Defaults to None.
            net (int, optional): network subnet id. Defaults to None.
            src (Addr, optional): source node address. Defaults to None.
            dst (Addr, optional): destination node address. Defaults to None.
            distance (int, optional): _description_. Defaults to None.
            battery (int, optional): battery change level. Defaults to None.
            pos (tuple, optional): current node position (x,y,z). Defaults to None.
            intfType (int, optional): type of physical interface IEEE802.15.4. Defaults to None.
            sensorType (int, optional): sensor type (temperature, humidity, etc. or sink/gateway). Defaults to None.
            port (int, optional): _description_. Defaults to None.
        """
        super().__init__(data, net, src, Addr(ct.BROADCAST_ADDR))        
        if data is None:
            self.setType(ct.BEACON)
            self.setLen(ct.DFLT_HDR_LEN+ct.BEACON_HDR_LEN) #TODO make the beacon packet as 12bytes only
            self.setDst(dst)
            self.setDistance(distance)
            self.setBattery(battery)
            self.setPosition(pos)            
            self.setIntfType(intfType)    
            self.setSensorType(sensorType)        
            self.setPort(port)
            if aggrpayload:
                self.setPayloadFromTo(aggrpayload, 0, self.getPayloadSize(), len(aggrpayload))
    
    def getPosition(self):
        val = int.from_bytes(self.getPayloadFromTo(ct.POS_INDEX, ct.POS_INDEX + ct.POS_LEN), 'big', signed=False)
        x = ( val >> 16 ) & 0xffff
        y = val & 0xffff
        z = 0
        pos = [x, y, z]        
        return (*[int(x) for x in pos],)
    
    def setPosition(self, pos:tuple):
        for i in range(len(pos)):
            if i <= 2: # z dimention is zero
                self.setPayloadFromTo(int.to_bytes(pos[i], 2, 'big', signed=False), 0, ct.POS_INDEX+(i*int(ct.POS_LEN/2)), int(ct.POS_LEN/2))
        return self

    def getSensorType(self):
        val = int.from_bytes(self.getPayloadFromTo(ct.INFO_INDEX, ct.INFO_INDEX+ct.INFO_LEN), 'big', signed=False)
        return getBitRange(val, ct.TYP_BIT_INDEX, ct.TYP_BIT_LEN)
    
    def setSensorType(self, val:int):
        new_val = setBitRange(int.from_bytes(self.getPayloadFromTo(ct.INFO_INDEX, ct.INFO_INDEX+ct.INFO_LEN), byteorder='big', signed=False), ct.TYP_BIT_INDEX, ct.TYP_BIT_LEN, int(val))        
        self.setPayloadFromTo(bitsToBytes('{0:b}'.format(new_val).zfill(8*ct.INFO_LEN)), 0, ct.INFO_INDEX, ct.INFO_LEN)  
        return self
    
    def getIntfType(self):
        val = int.from_bytes(self.getPayloadFromTo(ct.INFO_INDEX, ct.INFO_INDEX+ct.INFO_LEN), 'big', signed=False)
        return getBitRange(val, ct.INTF_BIT_INDEX, ct.INTF_BIT_LEN)
    
    def setIntfType(self, val:int):
        new_val = setBitRange(int.from_bytes(self.getPayloadFromTo(ct.INFO_INDEX, ct.INFO_INDEX+ct.INFO_LEN), byteorder='big', signed=False), ct.INTF_BIT_INDEX, ct.INTF_BIT_LEN, int(val))        
        self.setPayloadFromTo(bitsToBytes('{0:b}'.format(new_val).zfill(8*ct.INFO_LEN)), 0, ct.INFO_INDEX, ct.INFO_LEN)  
        return self
       
    def getPort(self):
        return ct.BASE_NODE_PORT + int.from_bytes(self.getPayloadFromTo(ct.PORT_INDEX, ct.PORT_INDEX+ ct.PORT_LEN), byteorder='big', signed=False)
    
    def setPort(self, port:int):
        self.setPayloadFromTo(bitsToBytes('{0:b}'.format(port).zfill(8*ct.PORT_LEN)), 0, ct.PORT_INDEX, ct.PORT_LEN)
        return self
        
    def getDistance(self):
        dist = self.getPayloadValue(ct.DIST_INDEX)
        return dist
    
    def setDistance(self, val):
        self.setPayloadValue(ct.DIST_INDEX, val)
        return self

    def getBattery(self):
        return int.from_bytes(self.getPayloadFromTo(ct.BATT_INDEX, ct.BATT_INDEX + ct.BATT_LEN), byteorder='big', signed=False)
    
    def setBattery(self, val:int):
        self.setPayloadFromTo(bitsToBytes('{0:b}'.format(val).zfill(8*ct.BATT_LEN)), 0, ct.BATT_INDEX, ct.BATT_LEN)
        return self

    def hasAggrConfPayload(self):
        return bool(self.getPayloadSize() > ct.BEACON_HDR_LEN)

    def getAggrConfPayload(self):
        return self.getPayloadFromTo(ct.BEACON_HDR_LEN, self.getPayloadSize())

    def getAggrConfig(self):
        payload = self.getPayloadFromTo(ct.BEACON_HDR_LEN, self.getPayloadSize())
        return {'dist':payload[0], 'dst':Addr(payload[1:3])}

class ConfigProperty(Enum):
    
    RESET = [0, 0]
    DIST = [1, 1]
    MY_ADDRESS = [2, 2]
    IS_AGGR = [3, 1]
    GET_RULE = [4, 1]
    REM_RULE = [5, -1]
    ADD_RULE = [6, -1]
    DRL_ACTION = [7, -1]
    
    def getValue(self):
        return self.value[0]

    def getSize(self):
        return self.value[1]
    
    @classmethod
    def fromString(cls, str):
        return bytes(cls(str).value)
    
    @classmethod
    def fromByte(cls, value):
        return cls([unsigned_int32(v).value for v in value]).__class__
    
    def __new__(cls, *values):
        obj = object.__new__(cls)
        # first value is canonical value
        obj._value_ = values[0]
        for other_value in values[1:]:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values
        return obj

    def __repr__(self):
        return '<%s.%s: %s>' % (
                self.__class__.__name__,
                self._name_,
                ', '.join([repr(v) for v in self._all_values]),
                )
    
    # def __str__(self):
    #     return 'Action type of {}'.format(self.name)
            
class ConfigPacket(Packet):
    """Configuration Packet"""
    def __init__(self, data:bytearray=None, net:int=None, src:Addr=None, dst:Addr=None, read:ConfigProperty=None, write:ConfigProperty=None, val:bytearray=None, path:list=None):
        """Initiate a configuration packet

        Args:
            data (bytearray, optional): array of bytes contains the packet header and payload. Defaults to None.
            net (int, optional): network subnet id. Defaults to None.
            src (Addr, optional): source node address. Defaults to None.
            dst (Addr, optional): destination node address. Defaults to None.
            read (ConfigProperty, optional): read a configuration. Defaults to None.
            write (ConfigProperty, optional): write a configuration. Defaults to None.
            val (bytearray, optional): array of bytes contains the configuration value. Defaults to None.
        """
        super().__init__(data=data, net=net, src=src, dst=dst)
        if data is None:
            self.setType(ct.CONFIG)
            self.setLen(ct.DFLT_HDR_LEN + 1 + len(path)*2 + 1) # header+path_size+path+config_id
            self.setPath(path)
        if read:
            self.setConfigId(read)
        elif write:
            self.setConfigId(write).setWrite().setParams(val, write.getSize())

    def getConfIndex(self):
        return self.getPayloadValue(ct.CNF_PATH_INDEX)*2 + 1 # get length of path in bytes + 1 (index 0 that contains the length of path)

    def isWrite(self):
        i = self.getConfIndex()
        return (self.getPayloadValue(i) >> ct.CNF_MASK_POS) == ct.CNF_WRITE

    def setWrite(self):
        i = self.getConfIndex()
        self.setPayloadValue(i, self.getPayloadValue(i) | (ct.CNF_WRITE << ct.CNF_MASK_POS))
        return self

    def getConfigId(self):
        i = self.getConfIndex()
        return self.getPayloadValue(i) & ct.CNF_MASK   
    
    def setConfigId(self, cls:ConfigProperty):        
        i = self.getConfIndex()
        self.setPayloadValue(i, cls.getValue())
        return self

    def getParams(self):
        i = self.getConfIndex()
        return self.getPayloadFromTo(i+1, self.getPayloadSize())

    def setParams(self, values, size):
        i = self.getConfIndex()
        if size != -1:
            self.setLen(self.getLen()+size)
            for j in range(size):
                self.setPayloadValue(i+j+1, values[j])
        else:
            self.setLen(self.getLen()+len(values))
            for j in range(len(values)):
                self.setPayloadValue(i+j+1, values[j])
        return self

    def getConfigPayload(self):
        i = self.getConfIndex()
        payload = self.getPayload()
        return payload[i:]

    def getConfigPayloadSize(self):
        return self.getPayloadSize() - self.getConfIndex()

    def getPath(self):
        path = []
        payload = self.getPathPayload()
        i = 1
        for _ in range(int(len(payload)/2)):
            path.append(Addr(payload[i:i+2]))
            i += 2
        return path
    
    def setPath(self, path:list):
        self.setPayloadValue(ct.CNF_PATH_INDEX, len(path))
        i = ct.CNF_PATH_INDEX+1
        for p in path:
            self.setPayloadValue(i, p.getHigh())
            self.setPayloadValue(i+1, p.getLow())
            i += 2

    def getPathPayload(self):  
        i = self.getConfIndex()
        payload = self.getPayload()
        return payload[ct.CNF_PATH_INDEX:i]  

    def getPathPayloadSize(self):
        return self.getConfIndex()

class DataPacket(Packet):
    """Data Packet"""
    def __init__(self, data:bytearray=None, net:int=None, src:Addr=None, dst:Addr=None, payload:bytearray=None):
        """Generate a data packet

        Args:
            data (bytearray, optional): array of bytes contains the packet header and payload. Defaults to None.
            net (int, optional): network subnet id. Defaults to None.
            src (Addr, optional): source node address. Defaults to None.
            dst (Addr, optional): destination node address. Defaults to None.
            payload (bytearray, optional): array of bytes contains the data packet payload. Defaults to None.
        """
        super().__init__(data=data, net=net, src=src, dst=dst)
        if data is None:
            self.setType(ct.DATA)
            self.setLen(ct.DFLT_HDR_LEN+len(payload))
            self.setPayload(payload)       

    def getData(self):
        return self.getPayload()

class OpenPathPacket(Packet):    
    """Open Path Packet"""
    def __init__(self, data:bytearray=None, net:int=None, src:Addr=None, dst:Addr=None, path:list=None):
        """Generate an open-path (source route) packet 

        Args:
            data (bytearray, optional): array of bytes contains the packet header and payload. Defaults to None.
            net (int, optional): network subnet id. Defaults to None.
            src (Addr, optional): source node address. Defaults to None.
            dst (Addr, optional): destination node address. Defaults to None.
            path (list, optional): a list contains the calculated routing path. Defaults to None.
        """
        super().__init__(data=data, net=net, src=src, dst=dst)
        if data is None:
            self.setType(ct.OPEN_PATH)
            self.setLen(ct.MTU)
            self.setPayloadValue(ct.OP_WINS_SIZE_INDEX, 0)
            self.setPath(path)

    def getPath(self):
        path = []
        payload = self.getPathPayload()
        i = 0
        for _ in range(int(len(payload)/2)):
            path.append(Addr(payload[i:i+2]))
            i += 2
        return path
    
    def setPath(self, path:list):
        self.setLen(ct.DFLT_HDR_LEN+(self.getPayloadValue(ct.OP_WINS_SIZE_INDEX) * ct.W_SIZE) + 1 + len(path)*2)
        i = (self.getPayloadValue(ct.OP_WINS_SIZE_INDEX) * ct.W_SIZE) + 1
        for p in path:
            self.setPayloadValue(i, p.getHigh())
            self.setPayloadValue(i+1, p.getLow())
            i += 2    
            
    def getWindows(self):
        windows = []
        n = self.getPayloadValue(ct.OP_WINS_SIZE_INDEX)
        for i in range(n-1):
            windows.append(Window(self.getPayloadFromTo(ct.OP_WINS_SIZE_INDEX + 1 + i, ct.OP_WINS_SIZE_INDEX + 1 + ct.W_SIZE + i)))
            i = i + ct.W_SIZE
        return windows
    
    def setWindows(self, windows):
        tmp = self.getPath()
        self.setPayloadValue(ct.OP_WINS_SIZE_INDEX, len(windows))
        i = ct.OP_WINS_SIZE_INDEX + 1
        for w in windows:
            self.setPayloadFromTo(w, 0, i, len(w))
            i = i + len(w)
        self.setPath(tmp)
        
    def getPathPayload(self):        
        payload = self.getPayload()
        return payload[(self.getPayloadValue(ct.OP_WINS_SIZE_INDEX) * ct.W_SIZE) + 1:]
    
class RegProxyPacket(Packet):
    """Register a Proxy Packet"""
    def __init__(self, data:bytearray=None, net:int=None, src:Addr=None, dPid:str=None, mac:str=None, port:int=None, isa:tuple=None):
        """Generate a reg-proxy packet (for registring a sink (gateway) to a controller)

        Args:
            data (bytearray, optional): array of bytes contains the packet header and payload. Defaults to None.
            net (int, optional): network subnet id. Defaults to None.
            src (Addr, optional): source node address. Defaults to None.
            dst (Addr, optional): destination node address. Defaults to None.
            dPid (str, optional): the process id of the sink (gateway) node. Defaults to None.
            mac (str, optional): the mac address of the sink (gateway) node. Defaults to None.
            port (int, optional): the port address of the sink (gateway) node. Defaults to None.
            isa (tuple, optional): the url address of the controller. Defaults to None.
        """
        super().__init__(data=data, net=net, src=src, dst=src)
        if data is None:
            self.setType(ct.REG_PROXY)
            self.setLen(ct.DFLT_HDR_LEN+ct.REG_HDR_LEN)
            self.setDpid(dPid)
            self.setMac(mac)
            self.setPort(port)        
            self.setNxh(src)
            self.setInetSocketAddress(isa)
        
    def getDpid(self):
        # str(self.getPayloadFromTo(ct.REG_DPID_INDEX, ct.REG_MAC_INDEX), 'UTF-8')
        return self.getPayloadFromTo(ct.REG_DPID_INDEX, ct.REG_DPID_INDEX+ct.DPID_LEN).decode()
    
    def setDpid(self, dpid:str):
        # bytearray(int(d) for d in dpid)
        self.setPayloadFromTo(dpid.encode(), 0, ct.REG_DPID_INDEX, ct.DPID_LEN)

    def getInetSocketAddress(self):
        try:
            host = '.'.join(str(x) for x in self.getPayloadFromTo(ct.REG_IP_INDEX, ct.REG_IP_INDEX + ct.IP_LEN))
            port = int.from_bytes(self.getPayloadFromTo(ct.REG_TCP_INDEX, ct.REG_TCP_INDEX+2), byteorder='big', signed=False)
            return (host, port)
        except: return warn("Unknown Host")

    def setInetSocketAddress(self, isa:tuple):
        i = 0
        for d in isa[0].split('.'):
            self.setPayloadFromTo(int.to_bytes(int(d), 1, 'big', signed=False), 0, ct.REG_IP_INDEX+i, int(ct.IP_LEN/4))
            i += 1
        self.setPayloadFromTo(bitsToBytes('{0:08b}'.format(isa[1]).zfill(2*8)), 0, ct.REG_TCP_INDEX, 2)
        
    def getMac(self):
        val = self.getPayloadFromTo(ct.REG_MAC_INDEX, ct.REG_MAC_INDEX + ct.MAC_LEN)
        return ':'.join( '{:02x}'.format(x) for x in val) 

    def setMac(self, mac:str):
        args = [ arg for arg in mac.split(':') ]
        for i in range(ct.MAC_LEN):
            self.setPayloadValue(ct.REG_MAC_INDEX + i, int(args[i], 16))

    def getPort(self):
        return int.from_bytes(self.getPayloadFromTo(ct.REG_PORT_INDEX, ct.REG_PORT_INDEX + ct.PORT_LEN), byteorder='big', signed=False)
        # b = self.getPayloadFromTo(ct.REG_PORT_INDEX, ct.REG_PORT_INDEX + ct.PORT_LEN)
        # print(b)
        ## b.reverse()
        # bigInt = 0
        # for byte in b:
        #     bigInt <<= 8
        #     bigInt |= byte 
        # return bigInt
    
    def setPort(self, port:int):
        self.setPayloadFromTo(bitsToBytes('{0:b}'.format(port).zfill(8*ct.PORT_LEN)), 0, ct.REG_PORT_INDEX, ct.PORT_LEN)

class ReportPacket(BeaconPacket):
    """Report Packet"""
    def __init__(self, data:bytearray=None, net:int=None, src:Addr=None, dst:Addr=None, distance:int=None, battery:int=None,
                    pos:tuple=None, intfType:int=None, sensorType:int=None, port:int=None):
        """Generate a report packet (for reporting changes in the topology to the controller

        Args:
            data (bytearray, optional): array of bytes contains the packet header and payload. Defaults to None.
            net (int, optional): network subnet id. Defaults to None.
            src (Addr, optional): source node address. Defaults to None.
            dst (Addr, optional): destination node address. Defaults to None.
            distance (int, optional): _description_. Defaults to None.
            battery (int, optional): battery charging level. Defaults to None.
            pos (tuple, optional): node position (x,y,z). Defaults to None.
            intfType (int, optional): physical interface type IEEE802.15.4. Defaults to None.
            sensorType (int, optional): sensor type attached to the node (temperature, humidity, etc.). Defaults to None.
            port (int, optional): _description_. Defaults to None.
        """
        super().__init__(data, net, src, dst, distance, battery, pos, intfType, sensorType, port)
        if data is None:
            self.setType(ct.REPORT)                
            self.setLen(ct.DFLT_HDR_LEN+ct.REPORT_HDR_LEN)
            self.setDst(dst)
            self.setNeighborsSize(0)

    def getNeighborsSize(self):
        return self.getPayloadValue(ct.NEIGH_INDEX)
    
    def setNeighborsSize(self, size:int):
        if size <= int(ct.MAX_NEIG):
            self.setPayloadValue(ct.NEIGH_INDEX, size)
            self.setPayloadSize(self.getPayloadSize() + (ct.NEIGH_LEN * size))
        else: warn("Too many neighbors")
        return self

    def getNeighbors(self):
        map = []
        n = self.getNeighborsSize()
        for i in range(n):
            # map.append({'addr': self.getNeighborAddress(i), 'rssi': self.getLinkQuality(i), 'color': self.getLinkColor(i)})
            map.append({'addr': self.getNeighborAddress(i), 'rssi': self.getLinkQuality(i)})
        return map

    def setNeighbors(self, map=None):
        i = 0
        self.setNeighborsSize(len(map))
        for key in map:
            self.setNeighborAddress(i, map[key].getAddr())
            self.setLinkQuality(i, map[key].getRssi())
            # self.setLinkColor(i, map[key].getColorInt())
            i += 1
        return self
    
    def getNeighborAddress(self, i_node):
        if i_node in range(self.getNeighborsSize()):
            addr = bytearray(2)            
            addr[0] = self.getPayloadValue(ct.NEIGH_INDEX + 1 + i_node * ct.NEIGH_LEN)
            addr[1] = self.getPayloadValue(ct.NEIGH_INDEX + 2 + i_node * ct.NEIGH_LEN)
            return Addr(addr)
        else: return warn("Index exceeds max number of neighbors")

    def setNeighborAddress(self, i_node, addr:Addr):
        if i_node in range(self.getNeighborsSize()):
            self.setPayloadValue((ct.NEIGH_INDEX + 1 + i_node * ct.NEIGH_LEN), addr.getHigh())
            self.setPayloadValue((ct.NEIGH_INDEX + 2 + i_node * ct.NEIGH_LEN), addr.getLow())
            # if i_node+1 > self.getNeighborsSize():
            #     self.setNeighborsSize(i_node+1)
        else: warn("Index exceeds max number of neighbors")
        return self
    
    def addNeighborAddress(self, addr:Addr, rssi:int=-50):
        size = self.getNeighborsSize()
        if size <= ct.MAX_NEIG:
            self.setPayloadValue((ct.NEIGH_INDEX + 1 + self.getNeighborsSize() * ct.NEIGH_LEN), addr.getHigh())
            self.setPayloadValue((ct.NEIGH_INDEX + 2 + self.getNeighborsSize() * ct.NEIGH_LEN), addr.getLow())
            self.setPayloadValue((ct.NEIGH_INDEX + 3 + self.getNeighborsSize() * ct.NEIGH_LEN), rssi)
            self.setNeighborsSize(size+1)
        else: warn("Index exceeds max number of neighbors")
        return self    

    def getLinkQuality(self, i_node):
        if i_node in range(self.getNeighborsSize()):
            return self.getPayloadValue(ct.NEIGH_INDEX + 3 + i_node * ct.NEIGH_LEN)
        else: return warn("Index exceeds max number of neighbors")

    def setLinkQuality(self, i_node, val):
        if i_node in range(self.getNeighborsSize()):
            self.setPayloadValue(ct.NEIGH_INDEX + 3 + i_node * ct.NEIGH_LEN, val)
        else: warn("Index exceeds max number of neighbors")

    # def getLinkColor(self, i_node):
    #     if i_node in range(self.getNeighborsSize()):
    #         return self.getPayloadValue(ct.NEIGH_INDEX + 4 + i_node * ct.NEIGH_LEN)
    #     else: return warn("Index exceeds max number of neighbors")

    # def setLinkColor(self, i_node, val):
    #     if i_node in range(self.getNeighborsSize()):
    #         self.setPayloadValue(ct.NEIGH_INDEX + 4 + i_node * ct.NEIGH_LEN, val)
    #     else: warn("Index exceeds max number of neighbors")   

    def getAggrPayload(self, prvpacket):
        # print('prvpacket: \n') 
        # print(' '.join(format(i, '08b') for i in prvpacket)) 
        # print('\n')
        # print('packet: \n') 
        # print(' '.join(format(i, '08b') for i in self.toByteArray())) 
        # print('\n')
        prvpacket = ReportPacket(prvpacket)
        payload = self.getSrc().getArray()+self.data[ct.TS_INDEX:ct.TS_INDEX+ct.TS_LEN]
        aggrhsize = len(payload)
        if self.getDistance() != prvpacket.getDistance():
            payload += int.to_bytes(ct.DIST_INDEX, 1, 'big', signed=False) + self.getPayloadValue(ct.DIST_INDEX)
        if self.getBattery() != prvpacket.getBattery():
            payload += int.to_bytes(ct.BATT_INDEX, 1, 'big', signed=False) + self.getPayloadFromTo(ct.BATT_INDEX, ct.BATT_INDEX+ct.BATT_LEN)
        if self.getPosition() != prvpacket.getPosition():
            payload += int.to_bytes(ct.POS_INDEX, 1, 'big', signed=False) + self.getPayloadFromTo(ct.POS_INDEX, ct.POS_INDEX+ct.POS_LEN)
        if self.getIntfType() != prvpacket.getIntfType():
            payload += int.to_bytes(ct.INFO_INDEX, 1, 'big', signed=False) + self.getPayloadValue(ct.INFO_INDEX)
        if self.getPort() != prvpacket.getPort():
            payload += int.to_bytes(ct.PORT_INDEX, 1, 'big', signed=False) + self.getPayloadFromTo(ct.PORT_INDEX, ct.PORT_INDEX+ct.PORT_LEN)
        # check node's neighbors reporting
        #TODO
        ngs = self.getNeighbors()
        ngs = {ng['addr'].__str__(): ng['rssi'] for ng in ngs}
        # print(ngs)
        prvngs = prvpacket.getNeighbors()
        prvngs = {prvng['addr'].__str__(): prvng['rssi'] for prvng in prvngs}
        # print(prvngs)
        rmngs = {}
        tmppl = bytearray()
        counter = 0
        for addr, rssi in ngs.items():
            if prvngs.get(addr):
                if rssi != prvngs[addr]:
                    tmppl += Addr(addr).getArray() + int.to_bytes(rssi, 1, 'big', signed=False)
                    counter += 1
            else: 
                tmppl += Addr(addr).getArray() + int.to_bytes(rssi, 1, 'big', signed=False)
                counter += 1
        for addr, rssi in prvngs.items():
            if not ngs.get(addr):
                tmppl += Addr(addr).getArray() + int.to_bytes(0, 1, 'big', signed=False)
                counter += 1
        if len(tmppl) != 0:
            payload += int.to_bytes(ct.NEIGH_INDEX, 1, 'big', signed=False) + int.to_bytes(counter, 1, 'big', signed=False) + tmppl
        payload = int.to_bytes(len(payload), 1, 'big', signed=False) + payload
        # print(f'ngs_size: {self.getNeighborsSize()} ngs: {ngs}\n')
        # print(f'prvngs_size {prvpacket.getNeighborsSize()} prvngs: {prvngs}\n')
        # print(f'rmngs: {rmngs}\n')
        # print('result: \n')
        # print(' '.join(format(i, '08b') for i in payload))
        return None if len(payload) == aggrhsize+1 else payload

class RequestPacket(Packet):
    """Request Packet"""
    def __init__(self, data:bytearray=None, net:int=None, src:Addr=None, dst:Addr=None, id:int=None, part:int=None, total:int=None, reqPayload:bytearray=None):
        """Generate a request packet (to request a routing rule for unknown arrived packet)

        Args:
            data (bytearray, optional): array of bytes contains the packet header and payload. Defaults to None.
            net (int, optional): network subnet id. Defaults to None.
            src (Addr, optional): source node address. Defaults to None.
            dst (Addr, optional): destination node address. Defaults to None.
            id (int, optional): _description_. Defaults to None.
            part (int, optional): _description_. Defaults to None.
            total (int, optional): _description_. Defaults to None.
            reqPayload (bytearray, optional): _description_. Defaults to None.
        """
        super().__init__(data=data, net=net, src=src, dst=dst)
        if data is None:
            self.setType(ct.REQUEST)
            self.setLen(ct.DFLT_HDR_LEN+ct.REQUEST_HDR_LEN)
            self.setId(id)
            self.setTotal(total)
            self.setPart(part)
            self.setReqPayload(reqPayload)

    def getReqPayload(self):
        return self.getPayloadFromTo(ct.TOTAL_INDEX + 1, self.getPayloadSize())

    def setReqPayload(self, reqPayload:bytearray):
        self.setPayloadFromTo(reqPayload, 0, ct.TOTAL_INDEX + 1, len(reqPayload))

    def getReqPayloadSize(self):
        return self.getPayloadSize() - (ct.TOTAL_INDEX + 1)    

    def getId(self):
        return self.getPayloadValue(ct.ID_INDEX)
    
    def setId(self, id:int):
        self.setPayloadValue(ct.ID_INDEX, id)

    def getPart(self):
        return self.getPayloadValue(ct.PART_INDEX)

    def setPart(self, part:int):
        self.setPayloadValue(ct.PART_INDEX, part)

    def getTotal(self):
        return self.getPayloadValue(ct.TOTAL_INDEX)
    
    def setTotal(self, total:int):
        self.setPayloadValue(ct.TOTAL_INDEX, total)

    @staticmethod
    def createReqPacket(net, src, dst, id, reqPayload):
        reqPackets = []
        n_packets = floor(len(reqPayload)/ct.REQUEST_PAYLOAD_SIZE)
        rem_bytes = len(reqPayload) % ct.REQUEST_PAYLOAD_SIZE
        for p in range(n_packets):
            reqPackets.append(RequestPacket(net=net, src=src, dst=dst, id=id, part=p+1, total=n_packets+1, reqPayload=reqPayload[p*ct.REQUEST_PAYLOAD_SIZE:(p+1)*ct.REQUEST_PAYLOAD_SIZE]))
        if rem_bytes != 0:
            reqPackets.append(RequestPacket(net=net, src=src, dst=dst, id=id, part=n_packets+1, total=n_packets+1, reqPayload=reqPayload[n_packets*ct.REQUEST_PAYLOAD_SIZE:len(reqPayload)]))
        return reqPackets
    
    @classmethod
    def mergeReqPackets(cls, rp1, rp2):
        if rp1.getPart() == 0:
            return DataPacket(rp1.getReqPayload().extend(rp2.getReqPayload()))
        else: return DataPacket(rp2.getReqPayload().extend(rp1.getReqPayload()))    
    
class ResponsePacket(Packet):
    """Response Packet"""
    def __init__(self, data:bytearray=None, net:int=None, src:Addr=None, dst:Addr=None, entry:Entry=None):
        """Generate response packet from the controller responding to a request from data plane

        Args:
            data (bytearray, optional): array of bytes contains the packet header and payload. Defaults to None.
            net (int, optional): network subnet id. Defaults to None.
            src (Addr, optional): source node address. Defaults to None.
            dst (Addr, optional): destination node address. Defaults to None.
            entry (Entry, optional): to be installed flow table entry. Defaults to None.
        """
        super().__init__(data=data, net=net, src=src, dst=dst)
        if data is None:
            self.setType(ct.RESPONSE)
            self.setRule(entry)
        
    def getRule(self):
        return Entry(self.getPayload())

    def setRule(self, entry:Entry):
        self.setPayload(entry.toByteArray())
                
class AggrPacket(Packet):
    """Aggregate Packet"""
    def __init__(self, data:bytearray=None, net:int=None, src:Addr=None, dst:Addr=None, aggrType:int=None, aggrPayload:bytearray=None):
        """Generate a Aggregate packet (to Aggregate packets)

        Args:
            data (bytearray, optional): array of bytes contains the packet header and payload. Defaults to None.
            net (int, optional): network subnet id. Defaults to None.
            src (Addr, optional): source node address. Defaults to None.
            dst (Addr, optional): destination node address. Defaults to None.
            type (int, optional): _description_. Defaults to None.
            aggrPayload (bytearray, optional): _description_. Defaults to None.
        """
        super().__init__(data=data, net=net, src=src, dst=dst)
        if data is None:
            self.setType(ct.AGGR)
            self.setLen(ct.DFLT_HDR_LEN+ct.AGGR_HDR_LEN)
            self.setAggrType(aggrType)
            self.setAggrPayload(aggrPayload)

    def getAggrPayload(self):
        return self.getPayloadFromTo(ct.AGGR_HDR_LEN , self.getPayloadSize())

    def setAggrPayload(self, aggrPayload:bytearray):
        self.setPayloadFromTo(aggrPayload, 0, ct.AGGR_HDR_LEN, len(aggrPayload))

    def getAggrPayloadSize(self):
        return self.getPayloadSize() - ct.AGGR_HDR_LEN
        
    def getAggrType(self):
        return self.getPayloadValue(ct.AGGR_TYPE_INDEX)

    def getAggrTypeName(self):
        return {
            0: 'DATA',
            1: 'BEACON',
            2: 'REPORT',
            3: 'REQUEST',
            4: 'RESPONSE',
            5: 'OPEN_PATH',
            6: 'CONFIG',
            7: 'REG_PROXY',
            8: 'AGRR'
        }.get(self.getPayloadValue(ct.AGGR_TYPE_INDEX), "")
        
    def setAggrType(self, type:int):
        self.setPayloadValue(ct.AGGR_TYPE_INDEX, type)

    @staticmethod
    def createAggrPacket(net, src, dst, type, aggr):
        aggrPayload = bytearray()
        for pl in aggr:
            # if type == ct.DATA:
            #     # reset sending packet timestamp to aggregation timestamp
            #     floatnum, realnum = modf(round(time.time(), 4))
            #     pl[2:6] = int(realnum).to_bytes(4, 'big')
            #     pl[6:8] = int(floatnum*10000).to_bytes(2, 'big')
            aggrPayload += pl
        return AggrPacket(net=net, src=src, dst=dst, aggrType=type, aggrPayload=aggrPayload)
    
if __name__ == '__main__':
    from ssdwsn.data.addr import Addr
    from ssdwsn.data.neighbor import Neighbor
    from ssdwsn.util.utils import getColorVal

    aggrMembs = {}
    cachedReports = []
    packet = ReportPacket(net=1, src=Addr('0.2'), dst=Addr('0.1'), distance=4, battery=65554, 
                            pos=(23,44,5), intfType=3, sensorType=2, port=65656)
    packet.setNeighbors({'1.0.3':Neighbor(packet.getSrc(), Addr('0.3'), 185, 65151, 65154), '1.0.4':Neighbor(packet.getSrc(), Addr('0.4'), 185, 65151, 65154)}) 
    print(' '.join(format(i, '08b') for i in packet.toByteArray()))
    src_id = f'{packet.getNet()}.{packet.getSrc()}'
    aggrMembs[src_id] = packet.toByteArray()
    packet = ReportPacket(net=1, src=Addr('0.2'), dst=Addr('0.1'), distance=4, battery=65322, 
                            pos=(23,44,5), intfType=3, sensorType=2, port=65656)
    packet.setNeighbors({'1.0.3':Neighbor(packet.getSrc(), Addr('0.3'), 185, 65151, 65154), '1.0.5':Neighbor(packet.getSrc(), Addr('0.5'), 185, 65151, 65154)}) 
    print(' '.join(format(i, '08b') for i in packet.toByteArray()))
    p = packet.getAggrPayload(aggrMembs[src_id])
    cachedReports.append(p)


    packet = ReportPacket(net=1, src=Addr('0.4'), dst=Addr('0.1'), distance=5, battery=65556, 
                            pos=(23,30,3), intfType=3, sensorType=2, port=65659)
    packet.setNeighbors({'1.0.5':Neighbor(packet.getSrc(), Addr('0.5'), 185, 65151, 65154), '1.0.6':Neighbor(packet.getSrc(), Addr('0.6'), 185, 65151, 65154)}) 
    print(' '.join(format(i, '08b') for i in packet.toByteArray()))
    src_id = f'{packet.getNet()}.{packet.getSrc()}'
    aggrMembs[src_id] = packet.toByteArray()
    packet = ReportPacket(net=1, src=Addr('0.4'), dst=Addr('0.1'), distance=5, battery=65222, 
                            pos=(23,30,3), intfType=3, sensorType=2, port=65659)
    packet.setNeighbors({'1.0.5':Neighbor(packet.getSrc(), Addr('0.5'), 185, 65151, 65154), '1.0.6':Neighbor(packet.getSrc(), Addr('0.6'), 185, 65151, 65154)}) 
    print(' '.join(format(i, '08b') for i in packet.toByteArray()))
    p = packet.getAggrPayload(aggrMembs[src_id])
    cachedReports.append(p)

    aggrP = AggrPacket.createAggrPacket(net=1, src=Addr('0.1'), dst=Addr('0.0'), type=ct.REPORT, aggr=cachedReports)
    # for sp in self.cachedReports[:-1]:
                        #     print(' '.join(format(i, '08b') for i in sp[8:])) 
                        #     print('\n')

    print(' '.join(format(i, '08b') for i in aggrP.toByteArray()))
    print('\n')
                                            
    # distance = int.from_bytes(int.to_bytes(255, 1, 'big', signed=False) + int.to_bytes(0, 1, 'big', signed=False), 'big', signed=False)
    # p = BeaconPacket(net=1, src=Addr('0.2'), dst=Addr('0.1'), distance=distance, battery=65554, 
    #                         pos=(23,44,5), intfType=3, sensorType=2, port=65656)

    # p = ReportPacket(net=1, src=Addr('0.2'), dst=Addr('0.1'), distance=distance, battery=65554, 
    #                         pos=(23,44,5), intfType=3, sensorType=2, port=65656)

    # p.setNeighborsSize(1)
    # p.setNeighborAddress(0, Addr('0.3'))
    # p.setLinkQuality(0, 185)
    # aggrpayload = None
    # isAggrHead = True
    # aggrpayload = bytearray(int.to_bytes(1, 1, 'big', signed=False)+int.to_bytes(0, 1, 'big', signed=False)+int.to_bytes(4, 1, 'big', signed=False))
    # p = BeaconPacket(net=1, src=Addr('0.2'), dst=Addr('0.1'), distance=4, battery=455412, 
    #                     pos=(23,44,5), intfType=3, sensorType=2, port=6662, aggrpayload=aggrpayload)
    # p.setNxh(Addr(ct.BROADCAST_ADDR))
    # print(' '.join(format(i, '08b') for i in p.toByteArray())) 
    # print('\n')  
    # print(p.hasAggrPayload())
    # print(p.getAggrConfig().get('distance'))
    # print(p.getAggrConfig().get('dst'))
    # print(' '.join(format(i, '08b') for i in p.getAggrPayload()))                                                     
    # print(p.getLen())

    # cp = ConfigPacket(net=1, src=Addr('0.0'), dst=Addr('0.4'), write=ConfigProperty.IS_AGGR, val=int(True).to_bytes(1, 'big'), path=[Addr('0.0'),Addr('0.1'), Addr('0.2'), Addr('0.4')])                     
    # print(' '.join(format(i, '08b') for i in cp.toByteArray()))                            
    # print(cp.getLen())
    # print(cp.getParams())
    # print(distance)
    # print(p.getDistance())

    # distance = 10
    # p = ReportPacket(net=1, src=Addr('0.2'), dst=Addr('0.1'), distance=distance, battery=6554, 
    #                         pos=(23,44,0), intfType=3, sensorType=2, port=55)


    # neighborTable = {}
    # tmp = {}
    # for i in range(3):
    #     tmp[i] = Neighbor(Addr('0.25'), 180, 92554, 233, color='red')
    # neighborTable = tmp

    # # p.setNeighbors(neighborTable)
    # print(len(neighborTable))
    # p.setNeighborsSize(len(neighborTable))
    # i=0
    # for key in neighborTable:
    #     p.setNeighborAddress(i, neighborTable[key].getAddr())
    #     p.setLinkQuality(i, neighborTable[key].getRssi())
    #     p.setLinkColor(i, neighborTable[key].getColorInt())
    #     i += 1
    # print(' '.join(format(i, '08b') for i in p.toByteArray())) 
    # # p.setNeighborsSize(1)
    # p.setNeighborAddress(0, Addr('0.3'))
    # p.setLinkQuality(0, 185)
    # print(getColorVal(p.getLinkColor(1)))
