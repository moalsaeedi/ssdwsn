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
from enum import Enum
from logging import warn
from typing import Type
from ssdwsn.util.utils import getBitRange, setBitRange, mergeBytes, getOperandFromString, getMathOperatorFromString, getMathOperatorToString, getNetworkPacketByteFromName, getNetworkPacketByteName
from ssdwsn.util.constants import Constants as ct
from ssdwsn.data.addr import Addr
from ctypes import ArgumentError, c_uint32 as unsigned_int32

class Action(Enum):
    """Action type

    Args:
        Enum (NULL): Null action
        Enum (FORWARD_U): Unicast forwarding action
        Enum (FORWARD_B): Multicast forwarding action
        Enum (DROP): Drop a packet
        Enum (ASK): Request the controller to get the forwarding rule
        Enum (SET): Set new flow-rule
        Enum (MATCH): Match the arriving packet with the installed rules

    Returns:
        value: int value of the action
        name: name of the action
    """
    NULL = 0
    FORWARD_U = 1
    FORWARD_B = 2
    DROP = 3
    ASK = 4
    SET = 5
    MATCH = 6 
    
    def getValue(self):
        return self.value
    
    @classmethod
    def fromString(cls, str):
        return bytes(cls(str).value)
    
    @classmethod
    def fromByte(cls, value):
        return cls(value).name
    
    # def __str__(self):
    #     return 'Action type of {}'.format(self.name)

class AbstractAction(object):
    """Abstract Action Type"""
    def __init__(self, actionType:Action=None, size:int=None, action:bytearray=None):
        """Initialte an abstract action

        Args:
            actionType (Action, optional): enum action type. Defaults to None.
            size (int, optional): size of array bytes of actions. Defaults to None.
            action (bytearray, optional): is array of bytes started with the actionType and follows with action values (instructions). Defaults to None.
        """
        self.actionTypeToBytes = {}
        if size is None:
            size = 0
        if action:
            self.action = action
        elif actionType:
            self.action = bytearray(size+1)
            self.setType(actionType)
            
    def getType(self):
        return self.action[ct.AC_TYPE_INDEX]
    
    def getTypeName(self):
        return{
            0: 'NULL',
            1: 'FORWARD_U',            
            2: 'FORWARD_B',
            3: 'DROP',
            4: 'ASK',
            5: 'SET',
            6: 'MATCH'       
        }.get(self.action[ct.AC_TYPE_INDEX], "")
    
    def setType(self, actionType:Action):
        self.action[ct.AC_TYPE_INDEX] = actionType.value
        
    def getValue(self, index:int=None):
        if index is None:
            return self.action[ct.AC_VALUE_INDEX:len(self.action)]
        else:
            # try:
            return self.action[index + 1]
            # except: IndexError
    
    def setValue(self, index:int=None, actionValue=None, action:bytearray=None):
        if not(index is None and actionValue is None):
            # try:
            self.action[index + 1] = actionValue
            # except: IndexError
        elif action:
            _action = bytearray(len(action)+1) 
            _action.append(self.getType())
            for i in action:
                _action.append(i)
            self.action = _action
    
    @staticmethod
    def fromActionValueToSize(val:int):
        return {
            1: ct.FD_SIZE,
            2: ct.FD_SIZE,
            3: ct.DROP_SIZE,
            4: ct.ASK_SIZE,
            5: ct.ST_SIZE,
            6: ct.MATCH_SIZE            
        }.get(val, -1)
    
    @staticmethod
    def build(data):
        if isinstance(data, str):
            switcher = {
             "FORWARD_U": (lambda : ForwardUnicastAction(strValue=data)),
             "FORWARD_B": (lambda : ForwardBroadcastAction()),
             "DROP": (lambda : DropAction()),
             "ASK": (lambda : AskAction()),
             "SET": (lambda : SetAction(strValue=data)),
             "MATCH": (lambda : MatchAction())
            }
            return switcher.get(data.split(' ')[0], 'No action type found!')()
        if isinstance(data, bytearray):
            switcher = {
             1: (lambda : ForwardUnicastAction(action=data)),
             2: (lambda : ForwardBroadcastAction()),
             3: (lambda : DropAction()),
             4: (lambda : AskAction()),
             5: (lambda : SetAction(action=data)),
             6: (lambda : MatchAction())
            }
            return switcher.get(data[ct.AC_TYPE_INDEX], 'No action type found!')()
        
    def toByteArray(self):
        return self.action
         
    def hash(self):
        return hash(self.action)        

class SetAction(AbstractAction):      
    """Set action type"""
    def __init__(self, action:bytearray=None, strValue:str=None):
        """Initate a set action

        Args:
            action (bytearray, optional): array of bytes contains the action instructions. Defaults to None.
            strValue (str, optional): action value expression. Defaults to None.
        """
        super().__init__(actionType = Action.SET, size = ct.SET_SIZE, action = action)
        if strValue:
            tmpRes, tmpLhs, tmpRhs = [], [], []
            operands = strValue.split(' ')
            if len(operands) == ct.SET_FULL_SET:
                res = operands[ct.SET_RES]
                lhs = operands[ct.SET_LHS]
                rhs = operands[ct.SET_RHS]

                tmpRes = self.getResFromString(res)
                tmpLhs = getOperandFromString(lhs)
                tmpRhs = getOperandFromString(rhs)
                # Example: SET P.10 = R.11 + 12
                # Result (P.10) >> [2, 10]
                self.setResType(tmpRes[0])
                self.setRes(tmpRes[1])
                # Left-hand Operand (R.11) >> [3, 11]
                self.setLhsOperandType(tmpLhs[0])
                self.setLhs(tmpLhs[1])
                # Operator (+)
                self.setOperator(getMathOperatorFromString(operands[ct.SET_OP]))
                # Right-hand Operand (12) [12]
                self.setRhsOperandType(tmpRhs[0])
                self.setRhs(tmpRhs[1])
                
            elif len(operands) == ct.SET_HALF_SET:
                res = operands[ct.SET_RES]
                lhs = operands[ct.SET_LHS]

                tmpRes = self.getResFromString(res)
                self.tmpLhs = getOperandFromString(lhs)

                self.setResType(tmpRes[0])
                self.setRes(tmpRes[1])

                self.setLhsOperandType(tmpLhs[0])
                self.setLhs(tmpLhs[1])

                self.setRhsOperandType(None)
                self.setRhs(0)


    def getRes(self):
        return mergeBytes(self.getValue(ct.SET_RES_INDEX_H), self.getValue(ct.SET_RES_INDEX_L))
                
    def getResFromString(self, val):
        tmp = []
        strVal = val.split(".")
        switcherLh = {
            "P": ct.PACKET,
            "R": ct.STATUS
        }
        tmp.append(switcherLh.get(strVal[0], 'error'))
        if tmp[0] == ct.PACKET:
            tmp.append(getNetworkPacketByteFromName(strVal[1]))
        elif tmp[0] == ct.CONST:
            tmp.append(strVal[0])
        else: tmp.append(strVal[1])
        
        return tmp

    def getResToString(self):
        tmp = self.getResLocation()
        if tmp == ct.PACKET:
            return Action.SET.name + " P." + str(getNetworkPacketByteFromName(self.getRes())) + " = "
        elif tmp == ct.STATUS:
            return Action.SET.name + " R." + str(self.getRes()) + " = "
        else: return ""
            
    def getResLocation(self):
        return getBitRange(self.getValue(ct.SET_OP_INDEX), ct.SET_RES_BIT, ct.SET_RES_LEN) + 2

    def setResType(self, val):
        self.setValue(index=ct.SET_OP_INDEX, actionValue=setBitRange(self.getValue(ct.SET_OP_INDEX), ct.SET_RES_BIT, ct.SET_RES_LEN, val))

    def setRes(self, val):
        self.setValue(index=ct.SET_RES_INDEX_L, actionValue=int(val))
        self.setValue(index=ct.SET_RES_INDEX_H, actionValue=int(val) >> 8)
    
    def getLhs(self):
        return mergeBytes(self.getValue(ct.SET_LEFT_INDEX_H), self.getValue(ct.SET_LEFT_INDEX_L))

    def getLhsOperandType(self):
        return getBitRange(self.getValue(ct.SET_OP_INDEX), ct.SET_LEFT_BIT, ct.SET_LEFT_LEN)

    def getLhsToString(self):
        tmp = self.getLhsOperandType()
        if tmp == ct.NULL:
            return ""
        elif tmp == ct.CONST:
            return str(self.getLhs())
        elif tmp == ct.PACKET:
            return "P." + str(getNetworkPacketByteName(self.getLhs()))
        elif tmp == ct.STATUS:
            return "R." + str(self.getLhs())
        else: return ""
    
    def setLhsOperandType(self, val):
        self.setValue(index=ct.SET_OP_INDEX, actionValue=setBitRange(self.getValue(ct.SET_OP_INDEX), ct.SET_LEFT_BIT, ct.SET_LEFT_LEN, val))

    def setLhs(self, val):
        self.setValue(index=ct.SET_LEFT_INDEX_L, actionValue=int(val))
        self.setValue(index=ct.SET_LEFT_INDEX_H, actionValue=int(val) >> 8)
            
    def setRhsOperandType(self, val):
        self.setValue(index=ct.SET_OP_INDEX, actionValue=setBitRange(self.getValue(ct.SET_OP_INDEX), ct.SET_RIGHT_BIT, ct.SET_RIGHT_LEN, val))
    
    def getRhs(self):
        return mergeBytes(self.getValue(ct.SET_RIGHT_INDEX_H), self.getValue(ct.SET_RIGHT_INDEX_L))

    def getRhsOperandType(self):
        return getBitRange(self.getValue(ct.SET_OP_INDEX), ct.SET_RIGHT_BIT, ct.SET_RIGHT_LEN)

    def getRhsToString(self):
        tmp = self.getRhsOperandType()
        if tmp == ct.NULL:
            return ""
        elif tmp == ct.CONST:
            return str(self.getRhs())
        elif tmp == ct.PACKET:
            return "P." + str(getNetworkPacketByteName(self.getRhs()))
        elif tmp == ct.STATUS:
            return "R." + str(self.getRhs())
        else: return ""
        
    def setRhs(self, val):
        self.setValue(index=ct.SET_RIGHT_INDEX_L, actionValue=int(val))
        self.setValue(index=ct.SET_RIGHT_INDEX_H, actionValue=(int(val) >> 8))
     
    def getOperator(self):
        return getBitRange(self.getValue(ct.SET_OP_INDEX), ct.SET_OP_BIT, ct.SET_OP_LEN)
           
    def setOperator(self, val):
        self.setValue(index=ct.SET_OP_INDEX, actionValue=setBitRange(self.getValue(ct.SET_OP_INDEX), ct.SET_OP_BIT, ct.SET_OP_LEN, val))
            
    def __str__(self):
        f = self.getResToString()
        l = self.getLhsToString()
        r = self.getRhsToString()
        o = getMathOperatorToString(self.getOperator())

        if not (l is None) and not (r is None):
            return '{} {} {} {}'.format(f, l, o, r)
        elif (r is None):
            return '{} {}'.format(f, l)
        else:
            return '{} {}'.format(f, r)

class MatchAction(AbstractAction):           
    """Match action type"""
    def __init__(self, action:bytearray=None):
        """Initiate a match action

        Args:
            action (bytearray, optional): array of bytes contains the action instructions. Defaults to None.
        """
        super().__init__(actionType = Action.MATCH, size = ct.MATCH_SIZE, action = action)        
        
    def __str__(self):
        return Action.MATCH.name
        
class DropAction(AbstractAction):          
    """Drop action type"""
    def __init__(self, action:bytearray=None):
        """Initiate a drop action

        Args:
            action (bytearray, optional): array of bytes contains the action instructions. Defaults to None.
        """
        super().__init__(actionType = Action.DROP, size = ct.DROP_SIZE, action = action)   

    def __str__(self):
        return Action.DROP.name
        
class AskAction(AbstractAction):            
    """Ask action type"""
    def __init__(self, action:bytearray=None):
        """Initiate an ask action

        Args:
            action (bytearray, optional): array of bytes contains the action instructions. Defaults to None.
        """
        super().__init__(actionType = Action.ASK, size = ct.ASK_SIZE, action = action)   
        
    def __str__(self):
        return Action.ASK.name
    
class ForwardAction(AbstractAction):        
    """Forward action"""
    def getNextHop(self):
        return Addr("{}.{}".format(self.getValue(ct.FD_NXH_INDEX), self.getValue(ct.FD_NXH_INDEX + 1)))

    def setNextHop(self, addr:Addr):
        self.setValue(ct.FD_NXH_INDEX, addr.getHigh()) 
        self.setValue(ct.FD_NXH_INDEX + 1, addr.getLow())
            
class ForwardUnicastAction(ForwardAction):           
    """Forward-unicast action"""
    def __init__(self, action:bytearray=None, strValue:str=None, nxtHop:Addr=None):
        """Initiate a forward-unicast action

        Args:
            action (bytearray, optional): array of bytes contains the action instructions. Defaults to None.
            strValue (str, optional): action value expression. Defaults to None.
            nxtHop (Addr, optional): next forwarding unicast hop. Defaults to None.
        """
        super().__init__(actionType = Action.FORWARD_U, size = ct.FD_SIZE, action = action)   
        if strValue:
            tmp = strValue.split(' ')
            # Example: FORWARD_U 1.5
            if tmp[0] == Action.FORWARD_U.name:     
                self.setNextHop(Addr(tmp[1]))
        elif nxtHop:
            self.setNextHop(nxtHop)
            
    def __str__(self):
        return '{} {}'.format(Action.FORWARD_U.name, str(self.getNextHop()))
            
class ForwardBroadcastAction(ForwardAction):           
    """Forwarding-multicast action"""
    def __init__(self, action:bytearray=None):
        """Initiate a forwarding-multicast action

        Args:
            action (bytearray, optional): array of bytes contains the action instructions. Defaults to None.
        """
        super().__init__(actionType = Action.FORWARD_B, size = ct.FD_SIZE, action = action)   
        self.setNextHop(Addr(ct.BROADCAST_ADDR))
    
    def __str__(self):
        return Action.FORWARD_B.name
        
        