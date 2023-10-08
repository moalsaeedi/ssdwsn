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

from ssdwsn.util.utils import getBitRange, getCompOperatorFromString, setBitRange, mergeBytes, getOperandFromString, getCompOperatorToString, getNetworkPacketByteName
from ssdwsn.util.constants import Constants as ct

class Window(object):
    """Matching Window"""
    def __init__(self, window:bytearray=None, size:int=None):
        """Initiate the matching part of a flow table entry

        Args:
            window (bytearray, optional): array of bytes contains the matching expression. Defaults to None.
            size (int, optional): size in bytes of a window. Defaults to None.
        """
        if not size:
            size = ct.W_SIZE
            
        if window:
            if len(window) == ct.W_SIZE:
                self.window = window
            else:
                self.window = bytearray(size)
        else:
            self.window = bytearray(size)
        
    @classmethod    
    def fromString(cls, val):
        frm = Window()
        operands = val.split(" ")
        if len(operands) == ct.W_LEN:            
            lhs = operands[0]
            tmpLhs = getOperandFromString(lhs)
            frm.setLhsOperandType(tmpLhs[0])
            frm.setLhs(tmpLhs[1])
            
            frm.setOperator(getCompOperatorFromString(operands[1]))

            rhs = operands[2]
            tmpRhs = getOperandFromString(rhs)
            frm.setRhsOperandType(tmpRhs[0])
            frm.setRhs(tmpRhs[1])

            if(lhs in ct.W_LHS_OPTIONS or rhs in ct.W_RHS_OPTIONS):
                frm.setSize(ct.W_SIZE_1)

        return frm

    def getLhsOperandType(self):
        return getBitRange(self.window[ct.W_OP_INDEX], ct.W_LEFT_BIT, ct.W_LEFT_LEN)
    
    def getLhs(self):
        return mergeBytes(self.window[ct.W_LEFT_INDEX_H], self.window[ct.W_LEFT_INDEX_L])    

    def getLhsToString(self):
        return {
            ct.CONST: self.getLhs(),
            ct.PACKET: "P.%s"% getNetworkPacketByteName(self.getLhs()),
            ct.STATUS: "R.%s"% self.getLhs()
        }.get(self.getLhsOperandType(), "")        

    def getRhsOperandType(self):
        return getBitRange(self.window[ct.W_OP_INDEX], ct.W_RIGHT_BIT, ct.W_RIGHT_LEN)
    
    def getRhs(self):
        return mergeBytes(self.window[ct.W_RIGHT_INDEX_H], self.window[ct.W_RIGHT_INDEX_L])    

    def getRhsToString(self):
        return {
            ct.CONST: self.getRhs(),
            ct.PACKET: "P.%s"% getNetworkPacketByteName(self.getRhs()),
            ct.STATUS: "R.%s"% self.getRhs()
        }.get(self.getRhsOperandType(), "")
        
    def getOperator(self):
        return getBitRange(self.window[ct.W_OP_INDEX], ct.W_OP_BIT, ct.W_OP_LEN)

    def setOperator(self, val):
        self.window[ct.W_OP_INDEX] = setBitRange(self.window[ct.W_OP_INDEX], ct.W_OP_BIT, ct.W_OP_LEN, val)
        return self

    def setLhsOperandType(self, val):
        self.window[ct.W_OP_INDEX] = setBitRange(self.window[ct.W_OP_INDEX], ct.W_LEFT_BIT, ct.W_LEFT_LEN, val)
        return self
       
    def setLhs(self, val):
        self.window[ct.W_LEFT_INDEX_H] = int(val) >> 8
        self.window[ct.W_LEFT_INDEX_L] = int(val) & 0xff
        return self
  
    def setLhsOperator(self, val):
        self.window[ct.W_OP_INDEX] = setBitRange(self.window[ct.W_OP_INDEX], ct.W_OP_BIT, ct.W_OP_LEN, val)
        return self
                 
    def setRhsOperandType(self, val):
        self.window[ct.W_OP_INDEX] = setBitRange(self.window[ct.W_OP_INDEX], ct.W_RIGHT_BIT, ct.W_RIGHT_LEN, val)        
        return self
       
    def setRhs(self, val):
        self.window[ct.W_RIGHT_INDEX_H] = int(val) >> 8
        self.window[ct.W_RIGHT_INDEX_L] = int(val) & 0xff
        return self
    
    def getSize(self):
        return getBitRange(self.window[ct.W_OP_INDEX], ct.W_SIZE_BIT, ct.W_SIZE_LEN)

    def setSize(self, val):
        self.window[ct.W_OP_INDEX] = setBitRange(self.window[ct.W_OP_INDEX], ct.W_SIZE_BIT, ct.W_SIZE_LEN, val)
        return self

    def getSizeToString(self):
        return ""
    
    def toByteArray(self):
        return self.window

    def hashCode(self):
        return hash(self.window)        
        
    def __str__(self):
        return '{} {} {}'.format(self.getLhsToString(), getCompOperatorToString(int(self.getOperator())), self.getRhsToString())