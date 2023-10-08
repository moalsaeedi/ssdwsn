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

from distutils.log import error
import re
from multiprocessing.sharedctypes import Value
from os import environ, system, getpid
from threading import Lock
from enum import Enum
from subprocess import Popen, PIPE, run, check_output
from ssdwsn.util.constants import Constants as ct
from sys import version_info as py_version_info
from ssdwsn.util.utils import ipAdd6, runCmd
if 'PYTHONPATH' in environ:
    path = environ[ 'PYTHONPATH' ].split(':') + path
from os.path import join as path_join
from sys import path, stdout, exc_info
lock = Lock()

class IntfType(Enum):
    """Interface Type

    Args:
        Enum (sixlowpan): 6LoWPAN Protocol Stack over IEEE802.15.4 (distance is 10 –100 m)
        Enum (lorawan): lowrawan protocol stack over IEEE802.15.4 (distance is 5 –50 km)

    Returns:
        value: int value of the interface type
        name: name of the interface type
    """
    sixlowpan = 0
    lorawan = 1
    
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
        if val == 0 or val == 'sixlowpan':
            params = {
            'antHeight': 1, #cm
            'antGain': 4, #dBm 
            'antRange': 100, #m A pixel equal to 1m  
            'txPower': 12, #dBm Ex. 36 dBm (4 watt)
            'freq': 2.48, #GHz
            'bandwidth': 250, #Kbps
            'channel': 26, # 2.48 GHz
            }
        if val == 1 or val == 'lorawan':
            params = {
            'antHeight': 1, #cm
            'antGain': 4, #dBm 
            'antRange': 100, #m A pixel equal to 1m  
            'txPower': 12, #dBm Ex. 36 dBm (4 watt)
            'freq': 2.48, #GHz
            'bandwidth': 50, #Kbps
            'channel': 26, # 2.48 GHz
            }
                           
        return params
    
    @staticmethod
    def fromStringToInt(str):
        switcher = {
            IntfType.sixlowpan.name: IntfType.sixlowpan.value,
            IntfType.lorawan.name: IntfType.lorawan.value
        }
        return switcher.get(str, -1)

class _Intf(object):
    """Abstract Interface object"""
    def __init__(self, nodeId, intfType:IntfType=None, ip:str=None, port:int=None, mac:str=None, **params):
        self.name = 'intf-'+nodeId
        self.type = intfType.getValue()        
        self.seq = port - ct.BASE_NODE_PORT
        self.nodeId = nodeId
        self.ip = ip
        self.port = port
        self.mac = mac
        self.params = {}
        self.params.update(params)
        
    def config(self, pid):
        pass     
    
    def getMac(self):
        return self.mac   
    
    def getAntHeight(self):
        return self.params['antHeight']
    
    def setAntHeight(self, val:int):
        self.params['antHeight'] = val
        
    def getAntGain(self):
        return self.params['antGain']
    
    def setAntGain(self, val:int):
        self.params['antGain'] = val
        
    def getAntRange(self):
        return self.params['antRange']

    def setAntRange(self, val:int):
        self.params['antRange'] = val

    def getTxPower(self):
        return self.params['txPower']

    def setTxPower(self, val:int):
        self.params['txPower'] = val  
        
    @classmethod
    def getIP6FromSeq(cls, seq=None, port=None):
        """
        To generate the ip6 from the node sequance
        * seq: the port - base_port -> the seq # of the node
        """
        seq = seq + 5
        if not(port is None):
            seq = (port - ct.BASE_NODE_PORT) + 5
        return ipAdd6(seq, prefixLen=64, ipBaseNum=ct.BASE_IP6) +'/%s' % 64       

    def runCmd(self, cmd):
        myenv = environ.copy()
        if 'VIRTUAL_ENV' in environ:
            myenv['PATH'] = ':'.join(
                [x for x in environ['PATH'].split(':')
                    if x != path_join(environ['VIRTUAL_ENV'], 'bin')])
        return Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, env=myenv)

class _WirelessIntf(_Intf):
    """Wireless Infterface object"""
    def __init__(self, nodeId:str, intfType:IntfType=None, ip:str=None, port:int=None, mac:str=None, **params):
        """Initiate a wireless interface

        Args:
            nodeId (_type_): node id
            intfType (IntfType, optional): interface type (6lowpan, lorawan, etc.). Defaults to None.
            ip (str, optional): interface ip address. Defaults to None.
            port (int, optional): interface port. Defaults to None.
            mac (str, optional): interface mac address. Defaults to None.
        """
        super().__init__(nodeId, intfType, ip, port, mac, **params)   
        self.name = 'wintf-'+nodeId
        self.seq = port - ct.BASE_NODE_PORT

    def getAntHeight(self):
        return self.params['antHeight']
    
    def getAntGain(self):
        return self.params['antGain']
    
    def getAntRange(self):
        return self.params['antRange']
    
    def getTxPower(self):
        return self.params['txPower']
    
    def getFreq(self):
        return self.params['freq']
            
    def get_intfs(self, cmd):
        'Gets the list of virtual wifs that already exist'
        if py_version_info < (3, 0):
            intfs = check_output\
                (cmd, shell=True).split("\n")
        else:
            intfs = check_output\
                (cmd, shell=True).decode('utf-8').split("\n")
        intfs.pop()
        ints_list = sorted(intfs)
        ints_list.sort(key=len, reverse=False)
        return ints_list     

class SixLowPan(_WirelessIntf):
    """LoPAN interface mac802154_hwsim (6lowpan)"""
    def __init__(self, nodeId:str, ip:str=None, port:int=None, mac:str=None, **params):
        """Initiate a 6lowpan interface
        The throughput under this standard is limited to 250 kbps, 
        and the frame length is limited to 127 bytes to ensure low packet and bit error rates in a lossy RF environment. 
        
        Args:
            nodeId (_type_): node id
            ip (str, optional): interface ip address. Defaults to None.
            port (int, optional): interface port. Defaults to None.
            mac (str, optional): interface mac address. Defaults to None.
        """
        super().__init__(nodeId, IntfType.sixlowpan, ip, port, mac, **params)        
        self.name = '6lowpan-'+nodeId
        self.params.update(IntfType.getParams(self.type))
        self.shortIP = "0x%0.4X" % self.seq
        # ip6
        self.ip6 = self.getIP6FromSeq(self.seq)
        
        # self.config()
        
    def config(self, pid=None):   
        panid=0xbeef
        '''
        # system(f'modprobe fakelb numlbs={1}')
        system('modprobe mac802154_hwsim')
        # system(f'ip netns delete {pid}')
        system('wpan-hwsim add >/dev/null 2>&1')
        # system(f'ip netns delete wpan{self.seq}')
        system(f'ip netns add wpan{self.seq}')
        phy = self.get_intfs(f"iwpan dev | grep -B 1 wpan{self.seq} | sed -ne 's/phy#\([0-9]\)/\1/p'")
        print(phy)
        system(f'ip netns exec iwpan phy phy{phy[0]} set netns name wpan{self.seq}')
        system(f'ip netns exec wpan{self.seq} iwpan dev wpan{self.seq} set pan_id {panid}')
        system(f'ip netns exec wpan{self.seq} ip link add link wpan{self.seq} name {self.name} type lowpan')
        system(f'ip netns exec wpan{self.seq} ip link set wpan{self.seq} up')
        system(f'ip netns exec wpan{self.seq} ip link set {self.name} up')
        '''
        
        ps = []
        # p = runCmd("modprobe mac802154_hwsim")
        # ps.append(p)
        # p = runCmd("wpan-hwsim add >/dev/null 2>&1")
        # ps.append(p)

        phy = check_output(f"iwpan dev | grep -B 1 wpan{self.seq} | sed -ne 's/phy\([^ ]\)/\1/p'", shell=True, text=True)
        phy = re.findall(r'[0-9]+', phy)
        # print(phy[0])
        # p = runCmd(f"ip link set wpan{self.seq} down; iwpan dev wpan{self.seq} set pan_id {panid}; iwpan dev wpan{self.seq} set short_addr {self.shortIP}; iwpan phy{phy[0]} set netns {pid}; iwpan phy{phy[0]} set channel 0 {int(self.params['channel'])}; ip link add link wpan{self.seq} name {self.name} type lowpan; ip link set wpan{self.seq} up; ip link set {self.name} up; ip addr add {self.ip6} dev {self.name}")
        # p.terminate()

        #lowpan interface with a 6LoWPAN 1280 MTU

        # check_output(f"ip link set wpan{self.seq} down; \
        #     iwpan dev wpan{self.seq} set pan_id {panid}; \
        #     iwpan dev wpan{self.seq} set short_addr {self.shortIP}; \
        #     iwpan phy{phy[0]} set netns {pid}; \
        #     iwpan phy{phy[0]} set channel 0 {int(self.params['channel'])}; \
        #     ip link add link wpan{self.seq} name {self.name} type lowpan; \
        #     ip link set wpan{self.seq} mtu 1500; \
        #     ip link set wpan{self.seq} up; \
        #     ip link set {self.name} up; \
        #     ip addr add {self.ip6} dev {self.name}",
        # shell=True, text=True)

        check_output(f"ip link set wpan{self.seq} down; \
            iwpan phy{phy[0]} set netns {pid}; \
            iwpan dev wpan{self.seq} set pan_id {panid}; \
            iwpan phy{phy[0]} set channel 0 {int(self.params['channel'])}; \
            ip link add link wpan{self.seq} name {self.name} type lowpan; \
            ip link set wpan{self.seq} up; \
            ip link set {self.name} up; \
            ip addr add {self.ip6} dev {self.name}",
        shell=True, text=True)


        # check_output(f"ip link set wpan{self.seq} down; \
        #     ip netns add wpan{self.seq}; \
        #     iwpan phy{phy[0]} set netns name wpan{self.seq}; \
        #     ip netns exec wpan{self.seq} iwpan dev wpan{self.seq} set pan_id {panid}; \
        #     ip netns exec wpan{self.seq} iwpan phy{phy[0]} set channel 0 {int(self.params['channel'])}; \
        #     ip netns exec wpan{self.seq} ip link add link wpan{self.seq} name {self.name} type lowpan; \
        #     ip netns exec wpan{self.seq} ip link set wpan{self.seq} up; \
        #     ip netns exec wpan{self.seq} ip link set {self.name} up; \
        #     ip netns exec wpan{self.seq} ip addr add {self.ip6} dev {self.name}",
        # shell=True, text=True)

        # ip link set wpan{self.seq} mtu 1500

        # p = runCmd(f"ip link set wpan{self.seq} down")
        # ps.append(p)
        # p = runCmd(f"iwpan dev wpan{self.seq} set pan_id {panid}")
        # ps.append(p)
        # p = runCmd(f"iwpan dev wpan{self.seq} set short_addr {self.shortIP}")
        # ps.append(p)
        # # p = runCmd(f"ip netns add wpan{self.seq}")
        # # ps.append(p)
        # p = runCmd(f"iwpan phy{phy[0]} set netns {pid}") #or pid instead 
        # ps.append(p)
        # p = runCmd(f"iwpan phy{phy[0]} set channel 0 {int(self.params['channel'])}") #channel-26 	2480
        # ps.append(p)
        # p = runCmd(f"ip link add link wpan{self.seq} name {self.name} type lowpan")
        # ps.append(p)
        # # p = runCmd(f"ip link set wpan{self.seq} mtu 1500")
        # # ps.append(p)
        # p = runCmd(f"ip link set wpan{self.seq} up")
        # ps.append(p)
        # p = runCmd(f"ip link set {self.name} up")
        # ps.append(p)
        # p = runCmd(f"ip addr add {self.ip6} dev {self.name}")
        # ps.append(p)
        # for p in ps:
        #     p.terminate()
        
        
        '''
        # system("modprobe mac802154_hwsim")        
        system("wpan-hwsim add >/dev/null 2>&1")        
        phy = check_output(f"iwpan dev | grep -B 1 wpan{self.seq} | sed -ne 's/phy\([^ ]\)/\1/p'", shell=True, text=True)
        phy = re.findall(r'[0-9]+', phy)
        # system(f"ip link set wpan{self.seq} down")        
        # system(f"ip netns add wpan{self.seq}")        
        # system(f"iwpan phy{phy[0]} set netns name wpan{self.seq}") #or pid instead  
        system(f"iwpan phy{phy[0]} set netns name ssdwsn") #or pid instead  
        system(f"iwpan dev wpan{self.seq} set pan_id {panid}")        
        # system(f"iwpan dev wpan{self.seq} set short_addr {self.shortIP}") 
        # system(f"iwpan phy{phy[0]} set channel 0 {int(self.params['channel'])}") #channel-26 	2480        
        system(f"ip link add link wpan{self.seq} name {self.name} type lowpan")        
        # # system(f"ip link set wpan{self.seq} mtu 1500")
        system(f"ip link set wpan{self.seq} up")        
        system(f"ip link set {self.name} up")        
        # system(f"ip addr add {self.ip6} dev {self.name}")
        '''

        '''
        system("modprobe mac802154_hwsim")
        system("wpan-hwsim add >/dev/null 2>&1")
        # phy = check_output(f"iwpan dev | grep -B 1 wpan{self.seq} | sed -ne 's/phy#\([0-9]\)/\1/p'", shell=True, text=True)
        phy = check_output(f"iwpan dev | grep -B 1 wpan{self.seq} | sed -ne 's/phy\([^ ]\)/\1/p'", shell=True, text=True)
        phy = re.findall(r'[0-9]+', phy)
        print(phy[0])
        wpan_list = self.get_intfs("iwpan dev 2>&1 | grep Interface | awk '{print $2}'")
        # system(f"ip netns delete wpan{self.seq}")
        # system(f"ip netns add wpan{self.seq}")
        system(f"ip link set wpan{self.seq} down")
        system(f"iwpan dev wpan{self.seq} set pan_id {panid}")
        system(f"iwpan dev wpan{self.seq} set short_addr {self.shortIP}")
        # system(f"iwpan phy phy{int(phy[0])} set netns name wpan{self.seq}") #or pid instead 
        system(f"iwpan phy{phy[0]} set netns {pid}") #or pid instead 
        # system(f"iwpan phy phy{phy[0]} set tx_power {int(self.params['txPower'])}")
        system(f"iwpan phy{phy[0]} set channel 0 {int(self.params['channel'])}") #channel-26 	2480
        # system(f'iwpan phy phy{phy[0]} interface add monitor{self.seq} type monitor')
        # system('ip link set wpan{} name wpan-{}'.format(self.seq, self.nodeId))
        # system('link set lo up')
        # system('ip link set {} down' .format('wpan-'+self.nodeId))
        system(f"ip link add link wpan{self.seq} name {self.name} type lowpan")
        # system(f'ip link set wpan{self.seq} mtu 1500')   
        # system('ip -6 addr add {} dev wpan-{}'.format(self.ip, self.nodeId))
        # system('ip -6 addr add {} dev {}'.format(self.ip, self.name))
        # system('ifconfig {} inet6 add {}'.format(self.name, self.ip))
        system(f"ip link set wpan{self.seq} up")
        # system(f'ip link set monitor{self.seq} up')
        system(f"ip link set {self.name} up")
        system(f"ip addr add {self.ip6} dev {self.name}")
        # ipp= check_output(f"ip addr show dev {self.name} | sed -e 's/^.*inet6 \([^ ]*\)\/.*$/\1/;t;d'", shell=True, text=True)
        # ipp= check_output(f"ip addr show dev {self.name} | grep -B 1 inet6 | sed -ne 's/inet6\([^ ]*\)\/.*$/\1/p'", shell=True, text=True).strip()
        # ipp= check_output(f"ip addr show dev {self.name} | grep -B 1 inet6 | sed -ne 's/inet6\([^ ]*\)/\1/p'", shell=True, text=True).strip().split(" ")
        # self.ip6 = ipp[1]
        # system('ip -6 route add default gw {} {}'.format(self.ip.split('/')[0], self.name))
        ##
        '''    
# https://wiki.polaire.nl/doku.php?id=atusb_6lowpan_802.15.4_fedora
# https://wiki.polaire.nl/doku.php?id=raspberry_pi_openlabs_6lowpan
# https://jan.newmarch.name/IoT/LinuxJournal/6LoWPAN/
        """
        system('echo 1 > /proc/sys/net/ipv4/ip_forward')
        system('iptables -t nat -A PREROUTING -s {} -p udp --sport {} -i {}'.format(self.ip, self.port, self.id+'-lowpan'))
        # system('iptables -t nat -A OUTPUT -s {} -p udp --dport {} -i {}'.format(self.ip, self.port, self.id+'-lowpan'))
        system('iptables -t nat -A POSTROUTING -o {} -p udp --dport {} -d {}'.format(self.id+'-lowpan', self.port, self.ip))
        # system('iptables -t nat -A INPUT -o {} -d {} --dport {}'.format(self.id+'-lowpan', self.ip, self.port))
        # system('ip -6 route del default')
        # system('ip -6 route add default dev {}'.format(self.id+'lowpan'))
        # system('ip netns exec {} ifconfig {} up {} netmask {}'.format(self.pid, 'wpan-'+self.id, self.ip6, self.ip6Mask))
        # system('ip netns exec {} route add default gw {} dev {}'.format(self.pid, self.ip6, 'wpan-'+self.id))
        """
        
class LoRaWan(_WirelessIntf):
    """LoWAN interface mac802154 (lorawan)"""
    def __init__(self, nodeId:str, ip:str=None, port:int=None, mac:str=None, **params):
        """Initate a lorawan interface

        Args:
            nodeId (_type_): node id.
            ip (str, optional): interface ip address. Defaults to None.
            port (int, optional): interface port. Defaults to None.
            mac (str, optional): interface mac address. Defaults to None.
        """
        super().__init__(nodeId, ip, port, mac, **params)

