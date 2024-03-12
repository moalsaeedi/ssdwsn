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

from argparse import Namespace
from distutils.log import info
from fcntl import DN_DELETE
from multiprocessing import Process, parent_process, set_start_method
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from subprocess import Popen, PIPE, run, check_output
from os import environ, kill, getpid, path, cpu_count
import traceback
import time
import re
from random import randint
import aiofiles
from numpy import append
from ssdwsn.data.battery import Dischargable, Chargable
import socket
import netifaces as ni
import json
import asyncio, logging
from ssdwsn.data.addr import Addr
from ssdwsn.data.neighbor import Neighbor
from ssdwsn.openflow.packet import Packet
from ssdwsn.util.constants import Constants as ct
from ssdwsn.openflow.action import Action, ForwardUnicastAction
from ssdwsn.util.utils import mergeBytes, mapRSSI, compare, Suspendable, CustomFormatter, ipAdd6, runCmd, portToAddrStr, zero_division, getColorVal
from ssdwsn.openflow.window import Window
from ssdwsn.openflow.entry import Entry
from ssdwsn.data.addr import Addr
from ssdwsn.data.neighbor import Neighbor
from ssdwsn.openflow.packet import BeaconPacket, DataPacket, OpenPathPacket, Packet, ReportPacket, RequestPacket, ConfigPacket, ConfigProperty, ResponsePacket, RegProxyPacket, AggrPacket
from ssdwsn.data.sensor import SensorType
from ssdwsn.data.intf import IntfType
import zmq
from zmq.asyncio import Context, Poller, ZMQEventLoop
from math import log, sqrt
import pandas as pd
from hashlib import md5 
import janus

#logging----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
#-----------------------------------

class Node:
    """Abstract Node Process"""
    def __init__(self, net:int, addr:Addr, portSeq:int, battery=None, pos:tuple=None, topo=None, ctrl=None, cls=None):
        """Create a node process

        Args:
            net (int): subnet id
            addr (Addr): node address
            portSeq (int): port sequance id (incremental) port = (start port addr) + portSeq
            battery (Dischargable/Chargable, optional): battery object. Defaults to None.
            pos (tuple, optional): node position on the grid (x,y,z). Defaults to None.
            cls (SensorType(Enum), optional): sensor type attached to the node. Defaults to None.
        """
        self.topofilename = topo if topo else 'topo'
        self.ctrl = ctrl
        self.seq = portSeq
        #Battery
        self.battery = None
        if not(battery is None): 
                self.battery = battery
        else: self.battery = Dischargable()
        #Requests count.
        self.requestId = 0
        #Accepted IDs
        self.acceptedId = []
        #Flow Table is a dict of Entry
        self.flowTable = FlowTable(self)
        #Node Position
        self.position = pos
        #A Mote becomes active after it receives a beacon message. A Sink is always active.        
        self.isActive = False
        #Sink node flag
        self.isSink = False
        #Is Aggr node
        self.isAggr = False
        #The address of the node typeof(Addr) Ex. 1.0.
        self.myAddress = addr
        #Sub net id 0 to 255.
        self.myNet = net
        #dictionary of neigbors {key=addr:value=Neigbor} addr format 000.000
        self.neighborTable = {}
        self.prevLinkToSink = {}
        self.cachedNeigs = []
        #Sensor Type
        self.sensorType = cls(self) if cls is not None else None 
        #Status Registers
        self.statusRegister = []
        #Node's assigned ip4
        self.ip = '127.0.0.1'
        #Stats of receiving/sending packets
        self.pos = {'value': pos, 'value_ts': pos, 'ts': round(time.time(), 4), 'rate': 0}
        self.sentBeacons = {'value': 0, 'value_ts': 0, 'ts': round(time.time(), 4), 'rate': 0}
        self.receivedBytes = {'value': 0, 'value_ts': 0, 'ts': round(time.time(), 4), 'rate': 0}
        self.sentBytes = {'value': 0, 'value_ts': 0, 'ts': round(time.time(), 4), 'rate': 0}
        self.receivedDataBytes = {'value': 0, 'value_ts': 0, 'ts': round(time.time(), 4), 'rate': 0}
        self.sentDataBytes = {'value': 0, 'value_ts': 0, 'ts': round(time.time(), 4), 'rate': 0}
        self.receivedPackets = {'value': 0, 'value_ts': 0, 'ts': round(time.time(), 4), 'rate': 0}
        self.sentPackets = {'value': 0, 'value_ts': 0, 'ts': round(time.time(), 4), 'rate': 0}
        self.receivedBytesOut = {'value': 0, 'value_ts': 0, 'ts': round(time.time(), 4), 'rate': 0}
        self.receivedPacketsOut = {'value': 0, 'value_ts': 0, 'ts': round(time.time(), 4), 'rate': 0}
        self.sentBytesIn = {'value': 0, 'value_ts': 0, 'ts': round(time.time(), 4), 'rate': 0}
        self.sentPacketsIn = {'value': 0, 'value_ts': 0, 'ts': round(time.time(), 4), 'rate': 0}
        self.dropedPackets = {'value': 0, 'value_ts': 0, 'ts': round(time.time(), 4), 'rate': 0}
        self.batt = {'value': self.battery.getLevel(), 'value_ts': self.battery.getLevel(), 'ts': round(time.time(), 4), 'rate': 0}
        self.rptti = {'value': ct.RP_TTI, 'value_ts': ct.RP_TTI, 'ts': round(time.time(), 4), 'rate': 0}
        self.betti = {'value': ct.BE_TTI, 'value_ts': ct.BE_TTI, 'ts': round(time.time(), 4), 'rate': 0}
        #statistics
        self.stats = bytearray()
        #Sync last updated time of network topology
        self.lastUpdatedGraphTimeStamp = 0
        #cache
        self.cachedReports = []
        self.cachedData = []
        #Interfaces
        self.intfs = []
        self.wintf = None
        self.ppm = None
        self.workers = []
        self.rxRadioSocket = None
        self.trxRadioSocket = None
        #ZeroMQ Socketio client
        self.zmqContext = None
        #Event loop
        self.loop = None
        #Rready node Flag
        self.ready = True
        #Time transmission intervals
        self.ts_dict = {}

    async def terminatef(self):
        logger.warn(f'node {self.id} is terminated .....................')
        try:
            # self.rxRadioSocket.close()          
            # self.trxRadioSocket.close()          
            self.sock.close()
        except:
            pass
        # for task in asyncio.all_tasks():
        #     try:
        #         task.cancel()  
        #     except:
        #         pass
        # self.loop.stop()

    async def run(self):
        """Start running the node (Mote/Sink) process"""        
        try:
            logger.info(f'[{self.id}] is running ...')  
            # setup the running loop
            self.loop = asyncio.get_event_loop()
            # self.zmqContext = Context.instance()
            # interface configuration
            self.shell = True   
            self.wintf.config(getpid())

            await self.connect()
        except Exception as e:
            logger.error(traceback.format_exc())
            self.ready = False
        finally:
            # await self.terminatef()
            pass

    async def connect(self):  
        """Configure sockets to (Controller/beers/visualizer) and run asynchronous workers
        following order of execution is required
        """             
        # WIRELESS RADIO (Transceiver): create a server udp endpoint to communicate with other wireless nodes
        # 1) configure radio transceiver---------------------------------------------------------------------
        self.scope_id = socket.if_nametoindex(self.wintf.name)
        self.rxRadioSocket = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM, 0)
        self.rxRadioSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # self.rxRadioSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        # self.rxRadioSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, str(self.wintf.name + '\0').encode('utf-8'))
        self.rxRadioSocket.setblocking(False)
        # print(self.wintf.ip6.split('/')[0])
        ((*_, sockaddr),) = socket.getaddrinfo(
            # self.wintf.ip6.split('/')[0]+'%'+str(self.scope_id),
            # self.wintf.ip6.split('/')[0]+'%'+str(self.scope_id),
            '::'+'%'+str(self.scope_id),
            # '::',
            self.wintf.port,
            socket.AF_INET6,
            socket.SOCK_DGRAM,
            0
        )
        
        await asyncio.sleep(1)
        # 2) Initialize OF protocol configuration and per-node specific parameters----------------------------
        await self.initssdwsn()
        await self.initNeighborTable()
        await asyncio.sleep(20)
        # 3) Run async workers--------------------------------------------------------------------------------
        # ASYNCHRONOUS WORKERS: Create asynchronous workers to handle network flow
        tasks = []
        pool = ThreadPoolExecutor(cpu_count())
        ppool = ProcessPoolExecutor(cpu_count())
        if self.isSink:
            tasks.append(self.loop.run_in_executor(pool, self.connCtrlWorker))
            tasks.append(self.loop.run_in_executor(pool, self.packetOutWorker))
        else:
            tasks.append(self.loop.run_in_executor(pool, self.sensingQueueWorker))
        tasks.append(self.loop.run_in_executor(pool, self.socketWorker))
        tasks.append(self.loop.run_in_executor(pool, self.checkBatteryWorker))
        # tasks.append(self.loop.run_in_executor(pool, self.updateNGWorker))
        tasks.append(self.loop.run_in_executor(pool, self.statusWorker))
        tasks.append(self.loop.run_in_executor(pool, self.reportWorker))
        tasks.append(self.loop.run_in_executor(pool, self.beaconWorker))

        await asyncio.gather(*tasks)

    async def txpacket(self, packet):
        """puts ready to be sent packets in a transmittion queue
        and decrements the TTL of a packet by one.

        Args:
            packet (_type_): different typs of packets (DATA, OPEN_PATH, REQUEST, RESPONSE, REPORT, BEACON)
        """
        packet.decrementTtl()
        asyncio.get_running_loop().run_in_executor(None, self.txHandler, packet)
        
    def txHandler(self, packet):
        """transmit the packet over the medium

        Args:
            packet (_type_): different typs of packets (DATA, OPEN_PATH, REQUEST, RESPONSE, REPORT, BEACON)
        """
        async def async_thread():
            if packet.isssdwsnPacket():
                try:
                    tmpNxtHop = packet.getNxh()
                    tmpDst = packet.getDst()
                    tmpId = f'{packet.getNet()}.{tmpNxtHop}'
                    srcid = f'{packet.getNet()}.{packet.getSrc()}'
                    if tmpDst.isBroadcast() or tmpNxtHop.isBroadcast():
                        for key, val in self.neighborTable.items():
                            packet.setPrh(self.myAddress)
                            lip6 = check_output(f"ip addr show dev 6lowpan-{str(key)} | grep -B 1 inet6 | sed -ne 's/inet6\([^ ]*\)/\1/p'", shell=True, text=True).strip().split(" ")
                            ((*_, sockaddr),) = socket.getaddrinfo(
                                # self.getIP6FromSeq(self.neighborTable[key].getPort()).split('/')[0]+'%'+str(socket.if_nametoindex('6lowpan-'+str(key))),
                                # 'ff02::1%'+str(socket.if_nametoindex('6lowpan-'+str(key))),
                                ni.ifaddresses('6lowpan-'+str(key))[socket.AF_INET6][0]['addr'],
                                # 'ff02::1',
                                # '::1%'+str(socket.if_nametoindex('6lowpan-'+str(key))),
                                # lip6[1].split('/')[0]+'%'+str(socket.if_nametoindex('6lowpan-'+str(key))),
                                val.getPort(),
                                socket.AF_INET6,
                                socket.SOCK_DGRAM,
                                0, socket.AI_PASSIVE
                            )
                            if srcid == self.id:
                                packet.setTS(round(time.time(), 4))

                            #Latency = Propagation Time + Transmission Time + Queuing Time + Processing Time
                            # Queuing Time is attached to the wireless interface
                            # Queuing Time + Transmission Time (simulation process)
                            # Iqnore processing Time
                            src_ts = packet.getTS()
                            if self.ts_dict.get(src_ts):
                                packet.setTS(round(src_ts + (round(time.time(), 4) - self.ts_dict[src_ts]), 4))
                                self.ts_dict.pop(src_ts)

                            time.sleep(val.getDist()/1e+8) # propagation Time = distance/speed_of_light
                            time.sleep((len(packet.toByteArray())*0.001*8)/(self.wintf.params['bandwidth'])) # Transmission Time = packet_len/network_bandwidth (250kbit/s)
                            ts1 = time.time()
                            self.sock.sendto(packet.toByteArray(), sockaddr)
                            ts2 = time.time()
                            self.battery.transmitRadio(ts2-ts1)
                            await self.updateStats('tx', packet)
                            logger.info(f'{self.id}-----> Sends Packet type ({packet.getTypeName()}) | {packet.getSrc()} --> {packet.getDst()} - Next Hop({packet.getNxh()})')
                    elif tmpId in self.neighborTable:
                        packet.setPrh(self.myAddress)
                        lip6 = check_output(f"ip addr show dev 6lowpan-{str(tmpId)} | grep -B 1 inet6 | sed -ne 's/inet6\([^ ]*\)/\1/p'", shell=True, text=True).strip().split(" ")
                        ((*_, sockaddr),) = socket.getaddrinfo(
                            # self.getIP6FromSeq(self.neighborTable[tmpId].getPort()).split('/')[0]+'%'+str(socket.if_nametoindex('6lowpan-'+str(tmpId))),
                            # 'ff02::1%'+str(socket.if_nametoindex('6lowpan-'+str(tmpId))),
                            ni.ifaddresses('6lowpan-'+str(tmpId))[socket.AF_INET6][0]['addr'],
                            # 'ff02::1',
                            # '::1%'+str(socket.if_nametoindex('6lowpan-'+str(tmpId))),
                            # lip6[1].split('/')[0]+'%'+str(socket.if_nametoindex('6lowpan-'+str(tmpId))),
                            self.neighborTable[tmpId].getPort(),
                            socket.AF_INET6,
                            socket.SOCK_DGRAM,
                            0, socket.AI_PASSIVE
                        )
                        # if src node set the sending timestamp
                        if srcid == self.id:
                            packet.setTS(round(time.time(), 4))

                        #Latency = Propagation Time + Transmission Time + Queuing Time + Processing Time
                        # Queuing Time is attached to the wireless interface
                        # Queuing Time + Transmission Time (simulation process)
                        # Iqnore processing Time
                        src_ts = packet.getTS()
                        if self.ts_dict.get(src_ts):
                            packet.setTS(round(src_ts + (round(time.time(), 4) - self.ts_dict[src_ts]), 4))
                            self.ts_dict.pop(src_ts)

                        time.sleep(self.neighborTable[tmpId].getDist()/1e+8) # propagation Time = distance/speed_of_light
                        time.sleep((len(packet.toByteArray())*0.001*8)/(self.wintf.params['bandwidth'])) # Transmission Time = packet_len/network_bandwidth (250kbit/s)
                        ts1 = time.time()
                        self.sock.sendto(packet.toByteArray(), sockaddr)
                        ts2 = time.time()
                        self.battery.transmitRadio(ts2-ts1) 
                        await self.updateStats('tx', packet)
                        logger.info(f'{self.id}-----> Sends Packet type ({packet.getTypeName()}) | {packet.getSrc()} --> {packet.getDst()} - Next Hop({packet.getNxh()})')
                except Exception as e:
                    logger.warn(e)
        asyncio.run(async_thread())
    
    async def rxPacket(self, packet, recv_ts, rx='rx'):
        """receives packets from the neigbors nodes or the controller (if node is sink)

        Args:
            packet (_type_): different typs of packets (DATA, OPEN_PATH, REQUEST, RESPONSE, REPORT, BEACON)
            recv_ts (_type_): packet's receiving timestamp

        Performance metrics calcualtion:--------------------
        Delay: The end-to-end delay refers to the time interval between the packet sent from a node till it is received at the base station. (sink).        
            this delay includes (Propagation Time + Transmission Time + Queuing Time + Aggregate delay + Processing Time)
        Throughput: The throughput is measured by the number of bits transmitted per unit time to the base station (sink). 
            cite: Singh, Samayveer. (2016). Energy efficient multilevel network model for heterogeneous WSNs. 
        Engineering Science and Technology, an International Journal. 20. 10.1016/j.jestch.2016.09.008. 
        ----------------------------------------------------
        """
        rssi = ct.RSSI_MAX #default RSSI
        dist = self.wintf.params['antRange'] #default Range
        if packet.isssdwsnPacket():
            srcid = f'{packet.getNet()}.{packet.getSrc()}'
            dstid = f'{packet.getNet()}.{packet.getDst()}'
            if self.neighborTable.get(srcid):
                dist = self.neighborTable[srcid].getDist()
                rssi = self.neighborTable[srcid].getRssi()
            ptype = packet.getType()
            send_ts = packet.getTS()
            delay = (recv_ts - send_ts)*1000
            if delay >= ct.MAX_DELAY:
                logger.info(f'DROPE IN {self.id} PACKET type ({packet.getTypeName()}) | {packet.getSrc()} --> {packet.getDst()} and Prev hope is {packet.getPrh()} DELAY: {delay} (ms)')
                await self.updateStats('drop', packet)
                return
            
            try:
                self.ts_dict[send_ts] = recv_ts
                await self.rxHandler((packet, dist, rssi))

                if dstid == self.id and self.isSink and ptype == ct.DATA:
                    self.ts_dict.pop(send_ts)
                    data = {"id":srcid,
                            "delay":delay,
                            "throughput":(len(packet.toByteArray())*0.001*8)/((recv_ts - send_ts))
                        }
                    self.stats.extend(b'ST'+b'perf'+str(json.dumps(data)).encode('utf-8')+b';')
                await self.updateStats(rx, packet)
                logger.info(f'{self.id}<----- Receives Packet type ({packet.getTypeName()}) | {packet.getSrc()} --> {packet.getDst()} and Prev hope is {packet.getPrh()})')
            except Exception as ex:
                logger.error(ex)
                logger.info(f'DROPE IN {self.id} PACKET TYPE ({packet.getTypeName()}) | {packet.getSrc()} --> {packet.getDst()} - Next Hop({packet.getNxh()})')
                await self.updateStats('drop', packet)

    def checkBatteryWorker(self):
        """check battery charging every one second.
        - decrement the battery level (6.8 mC every 1sec).
        - check if battery is empty, node process will be terminated.
        """
        async def async_thread():
            ts = round(time.time(), 4)
            while self.ready:
                ts1 = time.time()
                if (lambda: self.battery.getLevel())() == 0:
                    logger.warn(f'{self.id} BATTERY is empty...')
                    self.stats.extend(b'ST'+b'NODE'+str(json.dumps({'id':self.id, 'color':'black'})).encode('utf-8')+b';')
                    self.isActive = False
                    break           
                await asyncio.sleep(2)
                self.battery.keepAlive(time.time() - ts1)
                self.remvExpiredEnties(round(time.time(), 4) - ts)
                ts = round(time.time(), 4)
                await self.updateStats('batt')
        asyncio.run(async_thread())

    def socketWorker(self):
        async def async_thread():
            transport, protocol = await asyncio.get_running_loop().create_datagram_endpoint(
                lambda: _SocketProtocol(self.isSink),
                ('::'+'%'+str(self.scope_id), self.wintf.port), family=socket.AF_INET6,
                reuse_port=True)
            
            self.sock = Socket(transport, protocol)
            while self.ready:
                ts1 = time.time()
                data, addr, qsize = await self.sock.recvfrom()
                ts2 = time.time()
                self.battery.receivedRadio(ts2-ts1)
                if len(data) >= ct.DFLT_HDR_LEN and len(data) <= ct.MTU:
                    packet = Packet(data)
                    await self.rxPacket(Packet(data), round(time.time(), 4))

        asyncio.run(async_thread()) 

    def beaconWorker(self):
        """worker to schedual sending beacons to neighbors (triggered from the sink node)
        """        
        async def async_thread():
            while self.ready:
                await asyncio.sleep(self.betti['value']/ct.MILLIS_IN_SECOND)
                if self.isActive and self.sinkDistance < ct.DIST_MAX + 1:
                    for key, val in self.neighborTable.items():
                        if val.getToSinkDist() > self.sinkDistance+1:
                            await self.sendBeacon(acked=False, dst=Addr(re.sub(r'^.*?.', '', key)[1:]))
                    await self.updateLinkColor()
                await asyncio.sleep((ct.MAX_BE_TTI - self.betti['value'])/ct.MILLIS_IN_SECOND)
        asyncio.run(async_thread())
    
    def reportWorker(self):
        """worker to schedual sending reports to the controller
        """    
        async def async_thread(): 
            await asyncio.sleep(30)
            while self.ready:
                await asyncio.sleep(self.rptti['value']/ct.MILLIS_IN_SECOND)
                if self.isActive and self.sinkDistance < ct.DIST_MAX+1:
                    packet = await self.createReportPacket()
                    if self.isSink:
                        await self.controllerTx(packet)
                    else:
                        await self.runFlowMatch(packet)
                await asyncio.sleep((ct.MAX_RP_TTI - self.rptti['value'])/ct.MILLIS_IN_SECOND)
        asyncio.run(async_thread())    

    def statusWorker(self):
        """worker to publish the node's performance metrics (for evaluation purpose)
        """
        async def async_thread():
            self.zmqContext_visec = Context.instance()
            self.ctrlURL_visec = 'tcp://'+self.ctrl[0]+':'+str(self.ctrl[1]+3) 
            self.stvipub = self.zmqContext_visec.socket(zmq.PUB)
            self.stvipub.connect(self.ctrlURL_visec)
            while self.ready:
                if self.isActive:
                    data = {"id":self.id,
                            "lastupdate":round(time.time(), 4),
                            # "active":self.isActive,
                            "pos": self.pos['value'],
                            "batt":self.batt['value'],
                            "txbeacons":self.sentBeacons['rate'], 
                            "txbeacons_val":self.sentBeacons['value'], 
                            "txbeacons_val_ts":self.sentBeacons['value_ts'], 
                            "txpackets":self.sentPackets['rate'], 
                            "txpackets_val":self.sentPackets['value'], 
                            "txpackets_val_ts":self.sentPackets['value_ts'], 
                            "rxpackets":self.receivedPackets['rate'],
                            "rxpackets_val":self.receivedPackets['value'],
                            "rxpackets_val_ts":self.receivedPackets['value_ts'],
                            "txbytes":self.sentBytes['rate'],
                            "txbytes_val":self.sentBytes['value'],
                            "rxbytes":self.receivedBytes['rate'],
                            "rxbytes_val":self.receivedBytes['value'],
                            "txpacketsin":self.sentPacketsIn['rate'], 
                            "txpacketsin_val":self.sentPacketsIn['value'], 
                            "rxpacketsout":self.receivedPacketsOut['rate'],
                            "rxpacketsout_val":self.receivedPacketsOut['value'],
                            "txbytesin":self.sentBytesIn['rate'],
                            "txbytesin_val":self.sentBytesIn['value'],
                            "rxbytesout":self.receivedBytesOut['rate'],
                            "rxbytesout_val":self.receivedBytesOut['value'],
                            "drpackets":self.dropedPackets['rate'],
                            "drpackets_val":self.dropedPackets['value'],
                            "drpackets_val_ts":self.dropedPackets['value_ts'],
                            "rptti":self.rptti['value'],
                            "betti":self.betti['value']
                        }
                    self.stats.extend(b'ST'+b'ndtraffic'+str(json.dumps(data)).encode('utf-8')+b";")
                    self.stvipub.send(bytes(self.stats))
                    self.stats.clear()
                    self.batt['value_ts'] = 0
                    self.sentBeacons['value_ts'] = 0
                    self.sentPackets['value_ts'] = 0
                    self.sentPacketsIn['value_ts'] = 0
                    self.sentBytes['value_ts'] = 0
                    self.sentBytesIn['value_ts'] = 0
                    self.sentDataBytes['value_ts'] = 0
                    self.receivedPackets['value_ts'] = 0
                    self.receivedPacketsOut['value_ts'] = 0
                    self.receivedBytes['value_ts'] = 0
                    self.receivedBytesOut['value_ts'] = 0
                    self.receivedDataBytes['value_ts'] = 0
                    self.dropedPackets['value_ts'] = 0
                    self.pos['value_ts'] = 0
                    self.rptti['value_ts'] = 0
                    self.betti['value_ts'] = 0
                    await asyncio.sleep(2)
                else: await asyncio.sleep(1)
        asyncio.run(async_thread())

    async def runCmd(self, cmd):
        proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout = asyncio.subprocess.PIPE,
        stderr = asyncio.subprocess.PIPE)
        
        stdout, stderr = await proc.communicate()
        return stdout, stderr

    async def updateStats(self, ttype, packet=None): 
        """updates some performance metrics (for evaluation purpose)
        """
        if ttype == 'rx':
            ptype = packet.getType()
            ts1 = self.receivedPackets['ts']
            ts2 = round(time.time(), 4)
            self.receivedPackets['value'] += 1
            self.receivedPackets['value_ts'] += 1
            self.receivedPackets['rate'] = zero_division(self.receivedPackets['value'], (ts2 - ts1))
            # self.receivedPackets['ts'] = ts2
            ts1 = self.receivedBytes['ts']
            self.receivedBytes['value'] += packet.getLen()
            self.receivedBytes['value_ts'] += packet.getLen()
            self.receivedBytes['rate'] = zero_division(self.receivedBytes['value'], (ts2 - ts1))
            # self.receivedBytes['ts'] = ts2
            if ptype == ct.DATA:
                ts1 = self.receivedDataBytes['ts']
                self.receivedDataBytes['value'] += packet.getPayloadSize()
                self.receivedDataBytes['value_ts'] += packet.getPayloadSize()
                self.receivedDataBytes['rate'] = zero_division(self.receivedDataBytes['value'], (ts2 - ts1))            
                # self.receivedDataBytes['ts'] = ts2
        if ttype == 'rx-ctrl':
            ts1 = self.receivedPacketsOut['ts']
            ts2 = round(time.time(), 4)
            self.receivedPacketsOut['value'] += 1
            self.receivedPacketsOut['value_ts'] += 1
            self.receivedPacketsOut['rate'] = zero_division(self.receivedPacketsOut['value'], (ts2 - ts1))
            # self.receivedPacketsOut['ts'] = ts2
            ts1 = self.receivedBytesOut['ts']
            self.receivedBytesOut['value'] += packet.getLen()
            self.receivedBytesOut['value_ts'] += packet.getLen()
            self.receivedBytesOut['rate'] = zero_division(self.receivedBytesOut['value'], (ts2 - ts1))
            # self.receivedBytesOut['ts'] = ts2
        elif ttype == 'tx':
            ptype = packet.getType()
            ts1 = self.sentPackets['ts']
            ts2 = round(time.time(), 4)
            self.sentPackets['value'] += 1
            self.sentPackets['value_ts'] += 1
            self.sentPackets['rate'] = zero_division(self.sentPackets['value'], (ts2 - ts1))
            # self.sentPackets['ts'] = ts2
            ts1 = self.sentBytes['ts']
            self.sentBytes['value'] += packet.getLen()
            self.sentBytes['value_ts'] += packet.getLen()
            self.sentBytes['rate'] = zero_division(self.sentBytes['value'], (ts2 - ts1))
            # self.sentBytes['ts'] = ts2
            if ptype == ct.DATA:
                ts1 = self.sentDataBytes['ts']
                self.sentDataBytes['value'] += packet.getPayloadSize()
                self.sentDataBytes['value_ts'] += packet.getPayloadSize()
                self.sentDataBytes['rate'] = zero_division(self.sentDataBytes['value'], (ts2 - ts1))
                # self.sentDataBytes['ts'] = ts2
        elif ttype == 'tx-ctrl':
            ts1 = self.sentPacketsIn['ts']
            ts2 = round(time.time(), 4)
            self.sentPacketsIn['value'] += 1
            self.sentPacketsIn['value_ts'] += 1
            self.sentPacketsIn['rate'] = zero_division(self.sentPacketsIn['value'], (ts2 - ts1))
            # self.sentPacketsIn['ts'] = ts2
            ts1 = self.sentBytesIn['ts']
            self.sentBytesIn['value'] += packet.getLen()
            self.sentBytesIn['value_ts'] += packet.getLen()
            self.sentBytesIn['rate'] = zero_division(self.sentBytesIn['value'], (ts2 - ts1))
            # self.sentBytesIn['ts'] = ts2
        elif ttype == 'beacon':
            ts1 = self.sentBeacons['ts']
            ts2 = round(time.time(), 4)
            self.sentBeacons['value'] += 1
            self.sentBeacons['value_ts'] += 1
            self.sentBeacons['rate'] = zero_division(self.sentBeacons['value'], (ts2 - ts1))
            # self.sentBeacons['ts'] = ts2
        elif ttype == 'batt':
            ts1 = self.batt['ts']
            ts2 = round(time.time(), 4)
            self.batt['value'] = self.battery.getLevel()
            self.batt['value_ts'] = self.battery.getLevel()
            # self.batt['ts'] = ts2
        elif ttype == 'drop':
            ts1 = self.dropedPackets['ts']
            ts2 = round(time.time(), 4)
            self.dropedPackets['value'] += 1
            self.dropedPackets['value_ts'] += 1
            self.dropedPackets['rate'] = zero_division(self.dropedPackets['value'], (ts2 - ts1))
            # self.dropedPackets['ts'] = ts2
        elif ttype == 'rptti':
            ts1 = self.rptti['ts']
            ts2 = round(time.time(), 4)
            self.rptti['value'] = self.rptti['value']
            self.rptti['value_ts'] = self.rptti['value']
        elif ttype == 'betti':
            ts1 = self.betti['ts']
            ts2 = round(time.time(), 4)
            self.betti['value'] = self.betti['value']
            self.betti['value_ts'] = self.betti['value']
        elif ttype == 'pos':
            ts1 = self.pos['ts']
            ts2 = round(time.time(), 4)
            value2 = self.pos['value']
            self.pos['value'] = self.getPosition()
            self.pos['value_ts'] = self.getPosition()
            self.pos['rate'] = zero_division((value2 ^ self.pos['value']), (ts2 - ts1))
            self.pos['ts'] = ts2

    async def getFlowTableSize(self):
        """Get the OF size (number of entries)
        """
        # size of flowtable's entries in bytes
        size = 0
        for entry in self.flowTable.values():
            size += len(entry.toByteArray())
        return size
    
    async def initNeighborTable(self):
        """Wireless access medum"""
        """Wireless node can access the signal of nodes in its transmition range"""
        """This function is necessary to identify node's neighbors"""
        async with aiofiles.open('outputs/topo/'+str(self.topofilename), mode='r') as f:
            contents = await f.read()
        data = json.loads(contents)
        src_ngs = []
        src_id = data['nodeids'][data['nodeids'].index(self.id)]
        src_data = data['nodes'][data['nodeids'].index(self.id)]
        src_radio = IntfType.getParams(src_data['intftype'])
        for nd in data['nodeids']:
            dst_id = data['nodeids'][data['nodeids'].index(nd)]
            dst_data = data['nodes'][data['nodeids'].index(nd)]
            dst_radio = IntfType.getParams(dst_data['intftype'])
            dist = await self.getDistance(src_data['pos'], dst_data['pos'])
            if await self.isInRange(dist, dst_radio) and (dst_id != src_id):
                rssi = mapRSSI(await self.getRssi(src_radio, dst_radio, dist)) 
                addr = '{}.{}'.format(dst_id.split('.')[1], dst_id.split('.')[2])
                src_ngs.append({'id':dst_id,
                                'dist': dist,
                                'addr':addr,
                                'rssi':rssi,
                                'port':ct.BASE_NODE_PORT+dst_data['port']})
        if src_ngs:
            for ng in src_ngs:
                self.neighborTable[ng['id']] = Neighbor(ng['dist'], Addr(ng['addr']), ng['rssi'], ng["port"])            
    
    async def initFlowTable(self):
        """Add initial entries to the flow table
        Entry(0) -> Window(P.DST == myAddress) Window(P.TYP==3) Action(FORWARD_U myaddress) Stats(PERM, ..)
        the entry destination (dst) value is updated gradually during discovery process, so the node will end up 
        learn its distance, nextHopAddress to sink. If node is the sink itself, the distance is zero.
        for Non-sink motes, the distance to sink is initially set to 255 and change this value every time it compairse 
        its value to the received beacon packet from other nodes.
        """
        self.sinkAddress = self.myAddress
        toSink = Entry()
        # toSink.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1).setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST).setRhs(self.myAddress.intValue()))
        # toSink.addWindow(Window.fromString("P.TYP == 2"))
        toSink.addAction(ForwardUnicastAction(nxtHop=self.myAddress))
        toSink.getStats().setTtl(int(time.time()))
        toSink.getStats().setPermanent()        
        # add the initial forwarding rule to the sink at index TOSINK
        await self.insertRule(toSink)

    async def initStatusRegister(self):
        self.statusRegister = [0] * ct.STATUS_LEN
            
    async def initssdwsn(self):
        """Initializing necessary configurations before starting a Mote/Sink
        """
        await self.initStatusRegister()
        await self.initssdwsnSpecific()
        await self.initFlowTable()
  
    async def getOperand(self, packet, size, location, val):        
        if location == ct.NULL:
            return 0
        elif location == ct.CONST:
            return val
        elif location == ct.PACKET:
            intPacket = packet.toIntArray()   
            if size == ct.W_SIZE_0:
                if val >= len(intPacket):
                    return -1
                return intPacket[val]
            if size == ct.W_SIZE_1:
                if val + 1 >= len(intPacket):
                    return -1
                return mergeBytes(intPacket[val], intPacket[val + 1])
            return -1
        elif location == ct.STATUS:
            if size == ct.W_SIZE_0:
                if val >= len(self.statusRegister):
                    return -1
                return self.statusRegister[val]
            if size == ct.W_SIZE_1:
                if val + 1 >= len(self.statusRegister):
                    return -1
                return mergeBytes(self.statusRegister[val], self.statusRegister[val + 1])
            return -1
        else: return -1

    async def matchRule(self, fte, packet):
        """match a packet to a flow table entry

        Args:
            fte (Entry): flow table entry (Window, Rule/s, Stats)
            packet (_type_): different typs of packets (DATA, OPEN_PATH, REQUEST, RESPONSE, REPORT, BEACON)

        Returns:
            bool: matched? True/False
        """
        if fte.getWindows():        
            windows = fte.getWindows()     
            result = []   
            for window in windows:
                if await self.matchWindow(window, packet):
                    result.append(True)
            return len(result) == len(windows)
        else: False
        
    async def matchWindow(self, window, packet):
        """match every window in an entry to a packet header

        Args:
            window (Window): matching window in an entry
            packet (_type_): different typs of packets (DATA, OPEN_PATH, REQUEST, RESPONSE, REPORT, BEACON)

        Returns:
            bool: matched? True/False
        """
        operator = window.getOperator()
        size = window.getSize()
        lhs = await self.getOperand(packet, size, window.getLhsOperandType(), window.getLhs())
        rhs = await self.getOperand(packet, size, window.getRhsOperandType(), window.getRhs())
        return compare(operator, lhs, rhs)

    async def createBeaconPacket(self, acked, dst):
        """Create a Beacon Packet (needed for discovery process)"""
        p = BeaconPacket(net=self.myNet + (int(acked)*128), src=self.myAddress, dst=self.sinkAddress, distance=self.sinkDistance, battery=self.battery.getLevel(), 
            pos=self.position, intfType=await self.getIntfType(), sensorType=await self.getSensorType(), port=self.wintf.port-ct.BASE_NODE_PORT)
        if dst:
            p.setNxh(dst)
        else:
            p.setNxh(Addr(ct.BROADCAST_ADDR))     
        return p

    async def createReportPacket(self):
        """Create a Report Packet (for upating the controller with the Node's status/statistics"""
        p = ReportPacket(net=self.myNet, src=self.myAddress, dst=self.sinkAddress, distance=self.sinkDistance, battery=self.battery.getLevel(),
                         pos=self.position, intfType=await self.getIntfType(), sensorType=await self.getSensorType(), port=self.wintf.port-ct.BASE_NODE_PORT)   
        p.setNeighbors(self.neighborTable)
        return p
    
    async def sendBeacon(self, acked:bool, dst:Addr=None):
        packet = await self.createBeaconPacket(acked, dst)
        await self.txpacket(packet)
        await self.updateStats('beacon', packet)
    
    async def runAction(self, action, packet):
        """run a rule/s in a matched entry

        Args:
            action (Action): action/s in an entry
            packet (_type_): different typs of packets (DATA, OPEN_PATH, REQUEST, RESPONSE, REPORT, BEACON)
        """
        ac = action.getType()
        logger.info(f'{self.id} run ACTION ({action}) on (packet: {packet.getSrc()} --> {packet.getDst()})')
        if ac == Action.DROP.getValue():
            return
        elif ac == Action.FORWARD_U.getValue() or ac == Action.FORWARD_B.getValue():
            packet.setNxh(action.getNextHop())
            await self.txpacket(packet)
        elif ac == Action.SET.getValue():                
            operator = action.getOperator()
            lhs = await self.getOperand(packet, ct.W_SIZE_0, action.getLhsOperandType(), action.getLhs())
            rhs = await self.getOperand(packet, ct.W_SIZE_0, action.getRhsOperandType(), action.getRhs())
            if lhs == -1 or rhs == -1:
                logger.warn("Operators out of bound")
            
            res = self.doOperation(operator, lhs, rhs)
            if action.getResLocation() == ct.PACKET:
                p = packet.toIntArray()
                if action.getRes() >= len(p):
                    logger.warn("Result out of bound")
                p[action.getRes()] = res
                packet.setArray(p)
            else:
                self.statusRegister[action.getRes()] = res           
        elif ac == Action.ASK.getValue():
            rps = RequestPacket.createReqPacket(self.myNet, self.myAddress, self.sinkAddress, self.requestId + 1, packet.toByteArray())
            for rp in rps:
                await self.runFlowMatch(rp)
        elif ac == Action.MATCH.getValue():
            await self.txpacket(packet)  

    async def hasLinkToSink(self):
        """Check wether the node has already base link to the sink
        """      
        return bool(self.flowTable.get_matching('dst:'+str(self.sinkAddress.intValue())))

    async def getNextHopVsSink(self):
        """nex hop vs sink is the first configured entry at index TOSINK.

        Returns:
            Addr: next hop address
        """
        addr = self.myAddress
        matches = self.flowTable.get_matching('dst:'+str(self.sinkAddress.intValue()))
        if matches:
            max_key = max(matches, key=len)
            entry = matches[max_key]
            for action in entry.getActions():
                if action.getTypeName() == 'FORWARD_U':
                    addr = action.getNextHop()
        return addr
    
    async def searchRule(self, rule:Entry):
        """search if a same entry is installed previously. Prevent doublicated installed entries.

        Args:
            rule (Entry): a flow table entry

        Returns:
            int: the index of an entry if found, otherwise return -1
        """
        for key in self.flowTable:               
            if self.flowTable[key].__eq__(rule):
                return key
        return -1
            
    async def insertRule(self, rule:Entry):
        """insert non-dublicated entries to the flow table.

        Args:
            rule (Entry): flow table entry
        """
        # generate the wildcard key
        key = Entry.getEntryWCKey(rule)
        if not key:
            key = await self.searchRule(rule)
        self.flowTable[key] = rule

    async def  isAcceptedIdPacket(self, packet):
        return await self.isAcceptedIdAddress(packet.getDst())

    def isAcceptedIdAddress(self, addr:Addr):
        return (addr.__eq__(self.myAddress) or addr.isBroadcast() or self.acceptedId.__contains__(addr))

    async def execWriteConfig(self, packet:ConfigPacket):
        """Execute the payload of a Write Configuration Packet"""
        val = packet.getParams()
        conf = packet.getConfigId()
        ts = packet.getTS()
        if conf == ConfigProperty.MY_ADDRESS.getValue():
            self.myAddress = Addr(val)
        elif conf == ConfigProperty.DIST.getValue():
            self.sinkDistance == int.from_bytes(val, byteorder='big', signed=False)
        elif conf == ConfigProperty.REM_RULE.getValue():
            for key in self.flowTable.get_matching(val.decode()):
                logger.info(f"{self.id} REMOVE rule ({self.flowTable[key]}) at key {key}")
                # self.flowTable.pop(key)
                if key != '.*':
                    del self.flowTable[key]
        elif conf == ConfigProperty.ADD_RULE.getValue():
            entry = Entry(val)
            if len(entry.windows):
                await self.insertRule(entry)
        elif conf == ConfigProperty.RESET.getValue():
            self.reset()
        elif conf == ConfigProperty.DRL_ACTION.getValue():
            idx = 0
            if val[idx] == ct.DRL_AG_INDEX:   
                # '''
                if self.ctrl[2] == 'ATCP-ctrl':
                    self.isAggr = bool(int.from_bytes(val[idx+1:idx+1+ct.DRL_AG_LEN], byteorder='big', signed=False))
                idx += ct.DRL_AG_LEN+1
                # '''
            if val[idx] == ct.DRL_NH_INDEX:
                # '''
                act_nxh_node = Addr(val[idx+1:idx+1+ct.DRL_NH_LEN])
                if not act_nxh_node.__eq__(self.myAddress):  
                    entry = Entry()
                    entry.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1)
                        .setLhsOperandType(ct.PACKET).setLhs(ct.SRC_INDEX).setRhsOperandType(ct.CONST)
                        .setRhs(self.myAddress.intValue()))
                    # entry.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1)
                    #     .setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST)
                    #     .setRhs(self.sinkAddress.intValue()))
                    # entry.addWindow(Window.fromString("P.TYP == 2"))
                    entry.addAction(ForwardUnicastAction(nxtHop=act_nxh_node))
                    entry.getStats().setTtl(int(time.time()))
                    entry.getStats().setPermanent() #added   

                    # self.sinkDistance = self.neighborTable[str(self.myNet)+'.'+act_nxh_node.__str__()].getDist() + 1
                    # self.sinkRssi = self.neighborTable[str(self.myNet)+'.'+act_nxh_node.__str__()].getRssi()         
                    if len(entry.windows):
                        await self.insertRule(entry)
                else:
                    for key in self.flowTable.get_matching('src:'+str(self.myAddress.intValue())):
                        logger.info(f"{self.id} REMOVE rule ({self.flowTable[key]}) at key {key}")
                        if key != '.*':
                            del self.flowTable[key]
                # '''
                idx += ct.DRL_NH_LEN+1
            if val[idx] == ct.DRL_RT_INDEX:
                # '''
                if self.ctrl[2] == 'ATCP-ctrl':
                    self.rptti['value'] = abs((round(time.time(), 4) - ts)*ct.MILLIS_IN_SECOND + int.from_bytes(val[idx+1:idx+1+ct.DRL_RT_LEN], byteorder='big', signed=False))
                    if self.rptti['value'] > ct.MAX_RP_TTI:
                        self.rptti['value'] -= ct.MAX_RP_TTI
                    await self.updateStats('rptti')
                # '''
                idx += ct.DRL_RT_LEN+1

        elif conf == ConfigProperty.IS_AGGR.getValue():
            self.isAggr = bool(int.from_bytes(val, byteorder='big', signed=False))

    async def execReadConfig(self, packet:ConfigPacket):
        """Execute the payload of a Read Configuration Packet"""
        val = packet.getParams()
        size = len(packet.getConfigId())
        conf = packet.getConfigId()

        if conf == ConfigProperty.MY_ADDRESS.getValue():
            packet.setParams(self.myAddress.getArray(), size)
        elif conf == ConfigProperty.DIST.getValue():
            packet.setParams(int(self.sinkDistance).to_bytes(), size)
        elif conf == ConfigProperty.GET_RULE.getValue():
            i = int(val[0]) # index of the rule
            if i < len(self.flowTable):
                pass
            else: return False
        
        return True
                    
    async def execConfig(self, packet:ConfigPacket):
        """execute the config packet

        Args:
            packet (ConfigPacket): received CONFIG packet

        Returns:
            bool: True/False (True means read the requested config and send it back to the controller)
        """
        toBeSent = False
        if packet.isWrite():
            await self.execWriteConfig(packet)
        else: toBeSent = await self.execReadConfig(packet)
        return toBeSent
    
    async def runFlowMatch(self, packet):
        """find a matched entry in the flow table.

        Args:
            packet (_type_): unknown packet (search for a matching forwarding rule)
        """
        matches = []
        key = 'src:'+str(packet.getSrc().intValue())+'dst:'+str(packet.getDst().intValue())+'typ:'+str(packet.getType())
        matches = self.flowTable.get_matching(key)
        if matches:
            # get the most specific matched entry (rule)
            max_key = max(matches, key=len)
            entry = matches[max_key]
            for action in entry.getActions():
                await self.runAction(action, packet)
            entry.getStats().restoreIdle()
            entry.getStats().increasePCounter()
            entry.getStats().increaseBCounter(len(packet.toByteArray()))
            logger.info(f'{self.id} MATCHED rule ({entry}) entry_key({max_key}) for {packet.getTypeName()} Packet')           
        # if not matched, forward the unknown packet to the controller
        if not matches:
            rps = RequestPacket.createReqPacket(self.myNet, self.myAddress, self.sinkAddress, self.requestId+ 1, packet.toByteArray())
            for rp in rps:
                await self.runFlowMatch(rp)

    def remvExpiredEnties(self, span_time):
        """update the flow table. Remove entries with expired idle-timeout.
        """
        remvList = []
        for item, value in self.flowTable.items():
            idletimeout = value.getStats().getIdle()
            if idletimeout != ct.RL_IDLE_PERM:
                if idletimeout >= 1:
                    value.getStats().decrementIdle(abs(int(span_time)))
                else:
                    remvList.append(item)
        for item in remvList:
            logger.info(f"{self.id} REMOVE rule ({value.__str__()}) at key {item}")
            if item != '.*':
                del value

    async def updateLinkColor(self):
        """Updating the link color (for visualization purpose)"""
        if self.isActive:
            links = []
            source = self.id    
            sink_target = str(self.myNet)+'.'+(await self.getNextHopVsSink()).__str__()
            forward_nodes = []
            for key, entry in self.flowTable.items():
                for action in entry.getActions():
                    if action.getTypeName() == 'FORWARD_U':
                        forward_nodes.append(str(self.myNet)+'.'+action.getNextHop().__str__())
            for node in self.neighborTable:
                if node in forward_nodes:
                    if node == sink_target:
                        links.append({'source':source, 'target':node, 'color':'green'})
                    else:
                        links.append({'source':source, 'target':node, 'color':'orange'})
                else:
                    links.append({'source':source, 'target':node, 'color':'black'})
            
            self.stats.extend(b'ST'+b'LINK'+str(json.dumps({'id':source, 'links':links})).encode('utf-8')+b';')
            if not self.isSink:
                self.stats.extend(b'ST'+b'NODE'+str(json.dumps({'id':self.id, 'color':'orange' if self.isAggr else 'blue'})).encode('utf-8')+b';')
    
    async def rxHandler(self, p:list):
        """handle received packet

        Args:
            packet (Packet): different type of packets
            rssi (_type_): received signal strength
        """
        packet = p[0]
        dist = p[1]
        rssi = p[2]
        if packet.getLen() > ct.DFLT_HDR_LEN and packet.getNet() == self.myNet and packet.getTtl() != 0:
            ptype = packet.getType()
            if ptype == ct.DATA:
                #receive data packet
                await self.rxData(DataPacket(packet.toByteArray()))
            elif ptype == ct.BEACON:
                #receive beacon packet
                await self.rxBeacon(BeaconPacket(packet.toByteArray()), dist, rssi)
            elif ptype == ct.REPORT:
                #receive report packet
                await self.rxReport(ReportPacket(packet.toByteArray()))
            elif ptype == ct.REQUEST:
                #receive request packet (routing request to the controller)
                await self.rxRequest(RequestPacket(packet.toByteArray()))
            elif ptype == ct.RESPONSE:
                #receive response packet (routing response from the controller)
                await self.rxResponse(ResponsePacket(packet.toByteArray()))
            elif ptype == ct.OPEN_PATH:
                #receive open-path packet (to setup a routing rules to certain packet from/to the controller)
                await self.rxOpenPath(OpenPathPacket(packet.toByteArray()))
            elif ptype == ct.CONFIG:
                #receive config packet (write/read configuration to/from a node)
                await self.rxConfig(ConfigPacket(packet.toByteArray()))
            elif ptype == ct.AGGR:
                #receive aggregation packet
                await self.rxAggr(AggrPacket(packet.toByteArray()))            
   
    async def rxData(self, packet:DataPacket):
        """handle received data packets.

        Args:
            packet (_type_): DATA packet
        """ 
        if self.isSink:
            return
        await self.runFlowMatch(packet)
                
    async def rxBeacon(self, packet:BeaconPacket, dist, rssi):
        """handle received beacon packets.

        Args:
            packet (_type_): BEACON packet
        """
        if rssi >= ct.RSSI_MIN:
            tosink_dist = packet.getDistance()
            if tosink_dist+1 < self.sinkDistance: #(tosink_dist < self.sinkDistance) dynamic to sink link
                # advertisment from node to sink
                prvNxtHop = await self.getNextHopVsSink()
                self.sinkAddress = packet.getDst()
                self.isActive = True
                self.sinkDistance = tosink_dist+1   
                self.sinkRssi = rssi         
                toSink = Entry()
                # toSink.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1)
                #     .setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST)
                #     .setRhs(self.sinkAddress.intValue()))
                # toSink1.addWindow(Window.fromString("P.TYP == 2"))
                toSink.addAction(ForwardUnicastAction(nxtHop=packet.getSrc()))
                toSink.getStats().setTtl(int(time.time()))
                toSink.getStats().setPermanent() #added    
                await self.insertRule(toSink)

            if tosink_dist+1 != self.sinkDistance and packet.getSrc().__eq__(await self.getNextHopVsSink()):
                # update distance to sink in case the route to sink is updated
                self.sinkDistance = tosink_dist + 1
            # update neighborTable
            target = f'{packet.getNet()}.{packet.getSrc().__str__()}'
            self.neighborTable[target] = Neighbor(dist, packet.getSrc(), rssi, packet.getPort(), tosink_dist)
            if not packet.isAcked():
                await self.sendBeacon(acked=True, dst=packet.getSrc())
            await self.updateLinkColor()

    async def rxOpenPath(self, packet:OpenPathPacket):
        """handle received OPEN_PATH packets.

        Args:
            packet (_type_): OPEN_PATH packet
        """
        if await self.isAcceptedIdPacket(packet):
            path = packet.getPath()
            index= 0
            for i in range(len(path)-1):
                actual = path[i]
                if await self.isAcceptedIdAddress(actual):
                    break
                index += 1

            if index > 0:
                rule = Entry()
                rule.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1).setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST).setRhs(path[0].intValue()))
                rule.getWindows().extend(packet.getWindows())
                rule.addAction(ForwardUnicastAction(nxtHop=path[index - 1]))
                await self.insertRule(rule)

            if index < (len(path) - 1):
                rule = Entry()
                rule.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1).setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST).setRhs(path[len(path) - 1].intValue()))
                rule.getWindows().extend(packet.getWindows())
                rule.addAction(ForwardUnicastAction(nxtHop=path[index + 1]))
                await self.insertRule(rule)
                packet.setDst(path[index + 1])
                packet.setNxh(path[index + 1])
                await self.txpacket(packet)
        else:
            await self.runFlowMatch(packet)

    async def rxReport(self, packet:ReportPacket):
        """handle received REPORT packets.

        Args:
            packet (_type_): REPORT packet
        """
        if self.isSink:
            return await self.controllerTx(packet)
            # '''
        if self.isAggr:
            # aggregate non-redundant network view
            src_id = f'{packet.getNet()}.{packet.getSrc()}'
            if self.aggrMembs.get(src_id):
                pl = await packet.getAggrPayload(self.aggrMembs[src_id])
                if pl:
                    self.cachedReports.append(pl)
                self.aggrMembs[src_id] = packet.toByteArray()
            else:
                #save a copy of the packet to be used next time receiving packet from same src
                #TODO configure aggr head to be secure as it should have access to parse the packet
                self.aggrMembs[src_id] = packet.toByteArray()
                # packet.setDst(self.sinkAddress)
                # packet.setNxh(await self.getNextHopVsSink())
                await self.runFlowMatch(packet)
            #cache x report packets
            #create aggregation packet
            cachesize = sum([len(pl) for pl in self.cachedReports])
            if cachesize:
                if cachesize > (ct.MTU - (ct.DFLT_HDR_LEN + ct.AGGR_HDR_LEN)):
                    aggrP = AggrPacket.createAggrPacket(net=self.myNet, src=self.myAddress, dst=self.sinkAddress, type=ct.REPORT, aggr=self.cachedReports[:-1])
                    del self.cachedReports[:-1]
                    #forward the aggregated packet to the nearest sink
                    await self.runFlowMatch(aggrP)
        else:           
            cachesize = sum([len(pl) for pl in self.cachedReports])
            if cachesize:
                if cachesize > (ct.MTU - (ct.DFLT_HDR_LEN + ct.AGGR_HDR_LEN)):
                    aggrP1 = AggrPacket.createAggrPacket(net=self.myNet, src=self.myAddress, dst=self.sinkAddress, type=ct.REPORT, aggr=self.cachedReports[:-1])
                    aggrP2 = AggrPacket.createAggrPacket(net=self.myNet, src=self.myAddress, dst=self.sinkAddress, type=ct.REPORT, aggr=self.cachedReports[-1])
                    await self.runFlowMatch(aggrP1)
                    await self.runFlowMatch(aggrP2)
                else:
                    aggrP = AggrPacket.createAggrPacket(net=self.myNet, src=self.myAddress, dst=self.sinkAddress, type=ct.REPORT, aggr=self.cachedReports)
                    await self.runFlowMatch(aggrP)
                self.cachedReports = []
                self.aggrMembs = {}
            #forward the aggregated packet to the next sink/sub_sink
            await self.runFlowMatch(packet)

    async def rxAggr(self, packet:AggrPacket):
        """handle received Aggregate packets.

        Args:
            packet (_type_): Aggregate packet
        """
        if self.isSink:
            return await self.controllerTx(packet)
        await self.runFlowMatch(packet)

    async def rxConfig(self, packet:ConfigPacket):
        """handle received CONFIG packets.

        Args:
            packet (_type_): CONFIG packet
        """
        dst = packet.getDst()
        if self.isSink:
            if dst.__eq__(self.myAddress):
                await self.controllerTx(packet)
            else:
                path = packet.getPath()
                i = 0
                for p in path[:-1]:
                    if p.__eq__(self.myAddress):
                        packet.setNxh(path[i + 1])
                        await self.txpacket(packet)
                        break
                    i += 1
            
        elif dst.__eq__(self.myAddress):
            if await self.execConfig(packet):
                # if config packet is read-only, execute it and send the results back to the controller
                packet.setSrc(self.myAddress)
                path = packet.getPath().reverse()
                packet.setDst(path[len(path)-1])
                packet.setTtl(ct.TTL_MAX)
                await self.txpacket(packet)
            
        else:
            path = packet.getPath()
            i = 0
            for p in path[:-1]:
                if p.__eq__(self.myAddress):
                    packet.setNxh(path[i + 1])
                    await self.txpacket(packet)
                    break
                i += 1
                
    async def rxRequest(self, packet:RequestPacket):
        """handle received REQUEST packets
        forward the received REQUEST packet to the controller if current is gatway node 
        otherwise forward to the next node vs gateway.

        Args:
            packet (_type_): REQUEST packet
        """
        if await self.isAcceptedIdPacket(packet):
            if self.isSink:
                await self.controllerTx(packet)
            else:
                await self.runFlowMatch(packet)      
        elif await self.isAcceptedIdAddress(packet.getNxh()):
            if self.aggrDistance == 0:
                #TODO add the logic of aggregating data packets
                packet.setDst(self.sinkAddress)
                await self.runFlowMatch(packet)
            else:
                # find matched forwarding rule to the sink
                await self.runFlowMatch(packet)

    async def rxResponse(self, packet:ResponsePacket):
        """handle received RESPONSE packets.

        Args:
            packet (_type_): RESPONSE packet
        """
        if await self.isAcceptedIdPacket(packet):
            entry = packet.getRule()
            await self.insertRule(entry)
        else:
            await self.runFlowMatch(packet)
    
    async def getIntfType(self):
        """Get Node's Wireless interface type"""
        return self.wintf.type
    
    async def getSensorType(self):
        """Get Node's Sensor type"""
        if self.isSink:
            return SensorType.sink.getValue()
        else: return self.sensorType.type

    async def getPosition(self):
        """Get Node's Position in the simulation Graph"""
        pos = self.position
        x, y, z = round(pos[0], 0), round(pos[1], 0), round(pos[2], 0)
        return x, y, z
    
    async def setPosition(self, pos):
        """Set the Node's position in the simulation Graph"""
        self.position = (*[int(x) for x in pos.split(",")],) 
        
    async def getDistance(self, src_pos, dst_pos):
        """get distance between two nodes

        Args:
            src (tuple): source node position xyz
            dst (tuple): destination node position xyz

        Returns:
            int: distance
        """
        pos_src = [x for x in src_pos]
        pos_dst = [x for x in dst_pos]
        if len(pos_src) < 3:
            pos_src.append(0)
        if len(pos_dst) < 3:
            pos_dst.append(0)
        x = (int(pos_src[0]) - int(pos_dst[0])) ** 2
        y = (int(pos_src[1]) - int(pos_dst[1])) ** 2
        z = (int(pos_src[2]) - int(pos_dst[2])) ** 2
        dist = sqrt(x + y + z)
        return round(dist, 2)
    
    async def isInRange(self, dist, dst_radio):
        """check if node is in the transmission range of another node

        Args:
            dist (float): distance between two nodes
            dst_txrange (int): distination node radio tx range

        Returns:
            bool: True/False
        """
        if dist > dst_radio['antRange']:
            return False
        return True

    async def getRssi(self, src_radio, dst_radio, dist):
        """calculate the expected rssi using the configured propagation model

        Args:
            dst_radio (dict): source node radio params
            dst_radio (dict): destination node radio params

        Returns:
            int: rssi (dB)
        """
        return int(self.ppm.rssi(src_radio, dst_radio, dist))
    
    async def getSnr(self, src, dst, noise_th):
        """calculate signal to noise ratio using rssi (dB)
        SNR = RSSI – RF background noise

        Args:
            dst (NodeView): destination node
            noise_th (int): background noise (dB)

        Returns:
            int: Signal-to-Noise Ratio (dB)
        """
        return await self.getRssi(self, src, dst) - noise_th

    async def getTxPowerGivenRange(self):
        """calculate the transmission power of radio antenna using the given transmission range.
        for 2.4 GHz The limit is set to 20 dBm (100 mW) for OFDM and 18 dBm (63 mW) for CCK.
        """
        txpower = self.ppm.txPower(self)
        await self.setTxPower(txpower)
        if self.wintf.params['txPower'] == 1:
            min_range = int(self.ppm.range(self))
            if self.wintf.params['antRange'] < min_range:
                logger.info(f'{self.id}: the signal range should be changed to (at least) {min_range}')
        else:
            logger.info(f"{self.id}: the signal range of {self.wintf.params['antRange']} requires tx power equal to {txpower}dBm")
            
    async def setDefaultRange(self):
        self.wintf.params['antRange'] = self.ppm.range(self)

    async def setAntennaGain(self, gain):
        self.wintf.params['antGain'] = int(gain)
        await self.setDefaultRange()

    async def setAntennaHeight(self, height):
        self.wintf.params['antHeight'] = int(height)
        await self.setDefaultRange()

    async def setRange(self, range):
        self.wintf.params['antRange'] = int(range)
        await self.getTxPowerGivenRange()

    async def setTxPower(self, txpower):
        self.wintf.params['txPower'] = int(txpower)
        await self.setDefaultRange()
        
    async def getIP6FromSeq(self, port=None):
        """
        To generate the ip6 from the node sequance
        * seq: the port - base_port -> the seq # of the node
        """
        seq = self.seq + 5
        if not(port is None):
            seq = (port - ct.BASE_NODE_PORT) + 5
        return ipAdd6(seq, prefixLen=64, ipBaseNum=ct.BASE_IP6) +'/%s' % 64
    
class ClosedError(Exception):
    pass


class _SocketProtocol:

    def __init__(self, isSink):
        self._error = None
        if isSink:
            self._packets = janus.Queue(0).async_q
        else:
            self._packets = janus.Queue(ct.BUFFER_SIZE).async_q

    def connection_made(self, transport):
        pass

    def connection_lost(self, transport):
        self._packets.put_nowait(None)

    def datagram_received(self, data, addr):
        self._packets.put_nowait((data, addr, self._packets.qsize()))

    def error_received(self, exc):
        self._error = exc
        self._packets.put_nowait(None)

    async def recvfrom(self):
        return await self._packets.get()

    def raise_if_error(self):
        if self._error is None:
            return

        error = self._error
        self._error = None

        raise error
    
class Socket:
    """A UDP socket. Use :func:`~asyncudp.create_socket()` to create an
    instance of this class.

    """

    def __init__(self, transport, protocol):
        self._transport = transport
        self._protocol = protocol

    def close(self):
        """Close the socket.

        """

        self._transport.close()

    def sendto(self, data, addr=None):
        """Send given packet to given address ``addr``. Sends to
        ``remote_addr`` given to the constructor if ``addr`` is
        ``None``.

        Raises an error if a connection error has occurred.

        >>> sock.sendto(b'Hi!')

        """

        self._transport.sendto(data, addr)
        self._protocol.raise_if_error()

    async def recvfrom(self):
        """Receive a UDP packet.

        Raises ClosedError on connection error, often by calling the
        close() method from another task. May raise other errors as
        well.

        >>> data, addr = sock.recvfrom()

        """

        packet = await self._protocol.recvfrom()
        self._protocol.raise_if_error()

        if packet is None:
            raise ClosedError()

        return packet

    def getsockname(self):
        """Get bound infomation.

        >>> local_address, local_port = sock.getsockname()

        """

        return self._transport.get_extra_info('sockname')
    
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        self.close()

class FlowTable(dict):
    """Wildcard support dict that represents an OF table"""
    def __init__(self, node):
        super().__init__()
        self.node = node
    
    def __delitem__(self, key) -> None:
        super().__delitem__(key)
        asyncio.create_task(self.node.updateLinkColor())
   
    def __setitem__(self, item, value):
        if self.get(item):
            super().__setitem__(item, value)
            logger.info(f"{self.node.id} UPDATE rule ({value.__str__()}) at index {item}")
        else:
            if len(self) < ct.FT_MAX_ENTRIES or self.node.isSink:
                super().__setitem__(item, value)
                logger.info(f"{self.node.id} INSERT rule ({value.__str__()}) at index {item}")
            else: logger.info(f"{self.id} FULL flowtable of size:{len(self)}")
        
        asyncio.create_task(self.node.updateLinkColor())

    def pop(self, key, nval=None):
        super().pop(key, nval)        
        asyncio.create_task(self.node.updateLinkColor())
    
    def get_matching(self, event):
        return {key:self[key] for key in self if re.match(key, event)}    

class Mote(Node):
    """Wireless Mote"""
    def __init__(self, net:int, addr:Addr, portSeq:int, batt:int=None, pos:tuple=None, target:str=None, topo=None, ctrl=None, cls=None):
        """create a wireless mote

        Args:
            net (int): network subnet
            addr (Addr): the address of a node is two bytes with a maximum broadcasting address 255.255
            portSeq (int): sequance number of a node (incremental) a node port = base_port + portSeq
            batt (int, optional): battery charging level. Defaults to None.
            pos (tuple, optional): position in the grid (x,y,z). Defaults to None.
            target (str, optional): for simulation purpose a target destination node is assigned to each node. Defaults to None.
            cls (_type_, optional): sensor device attached to wireless node. Defaults to None.
        """
        self.id = f'{net}.{addr}'
        super().__init__(net, addr, portSeq, Dischargable(batt), pos, topo, ctrl, cls)
        # Mote's tx and rx Queues
        self.aggrMembs = {}     

    async def dataCallback(self, packet:DataPacket):
        """after a data packet has been received

        Args:
            packet (DataPacket): _description_
        """
        #TODO add a proper logic (as per the simulation purpose)
        pass

    def sensingQueueWorker(self):
        """Sensor worker to run and receive sensor's data"""
        async def async_thread():
            await asyncio.sleep(30)
            self.sensingQueue = janus.Queue(ct.BUFFER_SIZE).async_q
            asyncio.get_running_loop().run_in_executor(None, self.sensorType.run)
            seq = 0
            while self.ready:
                if self.isActive:
                    recv = await self.sensingQueue.get()
                    if self.isActive:
                        data = recv[0]
                        sqNo = recv[1]
                        sensingTime = recv[2]
                        #TODO partition data greater than DFLT_PAYLOAD_LEN to aggreate at the destination (sink)
                        packet = DataPacket(net=self.myNet, src=self.myAddress, dst=self.sinkAddress, payload=data)             
                        await self.runFlowMatch(packet)
                        seq += 1
                else: await asyncio.sleep(1)
        asyncio.run(async_thread())

    async def initssdwsnSpecific(self):
        """Initiating specific parameters configuration for a Mote"""
        self.sinkDistance = ct.DIST_MAX + 1
        self.sinkRssi = 0

class Sink(Node):
    """Sink/Gateway Node 
    a node that communicate directly with the controller"""
    def __init__(self, net:int, addr:Addr, portSeq:int, dpid:str, pos:tuple=None, topo=None, ctrl=None):
        """create a wireless mote

        Args:
            net (int): network subnet
            addr (Addr): the address of a node is two bytes with a maximum broadcasting address 255.255
            portSeq (int): sequance number of a node (incremental) a node port = base_port + portSeq
            dpid (str): datapath id
            pos (tuple, optional): position in the grid (x,y,z). Defaults to None.
            cls (_type_, optional): sensor device attached to wireless node. Defaults to None.
        """
        
        self.id = f'{net}.{addr}'
        super().__init__(net, addr, portSeq, Chargable(), pos, topo, ctrl)
        # Sinks tx and rx Queues (assume the sink has unlimited queue for sending/receiving packets)
        self.dpid = dpid
        self.addrController = ctrl
        
    def connCtrlWorker(self):
        """Worker for advertising Sink itself to the controller every 5 seconds"""
        async def async_thread():
            self.zmqContext_sec = Context.instance()
            self.ctrlURL_sec = 'tcp://'+self.ctrl[0]+':'+str(self.ctrl[1]+1)
            self.publisher = self.zmqContext_sec.socket(zmq.PUB)
            # self.publisher.setsockopt(zmq.SNDHWM, ct.CTRL_BUFF_SIZE)
            # self.publisher.setsockopt(zmq.NOBLOCK)
            self.publisher.connect(self.ctrlURL_sec)  
            while self.ready:
                if self.isActive:
                    rpp = RegProxyPacket(net=self.myNet, src=self.myAddress, dPid=self.dpid, mac=self.wintf.getMac(), port=self.wintf.port, isa=self.addrController)  
                    await self.controllerTx(rpp)
                    await asyncio.sleep(5)
                else: await asyncio.sleep(1)
        asyncio.run(async_thread())

    def packetOutWorker(self):
        """worker: listen to the controller Packet-Out publisher
        """
        async def async_thread():
            self.zmqContext_pri = Context.instance()
            self.ctrlURL_pri = 'tcp://'+self.ctrl[0]+':'+str(self.ctrl[1])
            self.subscriber = self.zmqContext_pri.socket(zmq.SUB)
            # self.subscriber.setsockopt(zmq.SUBSCRIBE, b'')
            self.subscriber.setsockopt(zmq.RCVHWM, ct.CTRL_BUFF_SIZE)
            self.subscriber.connect(self.ctrlURL_pri)
            self.subscriber.subscribe("OUT")
            while self.ready:
                if self.isActive:
                    packet = await self.subscriber.recv()
                    await self.rxPacket(Packet(packet[len('OUT'):]).setTS(round(time.time(), 4)), round(time.time(), 4), 'rx-ctrl')
                else: await asyncio.sleep(1)
        asyncio.run(async_thread())

    async def controllerTx(self, packet):
        """receives packets to be passed to the controller channel queue (ready-to-be transmitted packets)

        Args:
            packet (_type_): different typs of packets (REQUEST, REPORT)
        """   
        self.publisher.send(b'IN'+packet.toByteArray())
        await self.updateStats('tx-ctrl', packet)

    async def dataCallback(self, packet: DataPacket):
        await self.controllerTx(packet)

    async def getIntfType(self):
        return self.wintf.type
    
    async def initssdwsnSpecific(self):
        """Initiating specific parameters configuration for a Sink"""
        self.sinkDistance = 0
        self.sinkRssi = ct.CTRL_RSSI
        self.isActive = True
        self.isSink = True