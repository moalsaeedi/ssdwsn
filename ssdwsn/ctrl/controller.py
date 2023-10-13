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

from cProfile import run
from cmath import inf
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict, deque
import re
from os import cpu_count, path
import gym
from gym import spaces
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
from numpy.core.fromnumeric import ndim, size
from sklearn import preprocessing
import scipy.stats as stats
from ssdwsn.app.routing import Dijkstra
from ssdwsn.ctrl.graph import Graph
import time
import asyncio, logging
from ssdwsn.openflow.action import ForwardUnicastAction, DropAction
from ssdwsn.openflow.window import Window
from ssdwsn.util.utils import mergeBytes, Suspendable, CustomFormatter, getInterArrivalTime, moteAddr, mapRSSI, portToAddrStr, getColorVal, setBitRange, bitsToBytes, zero_division
from app.utilts import RunningMeanStd
from math import ceil, log, sqrt, atan2, pi
# from ssdwsn.util.log import warn, info, debug, error, output
from ssdwsn.openflow.entry import Entry
from ssdwsn.data.addr import Addr
from ssdwsn.app.routing import Dijkstra
from ssdwsn.openflow.packet import ConfigPacket, ConfigProperty, OpenPathPacket, Packet, DataPacket, ReportPacket, RequestPacket, ResponsePacket, AggrPacket, RegProxyPacket
from ssdwsn.data.sensor import SensorType
from ssdwsn.data.intf import IntfType
# from ssdwsn.ctrl.networkGraph import dijkstra, shortest_path, Graph
from ssdwsn.util.constants import Constants as ct
from ctypes import ArgumentError, c_uint32 as unsigned_int32
from expiringdict import ExpiringDict
import zmq
from zmq.asyncio import Context, Poller, ZMQEventLoop

import janus
import socketio
from socketio import exceptions
import json
from sys import stderr
import functools
import signal
from concurrent.futures import CancelledError
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split       

import zmq
from zmq.asyncio import Context, Poller, ZMQEventLoop

#logging----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
#-----------------------------------

class Controller:    
    """Controller Process"""
    def __init__(self, inetAddress:tuple=None, networkGraph:Graph=None):
        """Create a controller process
        Args:
            inetAddress (tuple, optional): controller address (IP:Port). Defaults to None.
            networkGraph (Graph, optional): network view. Defaults to None.
        """
        self.id = self.id = 'Controller-%s:%s'% (inetAddress[0], inetAddress[1])
        self.isStopped = False
        # self.cmd = input() # TODO
        # Packet-In/Out queues (buffers of controller's channel to the data plane)
        # self.packetInQueue = None
        # self.packetOutQueue = asyncio.Queue(0, loop=self.loop)
        # expiring cache dict of request packets
        self.requestCache = ExpiringDict(max_len=ct.CACHE_MAX_SIZE, max_age_seconds=ct.CACHE_EXP_TIME) 
        self.aggrCache = ExpiringDict(max_len=ct.CACHE_MAX_SIZE, max_age_seconds=ct.CACHE_EXP_TIME) 
        # controller's address (IP, port)
        self.inetAddress = inetAddress 
        # Network View
        self.networkGraph = networkGraph
        # Flow requests statistics (dict key: tuple of src and dst (src, dst))
        self.flowReqStats = defaultdict(list)
        self.rule2id = defaultdict(list)
        self.flowSetupSeq = []
        # sinks/gateways assigned to the controller
        self.sinks = []
        # aggregats nodes
        self.aggregates = []
        # list of the async workers
        self.workers = []
        # socketio
        self.sio = None
        self.zmqContext = None
        # event loop
        self.loop = None
        self.txpub = None
        self.txsub = None
        self.rxpub = None
        self.rxsub = None
        self.publisher = None
        self.subscriber = None
        self.txvipub = None
        self.txvisub = None
        self.rxvipub = None
        self.rxvisub = None
        self.vipub = None
        self.visub = None
        # Buffer of statistics from data plane
        self.buffer = {}
        self.buffer_size = 5
        self.perf_buffer = {}
        self.perf_buffer_size = 5

    async def terminatef(self):
        logger.warn(f'CONTROLLER {self.id} has been terminated ...')
        await self.sio.disconnect()
        self.txpub.close()
        self.txsub.close()
        self.rxpub.close()
        self.rxsub.close()
        self.publisher.close()
        self.subscriber.close()
        self.txvipub.close()
        self.txvisub.close()
        self.rxvipub.close()
        self.rxvisub.close()
        self.vipub.close()
        self.visub.close()
        self.zmqContext.term()
        for task in asyncio.all_tasks():
            try:
                task.cancel()
            except:
                pass
        self.loop.stop()

    async def run(self):
        """Start running the controller process"""      
        try:
            logger.info(f'[{self.id}] is running ...')  
            # setup the running loop
            self.loop = asyncio.get_event_loop()
            self.zmqContext = Context.instance()
            # socketio client 
            self.sio = socketio.AsyncClient(
                    ssl_verify=False, 
                    reconnection=True) #logger=True, engineio_logger=True
            self.sio.register_namespace(self.priNameSpace(self, self.loop, namespace='/ctrl'))
            self.sio.register_namespace(self.pupNameSpace(self, self.loop, namespace='/'))
            # initial queuing configuration
            # self.packetInQueue = asyncio.Queue(0)
            await self.connect()
            # await self.statsWorker()
            # await self.nodeSTWorker()
        except KeyboardInterrupt:
            logger.info('ctrl+c')
        finally:
            # await self.terminatef()
            pass
            
    async def connect(self):   
        """Configure sockets to (data plane elements/visualizer) and run asynchronous workers"""
        # CONTROLLER-DATA_PLANE PUB/SUB CHANNEL
        #-----------------------------------------------------------------#
        #                     ___________________         _____________   #
        #                    |       Proxy       |        |            |  #
        #  Subscriber <----- |rxpub <-----> rxsub| <----  |            |  #
        #                    |                   |        | Data Plane |  #
        #  Publisher  -----> |txsub <-----> txpub| ---->  |            |  # 
        #                    |___________________|        |____________|  #     
        #-----------------------------------------------------------------#
        
        # CONTROLLER PUB/SUB URLs: Connect to the data link registered elements
        self.ctrlURL_pri = 'tcp://{}:{}'.format(self.inetAddress[0], str(self.inetAddress[1]))
        self.ctrlURL_sec = 'tcp://{}:{}'.format(self.inetAddress[0], str(self.inetAddress[1]+1))

        self.txpub = self.zmqContext.socket(zmq.XPUB)
        # self.txpub.setsockopt(zmq.SNDHWM, ct.CTRL_BUFF_SIZE)
        self.txpub.bind(self.ctrlURL_pri)
        self.txsub = self.zmqContext.socket(zmq.XSUB)
        # self.txsub.setsockopt(zmq.RCVHWM, ct.CTRL_BUFF_SIZE)
        self.txsub.bind("inproc://btxctrl")
        
        # Subscribe to the data plane
        self.rxsub = self.zmqContext.socket(zmq.XSUB)
        # self.rxsub.setsockopt(zmq.RCVHWM, ct.CTRL_BUFF_SIZE)
        self.rxsub.bind(self.ctrlURL_sec)
        # Publish the received packets from the data plane to the controller
        self.rxpub = self.zmqContext.socket(zmq.XPUB)
        # self.rxpub.setsockopt(zmq.SNDHWM, ct.CTRL_BUFF_SIZE)
        self.rxpub.bind("inproc://brxctrl")                
        
        # Bi-directional pub/sub proxy
        self.loop.create_task(self.tx_proxy()) # txsub <--> txpub
        self.loop.create_task(self.rx_proxy()) # rxsub <--> rxpub
        
        # CONTROLLER packet-in/packet-out publisher and subscriber
        # Publish the processed packets to txsub
        self.publisher = self.zmqContext.socket(zmq.PUB)
        # self.publisher.setsockopt(zmq.SNDHWM, ct.CTRL_BUFF_SIZE)
        # self.publisher.setsockopt(zmq.NOBLOCK)
        self.publisher.connect('inproc://btxctrl')
        # subscribe the received packets from rxpub
        self.subscriber = self.zmqContext.socket(zmq.SUB)
        self.subscriber.connect('inproc://brxctrl')
        # self.subscriber.setsockopt(zmq.SUBSCRIBE, b'')
        self.subscriber.subscribe('IN')
        # self.subscriber.setsockopt(zmq.RCVHWM, ct.CTRL_BUFF_SIZE)


        # CONTROLLER-VISUALIZATION PUB/SUB CHANNEL (for visualization purpose only)
        #----------------------------------------------------------------#
        #                _______________________         _____________   #
        #               |       Proxy           |        |            |  #
        #  visub <----- |rxvipub <-----> rxvisub| <----  |            |  #
        #               |                       |        | Data Plane |  #
        #  vipub -----> |txvisub <-----> txvipub| ---->  |            |  # 
        #               |_______________________|        |____________|  #     
        #----------------------------------------------------------------#
        
        # CONTROLLER PUB/SUB URLs: Connect to the data link registered elements
        self.ctrlURL_vipri = 'tcp://{}:{}'.format(self.inetAddress[0], str(self.inetAddress[1]+2))
        self.ctrlURL_visec = 'tcp://{}:{}'.format(self.inetAddress[0], str(self.inetAddress[1]+3))

        self.txvipub = self.zmqContext.socket(zmq.XPUB)
        # self.txvipub.setsockopt(zmq.SNDHWM, ct.CTRL_BUFF_SIZE)
        self.txvipub.bind(self.ctrlURL_vipri)
        self.txvisub = self.zmqContext.socket(zmq.XSUB)
        # self.txvisub.setsockopt(zmq.RCVHWM, ct.CTRL_BUFF_SIZE)
        self.txvisub.bind("inproc://vitxctrl")
        
        # Subscribe to the data plane
        self.rxvisub = self.zmqContext.socket(zmq.XSUB)
        # self.rxvisub.setsockopt(zmq.RCVHWM, ct.CTRL_BUFF_SIZE)
        self.rxvisub.bind(self.ctrlURL_visec)
        # Publish the received packets from the data plane to the controller
        self.rxvipub = self.zmqContext.socket(zmq.XPUB)
        # self.rxvipub.setsockopt(zmq.SNDHWM, ct.CTRL_BUFF_SIZE)
        self.rxvipub.bind("inproc://virxctrl")                
        
        # Bi-directional pub/sub proxy
        self.loop.create_task(self.txvi_proxy()) # txvisub <--> txvipub
        self.loop.create_task(self.rxvi_proxy()) # rxvisub <--> rxvipub
        
        # CONTROLLER packet-in/packet-out publisher and subscriber
        # Publish the processed packets to txvisub
        self.vipub = self.zmqContext.socket(zmq.PUB)
        # self.vipub.setsockopt(zmq.SNDHWM, ct.CTRL_BUFF_SIZE)
        # self.vipub.setsockopt(zmq.NOBLOCK)
        self.vipub.connect('inproc://vitxctrl')
        # visub the received packets from rxvipub
        self.visub = self.zmqContext.socket(zmq.SUB)
        # self.visub.setsockopt(zmq.RCVHWM, ct.CTRL_BUFF_SIZE)
        self.visub.connect('inproc://virxctrl')
        # self.visub.setsockopt(zmq.SUBSCRIBE, b'')
        self.visub.subscribe('ST')
        # self.visub.setsockopt(zmq.RCVHWM, ct.CTRL_BUFF_SIZE)

        ################################
        # self.webURL = 'tcp://127.0.0.1:4455/showgraph'
        # self.webpub = self.zmqContext.socket(zmq.PUB)
        # self.webpub.connect(self.webURL)
        # self.websub = self.zmqContext.socket(zmq.SUB)
        # self.websub.connect(self.webURL)

        # while not self.sio.connected:
        #     time.sleep(0.025)
        # await self.sio.wait()
        

        await self.sio.connect(ct.SIM_URL)
        await asyncio.sleep(2)
        # CORE ASYNCHRONOUS WORKERS: Async workers to handle network flow
        tasks = []
        # tasks.append(self.loop.run_in_executor(None, self.topoDiscWorker))
        tasks.append(self.loop.run_in_executor(None, self.packetInWorker))
        tasks.append(self.loop.run_in_executor(None, self.ctrlLearningWorker))
        tasks.append(self.loop.run_in_executor(None, self.nodeSTWorker))
        await asyncio.gather(*tasks)
        # asyncio.create_task(self.topoDiscWorker())
        # asyncio.create_task(self.packetInQueueWorker())
        # asyncio.create_task(self.packetInWorker())
        # asyncio.create_task(self.nodeSTWorker())
        # CUSTOMIZED ASYNCHRONOUS WORKER: A specific controller's async worker
        
        # asyncio.create_task(self.ctrlLearningWorker())
        # if hasattr(self, "ctrlLearningWorker"):
        #     asyncio.create_task(self.ctrlLearningWorker())
        # while True:
        #     print('connect********************************************')
        #     await asyncio.sleep(2)

    async def tx_proxy(self):
        poller = Poller()
        poller.register(self.txpub, zmq.POLLIN)
        poller.register(self.txsub, zmq.POLLIN)
        while True:
            events = await poller.poll()
            events = dict(events)
            if self.txpub in events:
                msg = await self.txpub.recv()
                await self.txsub.send(msg)
            elif self.txsub in events:
                msg = await self.txsub.recv()
                await self.txpub.send(msg)
            
    async def rx_proxy(self):
        poller = Poller()
        poller.register(self.rxsub, zmq.POLLIN)
        poller.register(self.rxpub, zmq.POLLIN)
        while True:
            events = await poller.poll()
            events = dict(events)
            if self.rxsub in events:
                msg = await self.rxsub.recv()
                await self.rxpub.send(msg)
            elif self.rxpub in events:
                msg = await self.rxpub.recv()
                await self.rxsub.send(msg)
    
    async def txvi_proxy(self):
        poller = Poller()
        poller.register(self.txvipub, zmq.POLLIN)
        poller.register(self.txvisub, zmq.POLLIN)
        while True:
            events = await poller.poll()
            events = dict(events)
            if self.txvipub in events:
                msg = await self.txvipub.recv()
                await self.txvisub.send(msg)
            elif self.txvisub in events:
                msg = await self.txvisub.recv()
                await self.txvipub.send(msg)
            
    async def rxvi_proxy(self):
        poller = Poller()
        poller.register(self.rxvisub, zmq.POLLIN)
        poller.register(self.rxvipub, zmq.POLLIN)
        while True:
            events = await poller.poll()
            events = dict(events)
            if self.rxvisub in events:
                msg = await self.rxvisub.recv()
                await self.rxvipub.send(msg)
            elif self.rxvipub in events:
                msg = await self.rxvipub.recv()
                await self.rxvisub.send(msg)

    class priNameSpace(socketio.AsyncClientNamespace):
        def __init__(self, node, loop, namespace=None):
            super().__init__(namespace)
            self.node = node
            self.loop = loop
            self.prevNeigs = []

        def on_connect(self):
            pass

        def on_disconnect(self):
            pass

        async def on_sr_response(self, data):
            """listen to broadcasting Sync network view among controllers 
            (updates the the network view accordingly to maintain global visibility over the network)
            """
            for node in data["nodes"]:
                if not self.node.getGraph().getNode(node['id']):
                    id = self.node.getGraph().addNode(node['id'])
                    self.node.getGraph().setupNode(id, node['net'], node['pos'], node['color'], node['intftype'], node['datatype'], node['batt'], node['port'], node['antrange'], node['active'])
            for edge in data["links"]:
                if not self.node.getGraph().getEdge(edge['source'], edge['target']):
                    self.node.getGraph().addEdge(edge['source'], edge['target'], edge['rssi'], edge['color'])

        async def on_sr_getgraph(self):
            """handle a request to sync the network view with the visualizer
            """        
            data = json_graph.node_link_data(self.node.networkGraph.getGraph())
            if data['nodes']:
                await self.emit('showgraph', data = {'nodes': data['nodes'], 'links': data['links']})

    class pupNameSpace(socketio.AsyncClientNamespace):
        def __init__(self, node, loop, namespace=None):
            super().__init__(namespace)
            self.node = node
            self.loop = loop

        def on_connect(self):
            pass

        def on_disconnect(self):
            pass  

    # async def getDistance(self, src_pos, dst_pos):
    #     """get distance between two nodes

    #     Args:
    #         src (tuple): source node position xyz
    #         dst (tuple): destination node position xyz

    #     Returns:
    #         int: distance
    #     """
    #     pos_src = [x for x in src_pos]
    #     pos_dst = [x for x in dst_pos]
    #     if len(pos_src) < 3:
    #         pos_src.append(0)
    #     if len(pos_dst) < 3:
    #         pos_dst.append(0)
    #     x = (int(pos_src[0]) - int(pos_dst[0])) ** 2
    #     y = (int(pos_src[1]) - int(pos_dst[1])) ** 2
    #     z = (int(pos_src[2]) - int(pos_dst[2])) ** 2
    #     dist = sqrt(x + y + z)
    #     return round(dist, 2)
    
    # async def isInRange(self, dist, dst_radio):
    #     """check if node is in the transmission range of another node

    #     Args:
    #         dist (float): distance between two nodes
    #         dst_txrange (int): distination node radio tx range

    #     Returns:
    #         bool: True/False
    #     """
    #     if dist > dst_radio['antRange']:
    #         return False
    #     return True

    # async def getRssi(self, src_radio, dst_radio, dist):
    #     """calculate the expected rssi using the configured propagation model

    #     Args:
    #         dst_radio (dict): source node radio params
    #         dst_radio (dict): destination node radio params

    #     Returns:
    #         int: rssi (dB)
    #     """
    #     return int(self.ppm.rssi(src_radio, dst_radio, dist))
    
    # async def getSnr(self, src, dst, noise_th):
    #     """calculate signal to noise ratio using rssi (dB)
    #     SNR = RSSI – RF background noise

    #     Args:
    #         dst (NodeView): destination node
    #         noise_th (int): background noise (dB)

    #     Returns:
    #         int: Signal-to-Noise Ratio (dB)
    #     """
    #     return await self.getRssi(self, src, dst) - noise_th

    # async def getAdjacencyList(self):
    #     """Get Node's graph adjacencies (neighboring nodes)"""
    #     ngs = {}
    #     for node in self.networkGraph.graph.nodes(data=True):
    #         src_ngs = []
    #         src_id = str(node[0])
    #         src_data = node[1]
    #         src_radio = IntfType.getParams(src_data['intftype'])
    #         for nd in self.networkGraph.graph.nodes(data=True):
    #             dst_id = str(nd[0])
    #             dst_data = nd[1]
    #             dst_radio = IntfType.getParams(dst_data['intftype'])
    #             dist = await self.getDistance(src_data['pos'], dst_data['pos'])
    #             if await self.isInRange(dist, dst_radio) and (dst_id != src_id):
    #                 rssi = mapRSSI(await self.getRssi(src_radio, dst_radio, dist)) 
    #                 addr = '{}.{}'.format(dst_id.split('.')[1], dst_id.split('.')[2])
    #                 src_ngs.append({'id':dst_id,
    #                                 'dist': dist,
    #                                 'addr':addr,
    #                                 'rssi':rssi,
    #                                 'port':dst_data['port']})
    #         if src_ngs:
    #             # ngs['/'+src_id] = src_ngs   
    #             ngs[src_id] = src_ngs   
    #     return ngs

    # def topoDiscWorker(self):
    #     """worker: discover neighbors for each node
    #     Assuming that the node can descover its neigbors in its transmission range
    #     """
    #     # import aiofiles
    #     # async with aiofiles.open('outputs/topo/topo.json') as f:
    #     #     data = json.loads(await f.read())
    #     #     self.webpub.send(str(data).encode('utf-8'))
    #     async def async_thread():
    #         while True:  
    #             ngs = await self.getAdjacencyList()
    #             for key in ngs:                
    #                 self.vipub.send(key.encode('utf-8')+b'NG'+str(json.dumps(ngs[key])).encode('utf-8'))
    #                 await asyncio.sleep(ct.BE_TTI/ct.MILLIS_IN_SECOND)
    #     asyncio.run(async_thread())

    def nodeSTWorker(self):
        """Subscribe worker to handle a publisher's request to change the color/status of the graph link/node
        """
        async def async_thread():
            while True:  
                stats = await self.visub.recv()
                stats = stats.decode('utf-8').split(";")
                stats.pop()
                for data in stats:
                    if data[len('ST'):len('ST')+len('LINK')] == 'LINK':
                        data = json.loads(data[len('ST')+len('LINK'):])
                        # asyncio.create_task(self.emitStats('linkcolor', data=data))
                        for link in data['links']:
                            self.networkGraph.updateEdgeColor(link['source'], link['target'], link['color'])
                    elif data[len('ST'):len('ST')+len('NODE')] == 'NODE':
                        data = json.loads(data[len('ST')+len('NODE'):])
                        self.networkGraph.updateNodeColor(data['id'], data['color'])
                    elif data[len('ST'):len('ST')+len('perf')] == 'perf':
                        data = json.loads(data[len('ST')+len('perf'):])
                        for key,val in data.items():
                            # print(f'key:{key} val:{val}')
                            if key != 'id':
                                if self.perf_buffer.get(key):
                                    self.perf_buffer[key].append(val)
                                else: 
                                    self.perf_buffer[key] = deque(maxlen=self.perf_buffer_size)
                                    self.perf_buffer[key].append(val)
                                self.networkGraph.getNode(data['id'])[key] = zero_division(sum(self.perf_buffer[key]),len(self.perf_buffer[key]))
                    elif data[len('ST'):len('ST')+len('ndtraffic')] == 'ndtraffic':
                        data = json.loads(data[len('ST')+len('ndtraffic'):])
                        self.networkGraph.getNode(data['id'])['lastupdate'] = data['lastupdate']
                        # self.networkGraph.getNode(data['id'])['active'] = bool(data['active'])
                        self.networkGraph.getNode(data['id'])['pos'] = data['pos']
                        self.networkGraph.getNode(data['id'])['batt'] = data['batt']
                        self.networkGraph.getNode(data['id'])['betti'] = data['betti']
                        self.networkGraph.getNode(data['id'])['rptti'] = data['rptti']
                        for key,val in data.items():
                            # print(f'key:{key} val:{val}')
                            if key not in ['id', 'active', 'lastupdate', 'pos', 'batt', 'betti', 'rptti', 
                                'txpackets_val_ts', 'txbeacons_val_ts', 'rxpackets_val_ts', 'drpackets_val_ts']:
                                if self.buffer.get(key):
                                    self.buffer[key].append(val)
                                else: 
                                    self.buffer[key] = deque(maxlen=self.buffer_size)
                                    self.buffer[key].append(val)
                                self.networkGraph.getNode(data['id'])[key] = zero_division(sum(self.buffer[key]),len(self.buffer[key]))
                        await self.emitStats('txpacket', {'val':data['txpackets_val_ts']})
                        await self.emitStats('txbeacon', {'val':data['txbeacons_val_ts']})
                        await self.emitStats('rxpacket', {'val':data['rxpackets_val_ts']})
                        await self.emitStats('droppacket', {'val':data['drpackets_val_ts']})
                        
                    state_mean, _, _, _ = self.networkGraph.getStateMean()
                    if state_mean.size != 0: 
                        data = {
                            'avgdelay':round(state_mean[1], 4), 
                            'avgthroughput':round(state_mean[2], 4), 
                            'avgengcons':round(state_mean[3], 4)
                            }
                        await self.emitStats('showstats', data)
        asyncio.run(async_thread())

    def packetInWorker(self):
        """Subscriber worker: listen to the data plane packets publisher (Controller-Proxy/Sink channel)
        """
        async def async_thread():
            while True:
                packet = await self.subscriber.recv()
                packet = packet[len('IN'):]
                if len(packet) >= ct.DFLT_HDR_LEN and len(packet) <= ct.MTU:
                    packetIn = Packet(packet)                
                    await self.rxHandler(packetIn)
                    await self.emitStats('rxpacketin')
                    logger.info(f'--------------->CONTROLLER receives packet type ({packetIn.getTypeName()}) | {packetIn.getSrc()} --> {packetIn.getDst()} - Next Hop ({packetIn.getNxh()})')  
        asyncio.run(async_thread())
            
    async def getNodeAddr(self, net:int, sinkId:str, sinkAddr:Addr, dst:Addr, path:list):
        """send a CONFIG packet to get a Node's address
        """    
        cp = ConfigPacket(net=net, src=sinkAddr, dst=dst, read=ConfigProperty.MY_ADDRESS, path=path)
        await self.sendNetworkPacket(cp, sinkId, sinkAddr)
    
    async def getNodeDist(self, net:int, sinkId:str, sinkAddr:Addr, dst:Addr, path:list):  
        """send a CONFIG packet to get a Node's tosink distance
        """    
        cp = ConfigPacket(net=net, src=sinkAddr, dst=dst, read=ConfigProperty.DIST, path=path)
        await self.sendNetworkPacket(cp, sinkId, sinkAddr)

    async def getNodeRule(self, net:int, sinkId:str, sinkAddr:Addr, dst:Addr, index, path:list):    
        """send a CONFIG packet to get an entry/rule of certain node's OF index
        """    
        cp = ConfigPacket(net=net, src=sinkAddr, dst=dst, read=ConfigProperty.GET_RULE, val=int(index).to_bytes(1, 'big'), path=path)
        await self.sendNetworkPacket(cp, sinkId, sinkAddr)    
        
    async def getNodeRules(self, net:int, sinkAddr:Addr, dst:Addr):
        """send a CONFIG packets to get all entries/rules of a node
        """
        #TODO add the logic
        pass
    
    async def setNodeRule(self, net:int, sinkId:str, sinkAddr:Addr, dst:Addr, newVal:Entry, path:list):      
        """send a CONFIG packets to set an OF entry/rule in a node
        """
        cp = ConfigPacket(net=net, src=sinkAddr, dst=dst, write=ConfigProperty.ADD_RULE, val=newVal.toByteArray(), path=path)
        await self.sendNetworkPacket(cp, sinkId, sinkAddr)
        
    async def removeNodeRule(self, net:int, sinkId:str, sinkAddr:Addr, dst:Addr, key:str, path:list):  
        """send a CONFIG packets to remove an OF entry/rule of certain node's OF index
        """
        cp = ConfigPacket(net=net, src=sinkAddr, dst=dst, write=ConfigProperty.REM_RULE, val=bytearray(key.encode('utf-8')), path=path)
        await self.sendNetworkPacket(cp, sinkId, sinkAddr)
                
    async def resetNode(self, net:int, sinkId:str, sinkAddr:Addr, dst:Addr, path:list):
        cp = ConfigPacket(net=net, src=sinkAddr, dst=dst, write=ConfigProperty.RESET, path=path)
        await self.sendNetworkPacket(cp, sinkId, sinkAddr)

    async def setNodeAddr(self, net:int, sinkId:str, sinkAddr:Addr, dst:Addr, newVal:Addr, path:list):
        """send a CONFIG packet to set a Node's address
        """    
        cp = ConfigPacket(net=net, src=sinkAddr, dst=dst, write=ConfigProperty.MY_ADDRESS, val=newVal.getArray(), path=path)
        await self.sendNetworkPacket(cp, sinkId, sinkAddr)
        
    async def setNodeDist(self, net:int, sinkId:str, sinkAddr:Addr, dst:Addr, newVal:int, path:list): 
        """send a CONFIG packet to set a Node's tosink distance
        """    
        cp = ConfigPacket(net=net, src=sinkAddr, dst=dst, write=ConfigProperty.DIST, val=int(newVal).to_bytes(1, 'big'), path=path)    
        await self.sendNetworkPacket(cp, sinkId, sinkAddr)
    
    async def setNodeAggr(self, net:int, sinkId:str, sinkAddr:Addr, dst:Addr, newVal:bool, path:list):
        """send a CONFIG packet to set a Node as an aggregation header
        """    
        cp = ConfigPacket(net=net, src=sinkAddr, dst=dst, write=ConfigProperty.IS_AGGR, val=int(newVal).to_bytes(1, 'big'), path=path)   
        await self.sendNetworkPacket(cp, sinkId, sinkAddr)
        
    async def setDRLAction(self, net:int, sinkId:str, sinkAddr:Addr, dst:Addr, newVal:bytearray, path:list):
        """send a CONFIG packet to set a Node as an aggregation header
        """    
        cp = ConfigPacket(net=net, src=sinkAddr, dst=dst, write=ConfigProperty.DRL_ACTION, val=newVal, path=path)   
        await self.sendNetworkPacket(cp, sinkId, sinkAddr)
        
    async def sendPath(self, net:int, sinkId:str, sinkAddr:Addr, src:Addr, dst:Addr, path:list):
        """send a OPEN_PATH packet to configure flow rules along a routing path
        """    
        op = OpenPathPacket(net=net, src=src, dst=dst, path=path)
        await self.sendNetworkPacket(op, sinkId, sinkAddr)
        
    async def addNodeRule(self, net:int, sinkId:str, sinkAddr:Addr, dst:Addr, entry:Entry):
        rp = ResponsePacket(net=net, src=sinkAddr, dst=dst, entry=entry)
        await self.sendNetworkPacket(rp, sinkId, sinkAddr)
            
    async def rxHandler(self, packet):
        """handle received packets from the data plane
        Args:
            packet (_type_): received packet
        """
        if packet.isssdwsnPacket():
            pt = packet.getType()
            if pt == ct.REPORT:
                await self.rxReport(ReportPacket(packet.toByteArray()))
            elif pt == ct.REQUEST:
                await self.rxRequest(RequestPacket(packet.toByteArray()))
            elif pt == ct.CONFIG:
                await self.rxConfig(ConfigPacket(packet.toByteArray()))
            elif pt == ct.REG_PROXY:
                await self.rxRegProxy(RegProxyPacket(packet.toByteArray()))
            elif pt == ct.AGGR:
                await self.rxAggr(AggrPacket(packet.toByteArray()))

    async def rxReport(self, packet:ReportPacket):
        # async def async_thread():  
        modified = await self.networkGraph.updateMap(packet)
        if modified:
            ## reactive update to modification in the network topology
            # data = json_graph.node_link_data(self.networkGraph.getGraph())
            pass
        # asyncio.run(async_thread())

    async def rxRequest(self, packet:RequestPacket):
        # async def async_thread():  
        p = await self.putInRequestCache(packet)
        if not(p is None):
            await self.manageRoutingRequest(packet, p)
        # asyncio.run(async_thread())

    async def rxConfig(self, packet:ConfigPacket):
        # async def async_thread():  
        #TODO
        # key = ""
        # if cp.getConfigId() == ConfigProperty.GET_RULE:
        #     key = ("{} {} {} {}").format(cp.getNet(), cp.getSrc(), cp.getConfigId, cp.getParams()[0])                 
        # else:
        #     key = ("{} {} {}").format(cp.getNet(), cp.getSrc(), cp.getConfigId)
        pass
        # asyncio.run(async_thread())

    async def rxRegProxy(self, packet:RegProxyPacket):
        # async def async_thread():  
        # self.sinkAddress = packet.getSrc() #TODO multiple sinks or use broker addr to multi-subscribers sinks  
        sinkid = str(packet.getNet())+'.'+packet.getSrc().__str__()
        if sinkid not in self.sinks:      
            self.sinks.append(sinkid)
        # asyncio.run(async_thread())

    async def rxAggr(self, packet:AggrPacket):
        # async def async_thread():
        aggrtype = packet.getAggrType()
        # await self.parseAggrPacket(packet)
        # '''
        aggrpackets = await self.getAggrPackets(packet)
        for p in aggrpackets:
            if aggrtype == ct.REPORT:
                # await self.rxReport(p)
                # logger.error(f"SINK|||||||src: {p.getSrc().__str__()} NG: {[{'addr':x['addr'].__str__(), 'color':getColorVal(x['color'])} for x in p.getNeighbors()]}")
                # print(f'here you check...\n')
                # print(' '.join(format(i, '08b') for i in p.toByteArray())) 
                modified = await self.networkGraph.updateMap(p)
                if modified:
                    ## reactive update to modification in the network topology
                    # data = json_graph.node_link_data(self.networkGraph.getGraph())
                    pass
            # if aggrtype == ct.DATA:
            #     # await self.rxData(p)
            #     recvTime = round(time.time(), 4) #- ((ct.TTL_MAX + 1 - packet.getTtl()) * ct.SLOW_NET)
            #     sendTime = p.getTS()
            #     data = {"id":'{}.{}'.format(p.getNet(), p.getSrc().__str__()),
            #             "delay":(recvTime - sendTime)*1000,
            #             "throughput":(len(p.toByteArray())*0.001*8)/((recvTime - sendTime)*2)#packet_size/RTT
            #         }
            #     self.stvipub.send(b'ST'+b'rxpacket'+str(json.dumps(data)).encode('utf-8'))
            #     asyncio.create_task(self.emitStats('rxpacket', data))
        # '''
        # asyncio.run(async_thread())

    # async def parseAggrPacket(self, packet:AggrPacket):
    #     graph = self.networkGraph
    #     payload = packet.getAggrPayload()
    #     aggrtype = packet.getAggrType()
    #     index = 0
    #     if aggrtype == ct.REPORT:
    #         while len(payload) > index:
    #             index += 1
    #             pl_size = payload[index]              
    #             src = Addr(payload[index:index+ct.AGGR_SRC_LEN])
    #             index += ct.AGGR_SRC_LEN
    #             pl_size -= ct.AGGR_SRC_LEN
    #             src_id = str(packet.getNet())+'.'+src.__str__()
    #             ts = payload[index:index+ct.AGGR_TS_LEN]
    #             index += ct.AGGR_TS_LEN
    #             pl_size -+ ct.AGGR_TS_LEN
    #             for _ in range(pl_size):
    #                 if payload[index] == ct.DST_INDEX:
    #                     graph.updateDistance(src_id, payload[index+1])
    #                     index += 1
    #                 if payload[index] == ct.BATT_INDEX:
    #                     batt = int.from_bytes(payload[index+1:index+1+ct.BATT_LEN], byteorder='big', signed=False)
    #                     graph.updateBattery(src, batt)
    #                     index += ct.BATT_LEN
    #                 if payload[index] == ct.POS_INDEX:
    #                     pass
    #                 if payload[index] == ct.INFO_INDEX:
    #                     pass
    #                 if payload[index] == ct.PORT_INDEX:
    #                     pass
    #                 if payload[index] == ct.NEIGH_INDEX:
    #                     pass

    async def getAggrPackets(self, packet:AggrPacket):
        graph = self.networkGraph
        payload = packet.getAggrPayload()
        aggrtype = packet.getAggrType()
        index = 0
        plindex = 0
        packets = []
        # print('received aggr payload:\n')
        # logger.error(' '.join(format(i, '08b') for i in payload)) 
        # print('\n')
        # print(f'index: {index}\n')
        while len(payload) > index:
            pl_size = payload[index]
            src = Addr(payload[index+ct.AGGR_SRC_INDEX:index+ct.AGGR_SRC_INDEX+ct.AGGR_SRC_LEN])
            src_id = str(packet.getNet())+'.'+src.__str__()
            p = Packet(net=packet.getNet(), src=src, dst=packet.getDst())#TODO
            p.data[ct.TS_INDEX:ct.TS_INDEX+ct.TS_LEN] = payload[index+ct.AGGR_TS_INDEX:index+ct.AGGR_TS_INDEX+ct.AGGR_TS_LEN]
            p.data[ct.TYP_INDEX] = aggrtype
            plindex += ct.AGGR_PLHDR_LEN
            # p.setPayload(payload[index+ct.AGGR_TS_INDEX+ct.AGGR_TS_LEN:index+pl_size+1])
            # construct packet payload
            if aggrtype == ct.REPORT:

                # print(f'index: {index}\n')
                # print(f'plindex: {plindex}\n')
                pl = bytearray()
                if (index+pl_size) > plindex:
                    if payload[plindex] != ct.DIST_INDEX:
                        pl += int.to_bytes(graph.getNode(src_id)['distance'], 1, 'big', signed=False)
                    else:
                        pl += int.to_bytes(payload[plindex+1], 1, 'big', signed=False)
                        plindex += 2
                        # print(f'index: {index}\n')
                        # print(f'plindex: {plindex}\n')
                else:
                    pl += int.to_bytes(graph.getNode(src_id)['distance'], 1, 'big', signed=False)
                if (index+pl_size) > plindex:
                    if payload[plindex] != ct.BATT_INDEX:
                        pl += int.to_bytes(graph.getNode(src_id)['batt'], ct.BATT_LEN, 'big', signed=False)
                    else:
                        pl += payload[plindex+1:plindex+1+ct.BATT_LEN]
                        plindex += ct.BATT_LEN+1
                        # print(f'index: {index}\n')
                        # print(f'plindex: {plindex}\n')
                else:
                    pl += int.to_bytes(graph.getNode(src_id)['batt'], ct.BATT_LEN, 'big', signed=False)
                if (index+pl_size) > plindex:
                    if payload[plindex] != ct.POS_INDEX:
                        # logger.error(f"x: {int(graph.getNode(src_id)['x'])} y: {int(graph.getNode(src_id)['y'])}")
                        pl += int.to_bytes(int(graph.getNode(src_id)['x']), 2, 'big', signed=False) + int.to_bytes(int(graph.getNode(src_id)['y']), 2, 'big', signed=False)
                    else:
                        pl += payload[plindex+1:plindex+1+ct.POS_LEN]
                        # logger.error(f"x: {int.from_bytes(payload[plindex:plindex+2], 'big', signed=False)} y: {int.from_bytes(payload[plindex+2:plindex+ct.POS_LEN], 'big', signed=False)}")
                        plindex += ct.POS_LEN+1
                        # print(f'index: {index}\n')
                        # print(f'plindex: {plindex}\n')
                else:
                    # logger.error(f"x: {int(graph.getNode(src_id)['x'])} y: {int(graph.getNode(src_id)['y'])}")
                    pl += int.to_bytes(int(graph.getNode(src_id)['x']), 2, 'big', signed=False) + int.to_bytes(int(graph.getNode(src_id)['y']), 2, 'big', signed=False)
                if (index+pl_size) > plindex:
                    if payload[plindex] != ct.INFO_INDEX:
                        pl += int.to_bytes(0, ct.INFO_LEN, 'big', signed=False)
                        datatype = setBitRange(int.from_bytes(pl[ct.INFO_INDEX:ct.INFO_INDEX+ct.INFO_LEN], 'big', signed=False), ct.TYP_BIT_INDEX, ct.TYP_BIT_LEN, SensorType.fromStringToInt(graph.getNode(src_id)['datatype']))       
                        pl[ct.INFO_INDEX:ct.INFO_INDEX+ct.INFO_LEN] = bitsToBytes('{0:b}'.format(datatype).zfill(8*ct.INFO_LEN))

                        intftype = setBitRange(int.from_bytes(pl[ct.INFO_INDEX:ct.INFO_INDEX+ct.INFO_LEN], 'big', signed=False), ct.INTF_BIT_INDEX, ct.INTF_BIT_LEN, IntfType.fromStringToInt(graph.getNode(src_id)['intftype']))       
                        pl[ct.INFO_INDEX:ct.INFO_INDEX+ct.INFO_LEN] = bitsToBytes('{0:b}'.format(intftype).zfill(8*ct.INFO_LEN))
                    else:
                        pl += payload[plindex+1:plindex+1+ct.INFO_LEN]
                        plindex += ct.INFO_LEN+1
                        # print(f'index: {index}\n')
                        # print(f'plindex: {plindex}\n')
                else:
                    pl += int.to_bytes(0, ct.INFO_LEN, 'big', signed=False)
                    datatype = setBitRange(int.from_bytes(pl[ct.INFO_INDEX:ct.INFO_INDEX+ct.INFO_LEN], 'big', signed=False), ct.TYP_BIT_INDEX, ct.TYP_BIT_LEN, SensorType.fromStringToInt(graph.getNode(src_id)['datatype']))       
                    pl[ct.INFO_INDEX:ct.INFO_INDEX+ct.INFO_LEN] = bitsToBytes('{0:b}'.format(datatype).zfill(8*ct.INFO_LEN))

                    intftype = setBitRange(int.from_bytes(pl[ct.INFO_INDEX:ct.INFO_INDEX+ct.INFO_LEN], 'big', signed=False), ct.INTF_BIT_INDEX, ct.INTF_BIT_LEN, IntfType.fromStringToInt(graph.getNode(src_id)['intftype']))       
                    pl[ct.INFO_INDEX:ct.INFO_INDEX+ct.INFO_LEN] = bitsToBytes('{0:b}'.format(intftype).zfill(8*ct.INFO_LEN))
                if (index+pl_size) > plindex:
                    if payload[plindex] != ct.PORT_INDEX:
                        pl += int.to_bytes(graph.getNode(src_id)['port']-ct.BASE_NODE_PORT, ct.PORT_LEN, 'big', signed=False)
                    else:
                        pl += payload[plindex+1:plindex+1+ct.PORT_LEN]
                        plindex += ct.PORT_LEN+1
                        # print(f'index: {index}\n')
                        # print(f'plindex: {plindex}\n')
                else:
                    pl += int.to_bytes(graph.getNode(src_id)['port']-ct.BASE_NODE_PORT, ct.PORT_LEN, 'big', signed=False)
                if (index+pl_size) > plindex:
                    if payload[plindex] != ct.NEIGH_INDEX:
                        edges = list(graph.getGraph().edges(nbunch=src_id, data=True, keys=True))
                        tmppl = bytearray()
                        for edge in edges:
                            tmppl += Addr(re.sub(r'^.*?.', '', edge[1])[1:]).getArray() + int.to_bytes(edge[3]['rssi'], 1, 'big', signed=False)
                            logger.error(f"Addr: {Addr(re.sub(r'^.*?.', '', edge[1])[1:]).__str__()} rssi: {edge[3]['rssi']}")
                        pl += int.to_bytes(len(edges), 1, 'big', signed=False) + tmppl
                    else:
                        ngscount = int(payload[plindex+1]/ct.NEIGH_LEN)
                        plindex += 1
                        ngsrssi = {}
                        counter = 0
                        tmppl = bytearray()
                        for i in range(ngscount):
                            ngsrssi[str(packet.getNet())+'.'+Addr(payload[plindex+i*ct.NEIGH_LEN+1]+payload[plindex+i*ct.NEIGH_LEN+2]).__str__()] = payload[plindex+i*ct.NEIGH_LEN+3]
                        # print(f'index: {index}\n')
                        # print(f'plindex: {plindex}\n')
                        edges = list(graph.getGraph().edges(nbunch=src_id, data=True, keys=True))
                        for edge in edges:
                            for i in range(ngscount):
                                if ngsrssi.get(edge[1]):
                                    tmppl += Addr(re.sub(r'^.*?.', '', edge[1])[1:]).getArray() + int.to_bytes(ngsrssi[edge[1]], 1, 'big', signed=False)
                                    plindex += ct.NEIGH_LEN
                                    ngsrssi.pop(edge[1])
                                    counter += 1
                                else:
                                    tmppl += Addr(re.sub(r'^.*?.', '', edge[1])[1:]).getArray() + int.to_bytes(edge[3]['rssi'], 1, 'big', signed=False)
                                    plindex += ct.NEIGH_LEN
                                    counter += 1
                        for key, val in ngsrssi.items():
                            tmppl += Addr(re.sub(r'^.*?.', '', key)[1:]).getArray() + int.to_bytes(val, 1, 'big', signed=False)
                            plindex += ct.NEIGH_LEN
                            counter += 1
                        pl += int.to_bytes(counter, 1, 'big', signed=False) + tmppl
                    p.setPayload(pl)
                    p = ReportPacket(p.toByteArray())
                else:
                    edges = list(graph.getGraph().edges(nbunch=src_id, data=True, keys=True))
                    tmppl = bytearray()
                    for edge in edges:
                        tmppl += Addr(re.sub(r'^.*?.', '', edge[1])[1:]).getArray() + int.to_bytes(edge[3]['rssi'], 1, 'big', signed=False)
                    pl += int.to_bytes(len(edges), 1, 'big', signed=False) + tmppl
                p.setPayload(pl)
                p = ReportPacket(p.toByteArray())
            if aggrtype == ct.DATA:
                p.setPayload(pl)
                p = DataPacket(p.toByteArray())
            # logger.error(f'src: {p.getSrc().__str__()}')
            # logger.error(f'TS: {p.getTS()}')
            # logger.error(f'packet src: {p.getSrc().__str__()} dst:{p.getDst().__str__()} timestamp:{p.getTS()}')
            index += pl_size+1
            plindex = index
            # print(f'index: {index}\n')
            # print(f'plindex: {plindex}\n')
            # print('constucted packet:\n')
            # logger.error(' '.join(format(i, '08b') for i in p.toByteArray())) 
            packets.append(p)
        return packets

    async def mogrify(self, topic, msg):
        """ json encode the message and prepend the topic """
        return topic + ' ' + json.dumps(msg)

    async def demogrify(self, topicmsg:str):
        """ Inverse of mogrify() """
        json0 = topicmsg.find('{')
        topic = topicmsg[0:json0].strip()
        msg = json.loads(topicmsg[json0:])
        return topic, msg 

    async def runCmd(self, cmd):
        proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout = asyncio.subprocess.PIPE,
        stderr = asyncio.subprocess.PIPE)
        
        stdout, stderr = await proc.communicate()
        return stdout, stderr
                
    async def putInRequestCache(self, rp:RequestPacket):        
        if rp.getTotal() == 1:
            #TODO add request stats to reqStats dict (e.g., No. of reqs, inter-arraival, etc.)
            return DataPacket(rp.getReqPayload())
        key = rp.getSrc().__str__() + "." + str(rp.id)        
        if self.requestCache.__contains__(key):
            #TODO add request stats to reqStats dict (e.g., No. of reqs, inter-arraival, etc.)
            p = self.requestCache.pop(key)
            return RequestPacket.mergeReqPackets(p, rp)
        else: self.requestCache[key] = rp
        return None

    async def pathRulesSetup(self, net:int, sinkId:str, sinkAddr:Addr, src:Addr, route:list):
        if len(route) > 1:
            await self.sendPath(net, sinkId, sinkAddr, src, route[0], route)
            route.reverse()
            await self.sendPath(net, sinkId, sinkAddr, src, route[0], route)  
    
    async def srRulesSetup(self, req:RequestPacket, data:DataPacket):
        """Source Route Rules Setup"""
        cost = 0
        route = []
        net = req.getNet()
        srcNode = data.getSrc().__str__()
        dstNode = data.getDst().__str__()
        curNode = req.getSrc().__str__()
        sinkNode = req.getDst().__str__()
        cur = str(net) + "." + curNode
        dst = str(net) + "." + dstNode
        sink = str(net) + "." + sinkNode
        
        graph = self.networkGraph
        rt = self.routingApp(graph.getGraph(), graph.getLastModification())    
               
        try:
            #Forward data from the controller to the destination (sink_to_dst route)
            _, std_route = rt.getRoute(sink, dst)
            if std_route:
                await self.pathRulesSetup(net=net, sinkId=sink, sinkAddr=req.getDst(), src=req.getDst(), route=std_route)
            
                data.setSrc(req.getDst())
                await self.sendNetworkPacket(data, sink, sinkAddr=req.getDst())
                
                # for i in range(len(std_route) -1):
                #     rule = (sinkNode, dstNode, std_route[i].__str__(), std_route[i+1].__str__())
                #     self.flowReqStats[rule].append({'reqtime': time.time(), 'setuptime': time.time() - ct.RL_IDLE/4}) 
                #     if not self.rule2id.get(rule):
                #         self.rule2id[rule] = len(self.rule2id)
                #     self.flowSetupSeq.append([time.time(), time.time() - ct.RL_IDLE/4, self.rule2id[rule]])

            #Flow setup rules from the sink to the request cur (sink_to_cur route)
            _, stc_route = rt.getRoute(sink, cur)
            if stc_route:
                await self.pathRulesSetup(net=net, sinkId=sink, sinkAddr=req.getDst(), src=req.getDst(), route=stc_route)
                
                # for i in range(len(stc_route) -1):
                #     rule = (sinkNode, curNode, stc_route[i].__str__(), stc_route[i+1].__str__())
                #     self.flowReqStats[rule].append({'reqtime': time.time(), 'setuptime': time.time() - ct.RL_IDLE/4}) 
                #     if not self.rule2id.get(rule):
                #         self.rule2id[rule] = len(self.rule2id)
                #     self.flowSetupSeq.append([time.time(), time.time() - ct.RL_IDLE/4, self.rule2id[rule]])

            #Flow setup rules from the request cur to the flow destination (cur_to_dst route)     
            cost, route = rt.getRoute(cur, dst)
            features = []
            if route:
                # print(f'routing path from {cur} to {dst}: {route}')
                await self.pathRulesSetup(net=net, sinkId=sink, sinkAddr=req.getDst(), src=req.getDst(), route=route)
                
                for i in range(len(route) -1):
                    rule = (srcNode, dstNode, route[i].__str__(), route[i+1].__str__())
                    self.flowReqStats[rule].append({'reqtime': time.time(), 'setuptime': time.time() - ct.RL_IDLE/4}) 
                    if not self.rule2id.get(rule):
                        self.rule2id[rule] = len(self.rule2id)
                    self.flowSetupSeq.append([time.time(), time.time() - ct.RL_IDLE/4, self.rule2id[rule]])
                    
                    # features.append([srcNode, dstNode, route[i].__str__(), route[i+1].__str__(), reqtime, setuptime])                     
                # import pandas as pd
                # import numpy as np
                # dataset = pd.DataFrame(features)
                # dataset.to_csv('outputs/flowsetupstats.csv', mode='a+', sep='\t', index=False)                
                # route = pd.DataFrame(np.concatenate((np.array([[srcNode, dstNode, curNode, sinkNode]]), np.array([route])), axis=1))
                # route.to_csv('outputs/routes.csv', mode='a+', sep='\t', index=False)
            
        except Exception as e:
            logger.warn(e)
            
    async def h2hRuleSetup(self, req:RequestPacket, data:DataPacket):
        """Hop-to-Hop Routing Rule Setup"""
        cost = 0
        route = []
        net = req.getNet()
        srcNode = data.getSrc().__str__()
        dstNode = data.getDst().__str__()
        curNode = req.getSrc().__str__()
        sinkNode = req.getDst().__str__()
        cur = str(net) + "." + curNode
        dst = str(net) + "." + dstNode
        sink = str(net) + "." + sinkNode
                
        #Routing
        sinkAddr = req.getDst()
        graph = self.networkGraph
        rt = self.routingApp(graph.getGraph(), graph.getLastModification())  
        
        try:
            cost, route = rt.getRoute(cur, dst)
            if route:
                for i in range(len(route) -1):
                    rule = (srcNode, dstNode, route[i].__str__(), route[i+1].__str__())
                    #Flow Requests Statistics                
                    # self.flowReqStats[key]['reqcount'] += 1
                    # self.flowReqStats[key]['reqinterarrivaltime'], 
                    # self.flowReqStats[key]['reqtime'] = getInterArrivalTime(
                    #     self.flowReqStats[key]['reqtime'], 
                    #     self.flowReqStats[key]['reqinterarrivaltime'])
                    self.flowReqStats[rule].append({'reqtime': time.time(), 'setuptime': time.time() - ct.RL_IDLE/4}) 
                    if not self.rule2id.get(rule):
                        self.rule2id[rule] = len(self.rule2id)
                    self.flowSetupSeq.append([time.time(), time.time() - ct.RL_IDLE/4, self.rule2id[rule]])
            
            # print(f'routing path from {cur} to {dst}: {route}')
        
            ruleFwd = None
            ruleBwd = None
            index = 0
            for node in route:   
                cost, toSinkRoute = rt.getRoute(src=sink, dst=(str(net) + "." + node.__str__()))
                # print('to sink route from {} to {}: {}'.format(sink, (str(net) + "." + node.__str__()), toSinkRoute))
                tmpIndex = 0
                for toSinkNode in toSinkRoute:
                    # print('toSinkNode', toSinkNode.__str__())
                    if toSinkNode.__str__() == node.__str__():
                        if node.__str__() != route[len(route) - 1].__str__():
                            ruleFwd = Entry()                            
                            ruleFwd.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1).setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST).setRhs(route[len(route) - 1].intValue()))
                            ruleFwd.addAction(ForwardUnicastAction(nxtHop=route[index + 1]))                               
                            await self.addNodeRule(net, sinkAddr, toSinkNode, ruleFwd)
                            # await self.setNodeRule(net, sinkAddr, toSinkNode, ruleFwd)
                        # ruleBwd = Entry()
                        # ruleBwd.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1).setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST).setRhs(toSinkRoute[0].intValue()))
                        # ruleBwd.addAction(ForwardUnicastAction(nxtHop=toSinkRoute[tmpIndex - 1]))
                        # await self.addNodeRule(net, sinkAddr, toSinkNode, ruleBwd)  
                        # if node.__str__() != route[0].__str__():                 
                        #     ruleBwd = Entry()
                        #     ruleBwd.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1).setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST).setRhs(route[0].intValue()))
                        #     ruleBwd.addAction(ForwardUnicastAction(nxtHop=route[index - 1]))
                        #     self.addNodeRule(net, sinkAddr, toSinkNode, ruleBwd)  
                            
                    else:
                        ruleFwd = Entry()
                        ruleFwd.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1).setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST).setRhs(toSinkRoute[len(toSinkRoute) - 1].intValue()))
                        ruleFwd.addAction(ForwardUnicastAction(nxtHop=toSinkRoute[tmpIndex + 1]))
                        await self.addNodeRule(net, sinkAddr, toSinkNode, ruleFwd)
                        # await self.setNodeRule(net, sinkAddr, toSinkNode, ruleFwd)
                        # if tmpIndex != 0:
                        #     ruleBwd = Entry()
                        #     ruleBwd.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1).setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST).setRhs(toSinkRoute[0].intValue()))
                        #     ruleBwd.addAction(ForwardUnicastAction(nxtHop=toSinkRoute[tmpIndex - 1]))
                        #     await self.addNodeRule(net, sinkAddr, toSinkNode, ruleBwd)  
                    tmpIndex += 1                    
                index += 1    
            data.setSrc(req.getDst())
            await self.sendNetworkPacket(data, sink, sinkAddr)            
        except Exception as e:
            logger.warn(e)
        
    async def sendNetworkPacket(self, packet, sinkId:str, sinkAddr:Addr):
        """Publish a PACKET_OUT to the data plan (controller-Proxy/Sink channel)
        """
        packet.setNxh(sinkAddr)
        logger.info(f'CONTROLLER-----> sends packet type ({packet.getTypeName()}) | {packet.getSrc()} --> {packet.getDst()} - Next Hop ({packet.getNxh()})')
        self.publisher.send(b'OUT'+packet.toByteArray()) 
        await asyncio.sleep(0.025)
        await self.emitStats('txpacketout')
        
    async def emitStats(self, fun:str, data=None):
        """Emits topology/graph data to the web application server through Socketio
        """
        try:
            if data is None:
                await self.sio.emit(fun)
            else: await self.sio.emit(fun, data)
        except exceptions.BadNamespaceError:
            logger.warn(f' Controller: is not connect to Socketio server...')
        
class CtrlDijkstra(Controller):
    """A controller with a Dijkstra routing implementation
    """
    def __init__(self, inetAddress, networkGraph):
        super().__init__(inetAddress=inetAddress, networkGraph=networkGraph)
        self.routingApp = Dijkstra
        # self.start()
    
    async def setupNetwork(self):
        pass
    
    async def manageRoutingRequest(self, req:RequestPacket, data:DataPacket):
        #TODO if path size exceed the open path payload limit
        await self.srRulesSetup(req, data)
        # await self.h2hRuleSetup(req, data)

class CtrlLinkPred(Controller):
    """ A controller that can dynamically assign wighted distance to the sink based on the network performance
    """
    def __init__(self, inetAddress: tuple = None, networkGraph: Graph = None):
        super().__init__(inetAddress, networkGraph)
        self.name = 'LinkPred-ctrl'
        self.routingApp = Dijkstra
        # self.start()
        
    async def setupNetwork(self):
        pass
    
    async def manageRoutingRequest(self, req:RequestPacket, data:DataPacket):
        #TODO if path size exceed the open path payload limit
        await self.srRulesSetup(req, data)
        # self.h2hRuleSetup(req, data)
        
    def ctrlLearningWorker(self):
        async def async_thread():
            # run the RL model after 30 seconds of initiating the network simulation (discovery phase)
            await asyncio.sleep(15)
            # run the blocking ML code in separate thread (make sync code as (awaitable) async code)
            await self.loop.run_in_executor(None, self.run_loop_in_thread, self)
        asyncio.run(async_thread())

    def run_loop_in_thread(self, ctrl):
        async def async_thread():     
            from ssdwsn.app.agent import A2C_Agent
            # model = SAC(env=self.Env(ctrl, 'SAC_CTRL_SSDWSN') , batch_size=1, samples_per_epoch=5, lr=1e-3, alpha=0.002, tau=0.1)            
            agent = A2C_Agent(env=self.Env(ctrl, 'SAC_CTRL_SSDWSN') , batch_size=1, samples_per_epoch=5)            
            await agent.run()
        asyncio.run(async_thread())   

    class Env:
        def __init__(self, ctrl, name:str = None):
            super().__init__()
            self.name = name
            self.ctrl = ctrl

            self.networkGraph = self.ctrl.networkGraph   
            self.rt = self.ctrl.routingApp(self.networkGraph.getGraph(), self.networkGraph.getLastModification())  
            self.scaler = StandardScaler()
            # self.loop = asyncio.get_event_loop()
            self.obs_cols = ['batt', 'delay', 'throughput', 'engcons', 'distance', 'denisty', 'txpackets', 'txbytes', 'rxpackets',
            'rxbytes', 'drpackets', 'alinks', 'flinks', 'battvar']
            self.action_cols = ['lweight']
            self.observation_space = np.zeros((1, len(self.obs_cols)))
            self.action_space = np.zeros((1, len(self.action_cols)))
            self.max_action_space = np.ones((1, len(self.action_cols)))
            self.min_action_space = np.ones((1, len(self.action_cols))) * -1
            self.best_reward = 0
            # self.observation_space = spaces.Box(low=np.array([-inf, -inf, -inf, -inf, -inf, -inf]), high=np.array([inf, inf, inf, inf, inf, inf]))           
            # self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1])) 
            # self.action_cols = ['renergy', 'distance', 'denisty', 'alinks']
            # self.action_cols = ['renergy', 'denisty', 'alinks']
            # self.action_cols = ['lweight']
            self.max_action = None   
            self.min_action = None
            self.max_obs = None
            self.min_obs = None
            #performance metrics
            self.reward_history = []
            self.delay_history = []
            self.throughput_history = []
            self.energycons_history = []
            self.energyvar_history = [] 
            self.txpackets_history = []
            self.txbytes_history = []
            self.rxpackets_history = []
            self.rxbytes_history = []
            self.drpackets_history = []  
            
        async def step(self, action, obs, nodes):
            # future = asyncio.run_coroutine_threadsafe(self.runAction(action), self.loop)
            # return future.result()
            return await self.runAction(action, obs, nodes)
        
        def getObs(self, nds=None):
            """Get current network state"""

            '''['batt', 'delay', 'throughput', 'engcons', 'distance', 'denisty', 'txpackets', 'txbytes', 'rxpackets', \
            'rxbytes', 'drpackets', 'alinks', 'flinks', 'x', 'y', 'z', 'intftypeval', 'datatypeval']'''
            state, nodes = self.networkGraph.getState(nds)
            # states = self.scaler.fit_transform(states)
            self.observation_space = state
            # print(f'obs shape: {self.observation_space.shape}')
            # print(states)
            # self.max_obs = states.max(axis=1)
            # self.min_obs = states.min(axis=1)
            # self.action_space = np.hstack((states[:,0].reshape(-1,1), states[:,4].reshape(-1,1), states[:,5].reshape(-1,1), states[:,6].reshape(-1,1)))
            # self.max_action = self.action_space.high
            # self.min_action = self.action_space.low
            self.max_action = np.ones((1, 1))
            # self.min_action = np.zeros((1, 1))
            self.min_action = np.ones((1, 1)) * -1
            self.max_obs = self.observation_space.max(axis=0, keepdims=True)
            self.min_obs = self.observation_space.min(axis=0, keepdims=True)
            # states = self.scaler.fit_transform(states)
            # return states.mean(axis=0, keepdims=True)
            self.observation_space = np.column_stack((self.observation_space[:,0:13], np.sqrt(np.square(self.observation_space[:,0] - np.mean(self.observation_space[:,0])))))
            return self.observation_space, nodes

        def getReward(self, obs, prv_obs):
            # R = (avg_Throughput (kbit/sec)) / ((avg_delay (msec) /1000) * avg_energyconsumption (mC/sec))
            # scale the values between 0 and 10 then map the value to a value between [-1, 0]
            # x = (x * 10) / 100 then x = ((x - (0)) / (10 - (0))) - 1 # muliply the nominator by 2 if whant to make it between [-1,1]
            throughput = obs[:,2]
            prv_throughput = prv_obs[:,2]
            delay =  obs[:,1]
            prv_delay =  prv_obs[:,1]
            engcons =  obs[:,3]
            prv_engcons =  prv_obs[:,3]
            txpackets =  obs[:,6]
            prv_txpackets =  prv_obs[:,6]
            txbytes =  obs[:,7]
            prv_txbytes =  prv_obs[:,7]
            rxpackets =  obs[:,8]
            prv_rxpackets =  prv_obs[:,8]
            rxbytes =  obs[:,9]
            prv_rxbytes =  prv_obs[:,9]
            drpackets =  obs[:,10]
            prv_drpackets =  prv_obs[:,10]
            battvar =  obs[:,13]
            prv_battvar =  prv_obs[:,13]
            # R = ((throughput-prv_throughput)+(prv_delay-delay)+(prv_engcons-engcons))
            # R = ((throughput > prv_throughput).astype(float) + (delay < prv_delay).astype(float) + (engcons < prv_engcons).astype(float)**2)
            # R = np.where(throughput > prv_throughput, 1, -100).astype(float) + np.where(delay < prv_delay, 1, -100).astype(float) + np.where(engcons < prv_engcons, 1, -100).astype(float)
            R = np.where(throughput > prv_throughput, 1, 0).astype(float) + np.where(delay < prv_delay, 1, 0).astype(float) + np.where(engcons < prv_engcons, 1, 0).astype(float) + np.where(battvar < prv_battvar, 1, 0).astype(float) + np.where(drpackets < prv_drpackets, 1, 0).astype(float)
            # R = np.where(engcons < prv_engcons, 1, -zero_division(engcons, prv_engcons)).astype(float) + np.where(battstd < prv_battstd, 1, -zero_division(battstd, prv_battstd)).astype(float) + np.where(drpackets < prv_drpackets, 1, -zero_division(drpackets, prv_drpackets)).astype(float)
            # R = ((1/(delay+engcons+battvar+drpackets)) - 1) / throughput
            # R = throughput/ (delay+engcons+battvar+drpackets)
            # R = ((1/((delay-prv_delay)+(engcons-prv_engcons)+(battvar-prv_battvar)+(drpackets-prv_drpackets)+1)) - 1)/((throughput-prv_throughput)+1)
            # print(f'Reward: \n {R}')
            # R = (throughput/ self.max_obs[:,2])+(self.min_obs[:,1] / delay )+ (self.min_obs[:,3] / engcons)
            # R = (delay+engcons+battvar+drpackets)/throughput
            # R = np.sum((throughput/ self.max_obs[:,2])+(self.min_obs[:,1] / delay )+ (self.min_obs[:,3] / engcons)) 
            # R = np.sum((throughput/ 0.200)+(30.0 / delay )+ (5.0 / engcons)) / 3
            # R = np.sum((throughput/ self.max_obs[:,2])+(self.min_obs[:,1] / delay )+ (self.min_obs[:,3] / engcons)) - 3
            # R = np.divide(throughput, np.multiply(delay, engcons)) # [-1, 0] zero is the best reward
            # R = np.divide(throughput/ np.amax(obs[:,2]), np.multiply(delay / np.amax(obs[:,1]), engcons / np.amax(obs[:,3]))) # [-1, 0] zero is the best reward
            # R[R == inf] = -1
            return np.nan_to_num(R, nan=0).reshape((-1, 1)) #TODO divide by zero error
            # return sqrt((obs[1] / (obs[0]))**2) #TODO divide by zero error

        async def runAction(self, action, obs, obs_nds):
            action = list(action.flatten())
            obs_nds = list(obs_nds.flatten())
            act_nds = dict(zip(obs_nds, action))
            # max_obs = self.max_obs.flatten()
            # min_obs = self.min_obs.flatten()

            #interpret normalization of action between [x,y] = [-1,1] to its original value between [a,b] = [min_x,max_x]
            #value_ab = ((value_xy - x) / (y - x)) * (b - a) + a.            
            #interpret the value  
            # renergy = int(((action[0] - (-1)) / (1 - (-1))) * (max_action[0] - min_action[0]) + min_action[0]) #
            #interpret the value
            # distance = int(((action[1] - (-1)) / (1 - (-1))) * (max_action[1] - min_action[1]) + min_action[1]) #
            #interpret the value
            # denisty = int(((action[2] - (-1)) / (1 - (-1))) * (max_action[2] - min_action[2]) + min_action[2]) #
            #interpret the value
            # alinks = int(((action[3] - (-1)) / (1 - (-1))) * (max_action[3] - min_action[3]) + min_action[3]) #
            # find nodes with values >= action values
            _, nodes = self.networkGraph.getState()
            nodes = sorted(set(nodes))
            # print(f'action: {action}')
            # print(f'obs_nds: {obs_nds}')
            if action:
                idx = 0
                for node in obs_nds:#TODO install rule based on the action selected (select the best route to sink based on accumulated weights)
                    for sink in self.ctrl.sinks:
                        routes = nx.all_simple_paths(self.networkGraph.getGraph(), source=node, target=sink, cutoff=obs[idx,4]+2)
                        routes = list(routes)
                        if routes:
                            #value_ab = ((value_xy - x) / (y - x)) * (b - a) + a
                            # weights = [sum([(((val - min(rtv)) / (max(rtv) - min(rtv))) * (1 - (-1)) + (-1))  for val in rtv])/len(rtv) for rtv in 
                            #                     [[act_nds[nd] if act_nds.get(nd) else 0 for nd in rt] for rt in routes]]
                            weights = [sum([val  for val in rtv])/len(rtv) for rtv in 
                                                [[act_nds[nd] if act_nds.get(nd) else 0 for nd in rt] for rt in routes]]
                            # print(f'node: {node} routes: {routes}')
                            # print(f'node: {node} weights: {weights}')
                            # print(f'node: {node} min_route: {routes[weights.index(min(weights))]}')
                            # for route in routes:
                            #     print(f'node: {node} route: {route}')
                            # print(f'action nodes: {act_nds}')
                            # for route in all_routes:
                            nd = self.networkGraph.getNode(node)
                            # weight = ((nd.get('batt')/renergy) + (nd.get('distance')/distance)/ + (nd.get('denisty')/denisty) + (nd.get('alinks')/alinks))/self.action_space.shape[1]
                            # new_distance = int(weight*nd.get('distance') + nd.get('distance'))
                            # print(f'weight: {weight}')
                            # print(f'distance: {nd.get("distance")}')
                            # print(f'new-distance: {new_distance}')
                            sel_route = routes[weights.index(min(weights))]
                            sinkAddr=Addr(re.sub(r'^.*?.', '', sink)[1:])
                            i = len(sel_route) - 2
                            for _ in sel_route[:-1]:
                                _, route = self.rt.getRoute(node, sink)
                                dst = Addr(re.sub(r'^.*?.', '', sel_route[i])[1:])

                                if i == 0:
                                    src = Addr(re.sub(r'^.*?.', '', sel_route[i])[1:])
                                else:
                                    src = Addr(re.sub(r'^.*?.', '', sel_route[i-1])[1:])

                                entry = Entry()
                                entry.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1)
                                    .setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST)
                                    .setRhs(sinkAddr.intValue()))
                                entry.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1)
                                    .setLhsOperandType(ct.PACKET).setLhs(ct.SRC_INDEX).setRhsOperandType(ct.CONST)
                                    .setRhs(src.intValue()))
                                # entry.addWindow(Window.fromString("P.TYP == 2"))
                                entry.addAction(ForwardUnicastAction(nxtHop=Addr(re.sub(r'^.*?.', '', sel_route[i+1])[1:])))
                                entry.getStats().setTtl(int(time.time()))
                                entry.getStats().setPermanent() #added                
                                await self.ctrl.setNodeRule(net=int(sink.split('.')[0]), sinkId=sink, sinkAddr=sinkAddr, dst=dst, newVal=entry, path=route)
                                i -= 1
                    idx += 1
                # get the state (obs) of the network after 10sec from applying the action
                await asyncio.sleep(5)
            await asyncio.sleep(0.01)
            next_obs, _ = self.getObs(obs_nds)
            reward = self.getReward(next_obs, obs)
            # self.ctrl.logger.warn(f'next obs: {next_obs}')
            # self.ctrl.logger.warn(f'reward: {reward}')
            done = np.zeros((next_obs.shape[0],1), dtype=bool) #TODO change the logic when to set done to True (since it is a continouse process target optimization are always changing as per the network progress and resource drained)           
            info = np.empty((next_obs.shape[0],1), dtype=str)
            return next_obs, reward, done, info  
            # return next_obs, reward, info  

class DutySchedulerCtrl(Controller):
    """ A Controller that can dynamically choose the appropriate cluster head based on the network performance
    """
    def __init__(self, inetAddress: tuple = None, networkGraph: Graph = None):
        super().__init__(inetAddress, networkGraph)
        self.name = 'DutyScheduler-ctrl'
        self.routingApp = Dijkstra        
        # self.start()
         
    async def setupNetwork(self):
        pass
    
    async def manageRoutingRequest(self, req:RequestPacket, data:DataPacket):
        #TODO if path size exceed the open path payload limit
        await self.srRulesSetup(req, data)
        # self.h2hRuleSetup(req, data)
    
    def ctrlLearningWorker(self):
        async def async_thread():
            # run the RL model after 60 seconds of initiating the network simulation (discovery phase)
            await asyncio.sleep(180)
            # run the blocking ML code in separate thread (make sync code as (awaitable) async code)
            
            from ssdwsn.app.agent import SAC_Agent, A2C_Agent, SAC_Agent_, SAC_Agent__, TD3_Agent, PPO_Agent, PPO_Agent1, PPO_Agent2, PPO_Agent3, PPO_Agent4, PPO_Agent5, REINFORCE_Agent, PPO_GAE_Agent, PPO_MultiAgent
            # agent = SAC(env=self.Env(ctrl, 'SAC_CTRL_SSDWSN') , batch_size=2, samples_per_epoch=4, lr=1e-3, alpha=0.002, tau=0.1)            
            # agent = A2C(env=self.Env(ctrl, 'SAC_CTRL_SSDWSN') , batch_size=2, samples_per_epoch=4)            
            # await model.run()
            # action_dim = int((ct.MAX_RP_TTI - ct.MIN_RP_TTI)/ct.MAX_DELAY)
            agent = PPO_Agent4(ctrl=self, batch_size=50, samples_per_epoch=1)
            await agent.run()
            # await asyncio.gather(*[run_agent(env, node) for node in env.nodes])            
            # agent = A2C_Agent(env=self.Env(ctrl, 'SAC_CTRL_SSDWSN') , batch_size=20, samples_per_epoch=2)
            # agent = TD3_Agent(env=self.Env(ctrl, 'TD3_CTRL_SSDWSN') , batch_size=2, samples_per_epoch=4)
            
        asyncio.run(async_thread())

class StatePredCtrl(Controller):
    """ A Controller that can dynamically choose the appropriate cluster head based on the network performance
    """
    def __init__(self, inetAddress: tuple = None, networkGraph: Graph = None):
        super().__init__(inetAddress, networkGraph)
        self.name = 'StatePred-ctrl'
        self.routingApp = Dijkstra        
        # self.start()
        
    async def setupNetwork(self):
        pass
    
    async def manageRoutingRequest(self, req:RequestPacket, data:DataPacket):
        #TODO if path size exceed the open path payload limit
        await self.srRulesSetup(req, data)
        # self.h2hRuleSetup(req, data)
    
    def ctrlLearningWorker(self):
        async def async_thread():
            # run the RL model after 60 seconds of initiating the network simulation (discovery phase)
            await asyncio.sleep(120)
            # run the blocking ML code in separate thread (make sync code as (awaitable) async code)
            
            from ssdwsn.app.agent import SAC_Agent, A2C_Agent, SAC_Agent_, SAC_Agent__, TD3_Agent, PPO_Agent, PPO_Agent1, PPO_Agent2, PPO_Agent3, PPO_Agent4, PPO_Agent5, REINFORCE_Agent, PPO_GAE_Agent, PPO_MultiAgent
            # agent = SAC(env=self.Env(ctrl, 'SAC_CTRL_SSDWSN') , batch_size=2, samples_per_epoch=4, lr=1e-3, alpha=0.002, tau=0.1)            
            # agent = A2C(env=self.Env(ctrl, 'SAC_CTRL_SSDWSN') , batch_size=2, samples_per_epoch=4)            
            # await model.run()
            agent = PPO_Agent5(ctrl=self, batch_size=50, samples_per_epoch=2, obs_time=20)
            await agent.run()
            # await asyncio.gather(*[run_agent(env, node) for node in env.nodes])            
            # agent = A2C_Agent(env=self.Env(ctrl, 'SAC_CTRL_SSDWSN') , batch_size=20, samples_per_epoch=2)
            # agent = TD3_Agent(env=self.Env(ctrl, 'TD3_CTRL_SSDWSN') , batch_size=2, samples_per_epoch=4)
            
        asyncio.run(async_thread())
                    
'''
    class Env(gym.Env):
        """ Environemnt:
        Observations (states): {avg_delay, avg_throughput, avg_energyconsumption}
        Actions: is a multivariate normal distribution of a k-dimensional random vector X = ( X_1 , … , X_k )^T --> X ~ N(mu, seqma) 
        closest action values to mu are selected and then interpreted to the real environement action values
        """
        def __init__(self, ctrl, name:str = None):
            super().__init__()
            self.name = name
            self.ctrl = ctrl
            # self.loop = asyncio.get_event_loop()
            # self.observation_space = np.zeros((1, 3))
            # self.action_space = np.zeros((1, 3))
            # self.max_action = np.ones((1,3))
            self.observation_space = spaces.Box(low=np.array([-inf, -inf, -inf, -inf]), high=np.array([inf, inf, inf, inf]))           
            # self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1])) 
            self.action_space = spaces.Box(low=np.array([0]), high=np.array([1])) 
            self.max_action = self.action_space.high        

            self.action_names = ['renergy', 'distance', 'denisty'] 
            self.action_max_values = [2, 2, 2]
            self.action_min_values = [1, 1, 1]
            
        async def step(self, obs, obs_nodes, action):                        
            return await self.runAction(obs, obs_nodes, action)
        
        def getObs(self):
            """Get current environment state
            State = {avg_delay, avg_throughput, avg_energyconsumption} of the whole network
            """
            networkGraph = self.ctrl.networkGraph
            # update environemnt max and min real action values

            # get environement current state
            obs, obs_nodes = networkGraph.getState()
            # state = networkGraph.getState()
            # obs = np.array(state, ndmin=1)
            return obs, obs_nodes

        def getReward(self, obs):
            # R = avg_Throughput/(avg_delay * avg_energyconsumption)
            # obs = obs.flatten()
            R = np.divide(obs[:,2], np.multiply(obs[:,1], obs[:,3]))
            R[R == inf] = 0
            return np.nan_to_num(R, nan=0).reshape((-1, 1)) #TODO divide by zero error
            # return sqrt((obs[1] / (obs[0]))**2) #TODO divide by zero error

        async def runAction(self, obs, obs_nodes, action):
            # print(f'ops_nodes\n:{obs_nodes}')
            # print(f'action\n:{action}')
            idx = 0
            for ac in action.flatten():
                if ac >= 0.6:
                    # configure new selected aggregates
                    if obs_nodes[idx] not in self.ctrl.aggregates:
                        for sink in self.ctrl.sinks:
                            await self.ctrl.setNodeAggr(net=int(sink.split('.')[0]), sinkId=sink, sinkAddr=Addr(sink.split('.')[1]+'.'+sink.split('.')[2]), dst=Addr(obs_nodes[idx].split('.')[1]+'.'+obs_nodes[idx].split('.')[2]), newVal=True)
                        self.ctrl.aggregates.append(obs_nodes[idx])
                else:
                    # remove previously selected aggregates                                      
                    if obs_nodes[idx] in self.ctrl.aggregates:
                        for sink in self.ctrl.sinks:
                            await self.ctrl.setNodeAggr(net=int(sink.split('.')[0]), sinkId=sink, sinkAddr=Addr(sink.split('.')[1]+'.'+sink.split('.')[2]), dst=Addr(obs_nodes[idx].split('.')[1]+'.'+obs_nodes[idx].split('.')[2]), newVal=False)
                idx += 1
            # get the state (obs) of the network after 10sec from applying the action
            await asyncio.sleep(5)
            next_obs, _ = self.getObs() # TODO get next_obs of the obs_nodes (to make sure that the next_obs is right data of obs_nodes)
                    
            reward = self.getReward(next_obs)
            done = np.zeros((next_obs.shape[0],1), dtype=bool) #TODO change the logic when to set done to True (since it is a continouse process target optimization are always changing as per the network progress and resource drained)           
            info = np.empty((next_obs.shape[0],1), dtype=str)
            print(f'Env next_obs: {next_obs}')
            print(f'Env rewards: {reward}')
            return next_obs, reward, done, info           
'''                
class CtrlNextHopPred(Controller):
    
    def __init__(self, inetAddress, networkGraph):
        super().__init__(inetAddress, networkGraph)
        self.routingApp = Dijkstra
        # self.start()
    
    async def setupNetwork(self):
        pass
    
    async def manageRoutingRequest(self, req:RequestPacket, data:DataPacket):
        #TODO if path size exceed the open path payload limit
        await self.srRulesSetup(req, data)
        # await self.h2hRuleSetup(req, data)
    
    '''
    async def ctrlLearningWorker(self):
        import numpy as np
        import pandas as pd
        from ssdwsn.app.smartSDWSN import LearningModel
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.model_selection import train_test_split
                
        while True:
            # print('CtrlNextHopPred')
            networkGraph = self.networkGraph          
            graph = networkGraph.getGraph()
            nodes = list(set(graph.nodes(data=False)))
            nodes.sort()
            rt = self.routingApp(graph, networkGraph.getLastModification())    
            classes = list(set([node.split('.',1)[1] for node in nodes]))                               
            features = []
            targets = []
            if not list(graph.edges(data=False)):
                continue
            try:
                for src in nodes:
                    for dst in nodes:
                        cost, route = rt.getRoute(src, dst)
                        if route:
                            for i in range(len(route)-1):
                                srcNode = graph.nodes.get(src)
                                dstNode = graph.nodes.get(dst)
                                curNode = graph.nodes.get(str(srcNode['net'])+'.'+route[i].__str__())
                                datatype = srcNode['datatype']
                                srcaddr = route[0].intValue()
                                dstaddr = route[-1].intValue()
                                curnode = route[i].intValue()
                                srcport = srcNode['port']
                                dstport = dstNode['port']
                                curport = curNode['port']
                                curposx = curNode['pos'][0]
                                curposy = curNode['pos'][1]
                                curposz = 0
                                features.append([datatype, srcaddr, dstaddr, curnode, srcport, dstport, curport, curposx, curposy, curposz])
                                targets.append([route[i+1].__str__()])  
                cat_cols = ['datatype']
                con_cols = ['srcaddr', 'dstaddr', 'curaddr', 'srcport', 'dstport', 'curport', 'curposx', 'curposy', 'curposz']
                dataset = pd.concat([pd.DataFrame(features, columns=cat_cols+con_cols), pd.DataFrame(targets, columns=['route'])], axis=1)
                dataset.to_csv('outputs/dataset.csv', mode='w+', sep='\t', index=False)
                for cat in cat_cols:
                    dataset[cat] = dataset[cat].astype('category')
                cats = np.stack([dataset[cat].cat.codes.values for cat in cat_cols], axis=1)
                conts = np.stack([dataset[cont].values for cont in con_cols], axis=1)
                cat_szs = [len(dataset[col].cat.categories) for col in cat_cols]
                emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
                mbzr = MultiLabelBinarizer(classes=classes)    
                targets = mbzr.fit_transform(targets)      
                targets = np.array(targets)  
                np_features = np.concatenate((cats, conts), axis=1)
                x_train, x_test, y_train, y_test = train_test_split(np_features, targets, test_size = 0.2, random_state = 42)
                cat_train = x_train[:, : cats.shape[1]]
                cat_test = x_test[:, : cats.shape[1]]
                con_train = x_train[:, cats.shape[1]:conts.shape[1]+1]
                con_test = x_test[:, cats.shape[1]:conts.shape[1]+1]
                tr_dataset = pd.concat([pd.DataFrame(cat_train, columns=cat_cols), pd.DataFrame(con_train, columns=con_cols), pd.DataFrame(y_train, columns=classes)], axis=1)
                tr_dataset.to_csv('outputs/traindataset.csv', mode='w+', sep='\t', index=False)
                te_dataset = pd.concat([pd.DataFrame(cat_test, columns=cat_cols), pd.DataFrame(con_test, columns=con_cols), pd.DataFrame(y_test, columns=classes)], axis=1)
                te_dataset.to_csv('outputs/testdataset.csv', mode='w+', sep='\t', index=False)
                # print(f'`cat_train:{cat_train}')
                # print(f'cat_test:{cat_test}')
                # print(f'con_train:{con_train}')
                # print(f'con_test:{con_test}')
                # print(f'y_train:{y_train}')
                # print(f'y_test:{y_test}')
            except Exception as e:
                logger.error(e)
            rm = LearningModel(cat_train, cat_test, con_train, con_test, y_train, y_test, emb_szs, 50, 0.4)           
            await rm.nextHopPredModel(layers=[200,100], classes=classes)
    '''
class CtrlFullRoutePred(Controller):
    
    def __init__(self, inetAddress, networkGraph):
        super().__init__(inetAddress=inetAddress, networkGraph=networkGraph)
        self.routingApp = Dijkstra
        # self.start()
    
    async def setupNetwork(self):
        pass
    
    async def manageRoutingRequest(self, req:RequestPacket, data:DataPacket):
        #TODO if path size exceed the open path payload limit
        await self.srRulesSetup(req, data)
        # await self.h2hRuleSetup(req, data)
    
    '''
    async def ctrlLearningWorker(self):
        import numpy as np
        import pandas as pd
        from ssdwsn.app.smartSDWSN import LearningModel
        from sklearn.model_selection import train_test_split
                
        while True:
            # print('CtrlFullRoutePred')
            networkGraph = self.networkGraph          
            graph = networkGraph.getGraph()
            nodes = list(set(graph.nodes(data=False)))
            nodes.sort()
            rt = self.routingApp(graph, networkGraph.getLastModification())            
            classes = {nodes[i].split('.',1)[1]:i  for i in range(len(nodes))} 
            classes['nan'] = -1
            features = []
            targets = []
            if not list(graph.edges(data=False)):
                continue
            try:
                for src in nodes:
                    for dst in nodes:
                        cost, route = rt.getRoute(src, dst)
                        if route:
                            for i in range(len(route)-1):
                                srcNode = graph.nodes.get(src)
                                dstNode = graph.nodes.get(dst)
                                curNode = graph.nodes.get(str(srcNode['net'])+'.'+route[i].__str__())
                                datatype = srcNode['datatype']
                                srcaddr = route[0].intValue()
                                dstaddr = route[-1].intValue()
                                curnode = route[i].intValue()
                                srcport = srcNode['port']
                                dstport = dstNode['port']
                                curport = curNode['port']
                                curposx = curNode['pos'][0]
                                curposy = curNode['pos'][1]
                                curposz = 0
                                features.append([datatype, srcaddr, dstaddr, curnode, srcport, dstport, curport, curposx, curposy, curposz])                                
                                targets.append([classes[r.__str__()] for r in route[i+1:]]+([-1]*(len(nodes)-len(route[i+1:]))))
                cat_cols = ['datatype']
                con_cols = ['srcaddr', 'dstaddr', 'curaddr', 'srcport', 'dstport', 'curport', 'curposx', 'curposy', 'curposz']
                dataset = pd.concat([pd.DataFrame(features, columns=cat_cols+con_cols), pd.DataFrame(targets)], axis=1)
                dataset.to_csv('outputs/dataset.csv', mode='w+', sep='\t', index=False)
                for cat in cat_cols:
                    dataset[cat] = dataset[cat].astype('category')
                cats = np.stack([dataset[cat].cat.codes.values for cat in cat_cols], axis=1)
                conts = np.stack([dataset[cont].values for cont in con_cols], axis=1)
                targets = np.array(targets)
                cat_szs = [len(dataset[col].cat.categories) for col in cat_cols]
                emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
                np_features = np.concatenate((cats, conts), axis=1)
                x_train, x_test, y_train, y_test = train_test_split(np_features, targets, test_size = 0.2, random_state = 42)
                cat_train = x_train[:, : cats.shape[1]]
                cat_test = x_test[:, : cats.shape[1]]
                con_train = x_train[:, cats.shape[1]:conts.shape[1]+1]
                con_test = x_test[:, cats.shape[1]:conts.shape[1]+1]
                tr_dataset = pd.concat([pd.DataFrame(cat_train, columns=cat_cols), pd.DataFrame(con_train, columns=con_cols), pd.DataFrame(y_train)], axis=1)
                tr_dataset.to_csv('outputs/traindataset.csv', mode='w+', sep='\t', index=False)
                te_dataset = pd.concat([pd.DataFrame(cat_test, columns=cat_cols), pd.DataFrame(con_test, columns=con_cols), pd.DataFrame(y_test)], axis=1)
                te_dataset.to_csv('outputs/testdataset.csv', mode='w+', sep='\t', index=False)
                # print(f'`cat_train:{cat_train}')
                # print(f'cat_test:{cat_test}')
                # print(f'con_train:{con_train}')
                # print(f'con_test:{con_test}')
                # print(f'y_train:{y_train}')
                # print(f'y_test:{y_test}')
                # train_dataset = TabularDataset(cat_train, con_train, y_train)
                # test_dataset = TabularDataset(cat_test, con_test, y_test)       
            except Exception as e:
                logger.error(e)
            rm = LearningModel(cat_train, cat_test, con_train, con_test, y_train, y_test, emb_szs, 50, 0.4)           
            await rm.fullRoutePredModel(layers=[200,100], classes=classes)
    '''

class CtrlFlowRulesPred(Controller):
    
    def __init__(self, inetAddress, networkGraph):
        super().__init__(inetAddress=inetAddress, networkGraph=networkGraph)
        self.routingApp = Dijkstra
        # self.start()
    
    async def setupNetwork(self):
        pass
    
    async def manageRoutingRequest(self, req:RequestPacket, data:DataPacket):
        #TODO if path size exceed the open path payload limit
        await self.srRulesSetup(req, data)
        # await self.h2hRuleSetup(req, data)
    
    '''
    async def ctrlLearningWorker(self):    
        import numpy as np
        import pandas as pd
        from ssdwsn.app.smartSDWSN import TSLearningModel
        from sklearn.preprocessing import MinMaxScaler
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        # import tmp.apptmp.lstm_encoder_decoder as led
        import ssdwsn.app.plotting as plot

        plt.rcParams.update({'font.size': 17})
        
        while True:
            # print('CtrlFlowRulesPred')
            await asyncio.sleep(500) 
            networkGraph = self.networkGraph          
            graph = networkGraph.getGraph()                    
            features = []
            targets = []
            if not list(graph.edges(data=False)):
                continue
            try:                
                for i in range(len(self.flowSetupSeq)):
                    features.append([self.flowSetupSeq[i][1]])
                    targets.append([self.flowSetupSeq[i][2]])
                dataset = pd.concat([pd.DataFrame(features, columns=['setuptime']), pd.DataFrame(targets, columns=['rule'])], axis=1)
                dataset.sort_values(by='setuptime').reset_index(drop=True)
                dataset.to_csv('outputs/dataset.csv', mode='w+', sep='\t', index=False)
                sc = MinMaxScaler()
                training_data = sc.fit_transform(np.array(features))
                targets = np.array(targets)

                t = training_data
                y = targets
                t_train, y_train, t_test, y_test = gd.train_test_split(t, y, split = 0.8)
                # plot time series 
                plt.figure(figsize = (18, 6))
                plt.plot(t, y, color = 'k', linewidth = 2)
                plt.xlim([t[0], t[-1]])
                plt.xlabel('$t$')
                plt.ylabel('$y$')
                plt.title('Synthetic Time Series')
                plt.savefig('plots/synthetic_time_series.png')
                # plot time series with train/test split
                plt.figure(figsize = (18, 6))
                plt.plot(t_train, y_train, color = '0.4', linewidth = 2, label = 'Train') 
                plt.plot(np.concatenate([[t_train[-1]], t_test]), np.concatenate([[y_train[-1]], y_test]),
                        color = (0.74, 0.37, 0.22), linewidth = 2, label = 'Test')
                plt.xlim([t[0], t[-1]])
                plt.xlabel(r'$t$')
                plt.ylabel(r'$y$')
                plt.title('Time Series Split into Train and Test Sets')
                plt.legend(bbox_to_anchor=(1, 1))
                plt.tight_layout
                plt.savefig('plots/train_test_split.png')
                #----------------------------------------------------------------------------------------------------------------
                # window dataset
                # set size of input/output windows 
                iw = 80
                ow = 20
                s = 5
                # generate windowed training/test datasets
                Xtrain, Ytrain= gd.windowed_dataset(y_train, input_window = iw, output_window = ow, stride = s)
                Xtest, Ytest = gd.windowed_dataset(y_test, input_window = iw, output_window = ow, stride = s)
                # plot example of windowed data  
                plt.figure(figsize = (10, 6)) 
                plt.plot(np.arange(0, iw), Xtrain[:, 0, 0], 'k', linewidth = 2.2, label = 'Input')
                plt.plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, 0, 0]], Ytrain[:, 0, 0]]),
                        color = (0.2, 0.42, 0.72), linewidth = 2.2, label = 'Target')
                plt.xlim([0, iw + ow - 1])
                plt.xlabel(r'$t$')
                plt.ylabel(r'$y$')
                plt.title('Example of Windowed Training Data')
                plt.legend(bbox_to_anchor=(1.3, 1))
                plt.tight_layout() 
                plt.savefig('plots/windowed_data.png')
                #----------------------------------------------------------------------------------------------------------------
                # LSTM encoder-decoder
                # convert windowed data from np.array to PyTorch tensor
                X_train, Y_train, X_test, Y_test = gd.numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)
                # specify model parameters and train
                model = led.lstm_seq2seq(input_size = X_train.shape[2], hidden_size = 15)
                loss = await model.train_model(X_train, Y_train, n_epochs = 50, target_len = ow, batch_size = 5, training_prediction = 'mixed_teacher_forcing', teacher_forcing_ratio = 0.6, learning_rate = 0.01, dynamic_tf = False)
                # plot predictions on train/test data
                plot.plot_train_test_results(model, Xtrain, Ytrain, Xtest, Ytest)
                plt.close('all')

                t_seq_len = 4
                y_seq_len = 2
                sequences = []
                labels = []
                data_size = training_data.shape[0]
                for i in tqdm(range(data_size - t_seq_len - y_seq_len)):
                    sequence = training_data[i:i+t_seq_len]
                    label_position = i + t_seq_len
                    label = targets[label_position:label_position+y_seq_len]
                    sequences.append(sequence)
                    labels.append(label)
                sequences = np.array(sequences) 
                labels = np.array(labels)
                train_size = int(len(labels) * 0.8)
                test_size = len(labels) - train_size
                dataX = sequences
                dataY = labels
                print(f'X: {dataX}')
                print(f'Y: {dataY}')
                x_train = sequences[0:train_size]
                y_train = labels[0:train_size]
                x_test = sequences[train_size:len(sequences)]
                y_test = labels[train_size:len(labels)]      
                # plot time series 
                plt.figure(figsize = (18, 6))
                plt.plot(training_data, targets, color = 'k', linewidth = 2)
                plt.xlim([training_data[0], training_data[-1]])
                plt.xlabel('$t$')
                plt.ylabel('$y$')
                plt.title('Flow Setup Time Series')
                plt.savefig('outputs/flowsetup_time_series.png')
                plt.show()
                
                # tr_dataset = pd.concat([pd.DataFrame(np_features, columns=cat_cols), pd.DataFrame(y_train, columns=['setuptime'])], axis=1)
                # tr_dataset.to_csv('outputs/traindataset.csv', mode='w+', sep='\t', index=False)
                # te_dataset = pd.concat([pd.DataFrame(x_test, columns=cat_cols), pd.DataFrame(y_test, columns=['setuptime'])], axis=1)
                # te_dataset.to_csv('outputs/testdataset.csv', mode='w+', sep='\t', index=False)
                # print(f'`cat_train:{cat_train}')
                # print(f'cat_test:{cat_test}')
                # print(f'con_train:{con_train}')
                # print(f'con_test:{con_test}')
                # print(f'y_train:{y_train}')
                # print(f'y_test:{y_test}')
            except Exception as e:
                logger.error(e)
                from tkinter import messagebox
                messagebox.showerror('ERROR', e)
    
                
            # rm = TSLearningModel(x_train, x_test, y_train, y_test, y_seq_len, 50, 0.001, 0.2)     
            # await rm.seq2seq_lstm(layers=[100], scaler=sc)      
            
            try:
                for src in nodes:
                    for dst in nodes:
                        cost, route = await rt.getRoute(src, dst)
                        if route:
                            for i in range(len(route)-1):
                                srcNode = graph.nodes.get(src)
                                dstNode = graph.nodes.get(dst)
                                curNode = graph.nodes.get(str(srcNode['net'])+'.'+route[i].__str__())
                                rule = (route[0].__str__(), route[-1].__str__(), route[i].__str__(), route[i+1].__str__())
                                if self.flowReqStats.get(rule):
                                    for item in self.flowReqStats[rule]: 
                                        datatype = srcNode['datatype']
                                        srcaddr = route[0].intValue()
                                        dstaddr = route[-1].intValue()
                                        curnode = route[i].intValue()
                                        nxhroute = route[i+1].intValue()
                                        srcport = srcNode['port']
                                        dstport = dstNode['port']
                                        curport = curNode['port']
                                        curposx = curNode['pos'][0]
                                        curposy = curNode['pos'][1]
                                        curposz = 0
                                        reqtime = item['reqtime']
                                        setuptime = item['setuptime']
                                        features.append([datatype, srcaddr, dstaddr, curnode, nxhroute, srcport, dstport, curport, curposx, curposy, curposz, reqtime, setuptime])
                                        targets.append([rule])                            
                                    
                cat_cols = ['datatype']
                con_cols = ['srcaddr', 'dstaddr', 'curaddr', 'nxhroute', 'srcport', 'dstport', 'curport', 'curposx', 'curposy', 'curposz', 'reqtime', 'setuptime']
                dataset = pd.concat([pd.DataFrame(features, columns=cat_cols+con_cols), pd.DataFrame(targets, columns=['setuptime'])], axis=1)
                dataset.sort_values(by='reqtime').reset_index(drop=True)
                dataset.to_csv('outputs/dataset.csv', mode='w+', sep='\t', index=False)
                for cat in cat_cols:
                    dataset[cat] = dataset[cat].astype('category')
                cats = np.stack([dataset[cat].cat.codes.values for cat in cat_cols], axis=1)
                conts = np.stack([dataset[cont].values for cont in con_cols], axis=1)
                targets = np.array(targets)  
                training_data = np.concatenate((cats, conts, targets), axis=1)
                
                sc = MinMaxScaler()
                training_data = sc.fit_transform(training_data)    
                                
                t_seq_len = 2
                sequences = []
                labels = []
                data_size = training_data.shape[0]
                for i in tqdm(range(data_size - t_seq_len)):
                    sequence = training_data[i:i+t_seq_len, :-1]
                    label_position = i + t_seq_len
                    label = training_data[label_position, -1]
                    sequences.append(sequence)
                    labels.append([label])
                sequences = np.array(sequences)       
                labels = np.array(labels)
                
                train_size = int(len(labels) * 0.67)
                test_size = len(labels) - train_size
                dataX = np.array(sequences)
                dataY = np.array(labels)
                x_train = np.array(sequences[0:train_size])
                y_train = np.array(labels[0:train_size])
                
                x_test = np.array(sequences[train_size:len(sequences)])
                y_test = np.array(labels[train_size:len(labels)])       
    
                # tr_dataset = pd.concat([pd.DataFrame(np_features, columns=cat_cols), pd.DataFrame(y_train, columns=['setuptime'])], axis=1)
                # tr_dataset.to_csv('outputs/traindataset.csv', mode='w+', sep='\t', index=False)
                # te_dataset = pd.concat([pd.DataFrame(x_test, columns=cat_cols), pd.DataFrame(y_test, columns=['setuptime'])], axis=1)
                # te_dataset.to_csv('outputs/testdataset.csv', mode='w+', sep='\t', index=False)
                # print(f'`cat_train:{cat_train}')
                # print(f'cat_test:{cat_test}')
                # print(f'con_train:{con_train}')
                # print(f'con_test:{con_test}')
                # print(f'y_train:{y_train}')
                # print(f'y_test:{y_test}')
            except Exception as e:
                error(e)
                from tkinter import messagebox
                messagebox.showerror('ERROR', e)
                
            rm = TSLearningModel(x_train, x_test, y_train, y_test, 50, 0.001, 0.2)     
            await rm.lstm(layers=[100], scaler=sc)
            
    '''       
            
                                
# class MyCustomNamespace(socketio.AsyncClientNamespace):
    
#     async def on_connect(self):     
#         global sio, ngraph
#         data = json_graph.node_link_data(ngraph.getGraph())  
#         await emitStats('updateGraph', data)

#     async def on_disconnect(self):
#         pass

#     async def on_my_event(self, data):
#         self.emit('sr_response', data)