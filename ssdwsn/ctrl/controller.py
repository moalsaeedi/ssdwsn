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
from collections import defaultdict, deque
import re
from os import path
from networkx.readwrite import json_graph
from numpy.core.fromnumeric import ndim, size
from ssdwsn.app.routing import Dijkstra
from ssdwsn.ctrl.graph import Graph
import time
import asyncio, logging
from ssdwsn.openflow.action import ForwardUnicastAction
from ssdwsn.openflow.window import Window
from ssdwsn.util.utils import mergeBytes, CustomFormatter, setBitRange, bitsToBytes, zero_division
from math import ceil, log, sqrt, atan2, pi
from ssdwsn.openflow.entry import Entry
from ssdwsn.data.addr import Addr
from ssdwsn.app.routing import Dijkstra
from ssdwsn.openflow.packet import ConfigPacket, ConfigProperty, OpenPathPacket, Packet, DataPacket, ReportPacket, RequestPacket, ResponsePacket, AggrPacket, RegProxyPacket
from ssdwsn.data.sensor import SensorType
from ssdwsn.data.intf import IntfType
from ssdwsn.util.constants import Constants as ct
from expiringdict import ExpiringDict
import zmq
from zmq.asyncio import Context, Poller
import socketio
from socketio import exceptions
import json
from sys import stderr
from sklearn.model_selection import train_test_split

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
        self.buffer_size = 1
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
            await self.connect()
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
        tasks.append(self.loop.run_in_executor(None, self.packetInWorker))
        tasks.append(self.loop.run_in_executor(None, self.ctrlLearningWorker))
        tasks.append(self.loop.run_in_executor(None, self.nodeSTWorker))
        await asyncio.gather(*tasks)

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
                        for link in data['links']:
                            self.networkGraph.updateEdgeColor(link['source'], link['target'], link['color'])
                    elif data[len('ST'):len('ST')+len('NODE')] == 'NODE':
                        data = json.loads(data[len('ST')+len('NODE'):])
                        self.networkGraph.updateNodeColor(data['id'], data['color'])
                    elif data[len('ST'):len('ST')+len('perf')] == 'perf':
                        data = json.loads(data[len('ST')+len('perf'):])
                        for key,val in data.items():
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
                        self.networkGraph.getNode(data['id'])['pos'] = data['pos']
                        self.networkGraph.getNode(data['id'])['batt'] = data['batt']
                        self.networkGraph.getNode(data['id'])['betti'] = data['betti']
                        self.networkGraph.getNode(data['id'])['rptti'] = data['rptti']
                        for key,val in data.items():
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
                            'avgrenergy':round(state_mean[0], 4)
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
        modified = await self.networkGraph.updateMap(packet)
        if modified:
            ## reactive update to modification in the network topology
            pass

    async def rxRequest(self, packet:RequestPacket):
        p = await self.putInRequestCache(packet)
        if not(p is None):
            await self.manageRoutingRequest(packet, p)

    async def rxConfig(self, packet:ConfigPacket):
        #TODO
        pass

    async def rxRegProxy(self, packet:RegProxyPacket):
        #TODO multiple sinks or use broker addr to multi-subscribers sinks  
        sinkid = str(packet.getNet())+'.'+packet.getSrc().__str__()
        if sinkid not in self.sinks:      
            self.sinks.append(sinkid)

    async def rxAggr(self, packet:AggrPacket):
        aggrtype = packet.getAggrType()
        # '''
        aggrpackets = await self.getAggrPackets(packet)
        for p in aggrpackets:
            if aggrtype == ct.REPORT:
                modified = await self.networkGraph.updateMap(p)
                if modified:
                    ## reactive update to modification in the network topology
                    pass

    async def getAggrPackets(self, packet:AggrPacket):
        graph = self.networkGraph
        payload = packet.getAggrPayload()
        aggrtype = packet.getAggrType()
        index = 0
        plindex = 0
        packets = []
        while len(payload) > index:
            pl_size = payload[index]
            src = Addr(payload[index+ct.AGGR_SRC_INDEX:index+ct.AGGR_SRC_INDEX+ct.AGGR_SRC_LEN])
            src_id = str(packet.getNet())+'.'+src.__str__()
            p = Packet(net=packet.getNet(), src=src, dst=packet.getDst())#TODO
            p.data[ct.TS_INDEX:ct.TS_INDEX+ct.TS_LEN] = payload[index+ct.AGGR_TS_INDEX:index+ct.AGGR_TS_INDEX+ct.AGGR_TS_LEN]
            p.data[ct.TYP_INDEX] = aggrtype
            plindex += ct.AGGR_PLHDR_LEN
            # construct packet payload
            if aggrtype == ct.REPORT:
                pl = bytearray()
                if (index+pl_size) > plindex:
                    if payload[plindex] != ct.DIST_INDEX:
                        pl += int.to_bytes(graph.getNode(src_id)['distance'], 1, 'big', signed=False)
                    else:
                        pl += int.to_bytes(payload[plindex+1], 1, 'big', signed=False)
                        plindex += 2
                else:
                    pl += int.to_bytes(graph.getNode(src_id)['distance'], 1, 'big', signed=False)
                if (index+pl_size) > plindex:
                    if payload[plindex] != ct.BATT_INDEX:
                        pl += int.to_bytes(graph.getNode(src_id)['batt'], ct.BATT_LEN, 'big', signed=False)
                    else:
                        pl += payload[plindex+1:plindex+1+ct.BATT_LEN]
                        plindex += ct.BATT_LEN+1
                else:
                    pl += int.to_bytes(graph.getNode(src_id)['batt'], ct.BATT_LEN, 'big', signed=False)
                if (index+pl_size) > plindex:
                    if payload[plindex] != ct.POS_INDEX:
                        pl += int.to_bytes(int(graph.getNode(src_id)['x']), 2, 'big', signed=False) + int.to_bytes(int(graph.getNode(src_id)['y']), 2, 'big', signed=False)
                    else:
                        pl += payload[plindex+1:plindex+1+ct.POS_LEN]
                        plindex += ct.POS_LEN+1
                else:
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
            index += pl_size+1
            plindex = index
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
                
            #Flow setup rules from the sink to the request cur (sink_to_cur route)
            _, stc_route = rt.getRoute(sink, cur)
            if stc_route:
                await self.pathRulesSetup(net=net, sinkId=sink, sinkAddr=req.getDst(), src=req.getDst(), route=stc_route)
                
            #Flow setup rules from the request cur to the flow destination (cur_to_dst route)     
            cost, route = rt.getRoute(cur, dst)
            features = []
            if route:
                await self.pathRulesSetup(net=net, sinkId=sink, sinkAddr=req.getDst(), src=req.getDst(), route=route)
                
                for i in range(len(route) -1):
                    rule = (srcNode, dstNode, route[i].__str__(), route[i+1].__str__())
                    self.flowReqStats[rule].append({'reqtime': time.time(), 'setuptime': time.time() - ct.RL_IDLE/4}) 
                    if not self.rule2id.get(rule):
                        self.rule2id[rule] = len(self.rule2id)
                    self.flowSetupSeq.append([time.time(), time.time() - ct.RL_IDLE/4, self.rule2id[rule]])                    
            
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
                    self.flowReqStats[rule].append({'reqtime': time.time(), 'setuptime': time.time() - ct.RL_IDLE/4}) 
                    if not self.rule2id.get(rule):
                        self.rule2id[rule] = len(self.rule2id)
                    self.flowSetupSeq.append([time.time(), time.time() - ct.RL_IDLE/4, self.rule2id[rule]])
            
            ruleFwd = None
            ruleBwd = None
            index = 0
            for node in route:   
                cost, toSinkRoute = rt.getRoute(src=sink, dst=(str(net) + "." + node.__str__()))
                tmpIndex = 0
                for toSinkNode in toSinkRoute:
                    if toSinkNode.__str__() == node.__str__():
                        if node.__str__() != route[len(route) - 1].__str__():
                            ruleFwd = Entry()                            
                            ruleFwd.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1).setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST).setRhs(route[len(route) - 1].intValue()))
                            ruleFwd.addAction(ForwardUnicastAction(nxtHop=route[index + 1]))                               
                            await self.addNodeRule(net, sinkAddr, toSinkNode, ruleFwd)
                    else:
                        ruleFwd = Entry()
                        ruleFwd.addWindow(Window().setOperator(ct.EQUAL).setSize(ct.W_SIZE_1).setLhsOperandType(ct.PACKET).setLhs(ct.DST_INDEX).setRhsOperandType(ct.CONST).setRhs(toSinkRoute[len(toSinkRoute) - 1].intValue()))
                        ruleFwd.addAction(ForwardUnicastAction(nxtHop=toSinkRoute[tmpIndex + 1]))
                        await self.addNodeRule(net, sinkAddr, toSinkNode, ruleFwd)
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
      
class ATCP_ctrl(Controller):
    """ Adaptive Traffic Control PPO-DRL based Controller
    """
    def __init__(self, inetAddress: tuple = None, networkGraph: Graph = None):
        super().__init__(inetAddress, networkGraph)
        self.name = 'ATCP-ctrl'
        self.routingApp = Dijkstra        
        # self.start()
         
    async def setupNetwork(self):
        pass
    
    async def manageRoutingRequest(self, req:RequestPacket, data:DataPacket):
        #TODO if path size exceed the open path payload limit
        await self.srRulesSetup(req, data)
    
    def ctrlLearningWorker(self):
        async def async_thread():
            # run the PPO_DRL model after 300 seconds of initiating the network simulation (discovery phase)
            await asyncio.sleep(300)
            from ssdwsn.app.agent import PPO_ATCP
            agent = PPO_ATCP(ctrl=self, batch_size=250, samples_per_epoch=1)
            await agent.run()
        asyncio.run(async_thread())

class NSFP_ctrl(Controller):
    """ Network State Forcastin PPO_DRL based Controller
    """
    def __init__(self, inetAddress: tuple = None, networkGraph: Graph = None):
        super().__init__(inetAddress, networkGraph)
        self.name = 'NSFP-ctrl'
        self.routingApp = Dijkstra        
        # self.start()
        
    async def setupNetwork(self):
        pass
    
    async def manageRoutingRequest(self, req:RequestPacket, data:DataPacket):
        #TODO if path size exceed the open path payload limit
        await self.srRulesSetup(req, data)
    
    def ctrlLearningWorker(self):
        async def async_thread():
            # run the RL model after 300 seconds of initiating the network simulation (discovery phase)
            await asyncio.sleep(300)
            from ssdwsn.app.agent import PPO_NSFP
            agent = PPO_NSFP(ctrl=self, batch_size=250, samples_per_epoch=1, obs_time=15)
            await agent.run()
        asyncio.run(async_thread())
        