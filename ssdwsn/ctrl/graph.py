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

import networkx as nx
import time
from multiprocessing import Process
import numpy as np
from networkx.classes.function import density
from ssdwsn.data.sensor import SensorType
from ssdwsn.util.constants import Constants as ct
from ssdwsn.util.utils import getColorVal
from ssdwsn.openflow.packet import ReportPacket
from ssdwsn.data.intf import IntfType
import pandas as pd

class Graph:    
    """Network Graph (Statefull network topology view)"""
    def __init__(self, directed=True):
        """Create a new controller's global view of the network
        lastModification: last timestamp a node is modified
        mxNodes: maximun number of nodes in the simulation
        mnPort: minimum port address value
        mxPort: maximum port address value 
        gridWidth, gridHeight: simulation grid sizes
        lastCheck: last time the graph consistency has been checked
        
        Args:
            directed (bool, optional): if true means that an edge is directed from one vertex (node) to another. Defaults to True.
        """
        if directed:
            self.graph = nx.MultiDiGraph()
        else: self.graph = nx.MultiGraph()
        self.lastModification = 0
        self.mxNodes = ct.MAX_NODES
        self.mnPort = ct.BASE_NODE_PORT
        self.mxPort = ct.TOP_MOTE_PORT
        self.gridWidth = 10000
        self.gridHeight = 10000
        self.lastCheck = round(time.time(), 4)
        self.rmNodes = []
        # socketio
        self.sio = None
        # zmq
        self.zmqContext = None

    def addNode(self, id:str):
        """Add new node to the graph

        Args:
            id (str): node id

        Returns:
            str: node id
        """
        self.graph.add_node(id)
        # self.graph.nodes[id]['weight'] = 0.0
        self.graph.nodes[id]["lastupdate"] = round(time.time(), 4)
        self.graph.nodes[id]["batt"] = 0.0
        self.graph.nodes[id]["delay"] = 0.0 #delay of sending packets to sink
        self.graph.nodes[id]["throughput"] = 0.0 #throughput of link to sink
        self.graph.nodes[id]["alinks"] = 0.0 #node active links
        self.graph.nodes[id]["flinks"] = 0.0 #node idle links
        self.graph.nodes[id]["txpackets"] = 0.0 #node transmitted packets per sec
        self.graph.nodes[id]["txpackets_val"] = 0.0 #node transmitted packets
        self.graph.nodes[id]["txbytes"] = 0.0 #node transmitted bytes per sec
        self.graph.nodes[id]["txbytes_val"] = 0.0 #node transmitted bytes
        self.graph.nodes[id]["rxpackets"] = 0.0 #node received packets per sec
        self.graph.nodes[id]["rxpackets_val"] = 0.0 #node received packets
        self.graph.nodes[id]["rxbytes"] = 0.0 #node received bytes per sec
        self.graph.nodes[id]["rxbytes_val"] = 0.0 #node received bytes
        self.graph.nodes[id]["txpacketsin"] = 0.0 #sink node transmitted packetsIn per sec
        self.graph.nodes[id]["txpacketsin_val"] = 0.0 #sink node transmitted packetsIn
        self.graph.nodes[id]["txbytesin"] = 0.0 #sink node transmitted bytesIn per sec
        self.graph.nodes[id]["txbytesin_val"] = 0.0 #sink node transmitted bytesIn
        self.graph.nodes[id]["rxpacketsout"] = 0.0 #sink node received packetsOut per sec
        self.graph.nodes[id]["rxpacketsout_val"] = 0.0 #sink node received packetsOut
        self.graph.nodes[id]["rxbytesout"] = 0.0 #sink node received bytesOut per sec
        self.graph.nodes[id]["rxbytesout_val"] = 0.0 #sink node received bytesOut
        self.graph.nodes[id]["drpackets"] = 0.0 #node dropped packets per sec
        self.graph.nodes[id]["drpackets_val"] = 0.0 #node dropped packets
        self.graph.nodes[id]["betti"] = 0.0 #Neighbors Discovery beacons transmission time interval
        self.graph.nodes[id]["rptti"] = 0.0 #node reporting transmission time interval
        return id
    
    def setupNode(self, id:str, net:int, pos:tuple=None, color:str=None, intftype:str=None, datatype:str=None, 
        batt:int=None, port:int=None, antrange:int=None, distance:int=None, denisty:int=None, lastupdate:float=None, active:bool=True):
        """Setup a new added node (information provided from data plane)
            
        Args:
            id (str): node id
            net (int): subnet id
            pos (tuple, optional): node position in the grid (x,y,z). Defaults to None.
            color (str, optional): node visualization color (blue for a Mote, red for a sink/gateway). Defaults to None.
            intftype (str, optional): physical node interface IEEE802.15.4. Defaults to None.
            datatype (str, optional): sensor type. Defaults to None.
            batt (int, optional): resedual energy. Defaults to None.
            port (int, optional): port address. Defaults to None.
            antrange (int, optional): antenna range. Defaults to None.
            distance (int, optional): distance to sink (hops count to sink). Defaults to None.
            denisty (int, optional): neighbors size (node degree). Defaults to None.
            lastupdate (float, optional): network status reporting time. Defaults to None.
        """        
        self.graph.nodes[id]["id"] = id
        self.graph.nodes[id]["net"] = net
        self.graph.nodes[id]["pos"] = pos
        self.graph.nodes[id]["x"] = pos[0]
        self.graph.nodes[id]["y"] = pos[1]
        self.graph.nodes[id]["z"] = 0.0
        if color:
            self.graph.nodes[id]["color"] = color
        self.graph.nodes[id]["intftype"] = intftype
        self.graph.nodes[id]["intftypeval"] = IntfType.fromStringToInt(intftype)
        self.graph.nodes[id]["datatype"] = datatype
        self.graph.nodes[id]["datatypeval"] = SensorType.fromStringToInt(datatype)
        if lastupdate > self.graph.nodes[id]["lastupdate"]:
            self.graph.nodes[id]["batt"] = batt
        self.graph.nodes[id]["port"] = port
        self.graph.nodes[id]["antrange"] = antrange
        self.graph.nodes[id]["distance"] = distance
        self.graph.nodes[id]["denisty"] = denisty
        links = list(self.graph.edges(id))
        if links:
            # print(f'links: {links}')
            # print(list(self.graph.edges(id, data="active")))
            alinks = len([edge[1] for edge in list(self.graph.edges(id, data="color")) if (not(edge[1] == 'black'))])
            flinks = len([edge[1] for edge in list(self.graph.edges(id, data="color")) if (edge[1] == 'black')])
            # print(f'active links: {alinks}')
            self.graph.nodes[id]["alinks"] = alinks #active links to neighbors
            self.graph.nodes[id]["flinks"] = flinks #free (non-active) links to neighbors
        if batt == ct.MAX_LEVEL:
            self.graph.nodes[id]["issink"] = True
        else:
            self.graph.nodes[id]["issink"] = False
        self.graph.nodes[id]["active"] = active
        self.graph.nodes[id]["lastupdate"] = lastupdate
    
    def getNode(self, id:str):        
        return self.graph.nodes[id]

    def getGraph(self):
        return self.graph
    
    def getState(self, nodes=None, cols:list=None):
        """An observation (state) of the network extracted from the Controllers' Golbal visibility Graph
        
        batt: node's resedual energy
        delay: propagation delay from the data plane node to the sink (msec)
        throughput: rate at which data traverses a link (node-to-sink) (kbps)
        distance: number of hops from a node to the sink
        denisty: number of node's neighbors (adjacencies)
        txpackets: number of transmitted packets per second in a Node
        rxpackets: number of received packets per second in a Node
        drpackets: number of droped packets per second in a Node   
        betti: transmission time interval (TTI) for Neighbor Discovery    
        rptti: transmission time interval (TTI) for reporting topological ans statistical info to the sink      
        alinks: number of active links (W channels) of a Node (links to neighbors that have OF rules configured)
        flinks: number of idel links (W channels) of a Node (links to neighbors that have no OF rules configured)
        x: Node's x position in the Graph
        y: Node's y position in the Graph
        z: Node's z position in the Graph (2d dimention z=0) 
        intftypeval: interface type value (e.g., 6lowpan intf = 0)       
        datatypeval: sensor data type value (e.g., temperature data type value = 3)
        """
        state = np.array(list(self.graph.nodes(data=True)))[:,1].tolist()
        state_nodes = np.array(list(self.graph.nodes)).reshape(-1,1)
        # state = pd.DataFrame(state)[['batt', 'delay', 'throughput', 'distance', 'denisty', 'txpackets', 'txbytes', 'rxpackets', \
        #     'rxbytes', 'energycons', 'alinks', 'flinks', 'x', 'y', 'z', 'intftypeval', 'datatypeval', 'active', 'id']].values
        
        # get state variables
        state_cols = ['batt', 'delay', 'throughput', 'distance', 'denisty', 'txpackets', 'txpackets_val', 'txbytes', 'txbytes_val', 'rxpackets', \
                'rxpackets_val', 'rxbytes', 'rxbytes_val', 'drpackets', 'drpackets_val', 'alinks', 'flinks', 'x', 'y', 'z', 'intftypeval', 'datatypeval', 'txpacketsin', 'txpacketsin_val', 'txbytesin', \
                'txbytesin_val', 'rxpacketsout', 'rxpacketsout_val', 'rxbytesout', 'rxbytesout_val', 'port', 'betti', 'rptti']        
        
        _cols = cols if cols else state_cols
        extended_cols = ['issink', 'active', 'id']
        _nodes = nodes if not(nodes is None) else state_nodes
        
        state = pd.DataFrame(state, columns=_cols+extended_cols).values
        # get indexes of rows and columns where node (active, non-sink (battery level is not equal to the maximum value (non-chargable)), distance>0 and alinks>0)
        state_idxs = np.where((np.isin(state[:,-1] , _nodes) & (state[:,-3] == False) & (state[:,-2] == True)))
        # get indexx of the sinks
        sink_idxs = np.where((state[:,-3] == True))
        state = state[:,:len(_cols)]
        
        # networks' nodes state
        _state = state[state_idxs,:].squeeze(0).astype(float)
        _state_nodes = state_nodes[state_idxs,:].squeeze().reshape(-1,1)

        # sink nodes state
        _sink_state = None
        _sink_state = state[sink_idxs,:].squeeze(0).astype(float)
        _sink_state_nodes = state_nodes[sink_idxs,:].squeeze().reshape(-1,1)
        return _state, _state_nodes, _sink_state, _sink_state_nodes

    def getStateMean(self):
        """The mean of the network's observation (state)
        """
        state, state_nodes, sink_state, sink_state_nodes = self.getState()
        state_mean = state.mean(axis=0, keepdims=True).flatten() if state.size != 0 else np.zeros(shape=state.shape).flatten()
        sink_state_mean = sink_state.mean(axis=0, keepdims=True).flatten() if sink_state.size != 0 else np.zeros(shape=sink_state.shape).flatten()
        return state_mean, state_nodes, sink_state_mean, sink_state_nodes
    
    def getLastModification(self):
        return self.lastModification
    
    def removeNode(self, id:str):
        # first remove its edges
        self.removeNodeEdges(id)
        self.graph.remove_node(id)         

    def removeEdge(self, frm:str, to:str):
        self.graph.remove_edge(frm, to)
    
    def removeNodeEdges(self, id:str):
        rmEdges = []
        for e in self.graph.edges():
            if id in e:
                rmEdges.append(e)
        self.graph.remove_edges_from(rmEdges)
        self.graph.nodes[id]['color'] = 'gray'
        self.graph.nodes[id]['active'] = False
            
    def nodereporTtime(self, id:str, now:float):     
        self.graph.nodes[id]["lastupdate"] = now
     
    def getPosition(self, id:str):
        return self.graph.nodes[id]['pos']
   
    def updatePosition(self, id:str, x:int, y:int):
        self.graph.nodes[id]["pos"] = (x, y)
    
    def getDistance(self, id:str):
        return self.graph.nodes[id]['distance']

    def updateNodeState(self, id:str, *params, val):
        for i in params:
            self.graph.nodes[id][params[i]] = val

    def updateBattery(self, id:str, batt:int):
        self.graph.nodes[id]["batt"] = batt
        
    def isActive(self, id:str):
        return self.graph.nodes[id]['active']
    
    def getNodeColor(self, id:str):
        return self.graph.nodes[id]['color']
    
    def setNodeColor(self, id:str, val:str):
        self.graph.nodes[id]['color'] = val

    def addEdge(self, frm:str, to:str, rssi:int, color:str):
        """Add an edge

        Args:
            frm (str): from node id
            to (str): to node id
            rssi (int): rssi value in destination node
            color (str): color of the edge

        Returns:
            str: edge key <srcAddr-desAddr>
        """
        self.graph.add_edge(
            frm, to, 
            key="{}-{}".format(frm, to), 
            rssi=rssi, 
            color=color, 
            weight= ( #avg weight (heigher weight better link) w = (rssi/mrssi + 1 - (delay/mdelay) + throughput/bandwidth + 1 - ((dist+1)/mdist))/4
                (rssi/ct.RSSI_MAX)+ #link quality to the target node
                (1- (self.graph.nodes[to]["delay"]/ct.MAX_DELAY))+ #ratio of delay (target node to sink) to the maximum accepted delay
                (self.graph.nodes[to]["throughput"]/IntfType.getParams(self.graph.nodes[to]["intftype"])['bandwidth'])+ #ratio of throughput (target node to sink) to the maximum bandwidth
                (1- ((self.graph.nodes[to]["distance"]+1)/(ct.DIST_MAX+1))))/4,
            time=round(time.time(), 4)
        )
        return "{}-{}".format(frm, to)
    
    def getEdge(self, frm:str, to:str):
        return self.graph.get_edge_data(frm, to)

    def updateEdgeColor(self, frm:str, to:str, color:str):
        key = "{}-{}".format(frm, to)
        if self.getEdge(frm, to):
            self.graph.edges[frm, to, key]['color'] = color

    def updateEdgeWeight(self, frm:str, to:str, weight:float):
        key = "{}-{}".format(frm, to)
        if self.getEdge(frm, to):
            self.graph.edges[frm, to, key]['weight'] = weight

    def updateNodeColor(self, id:str, color:str):
        self.graph.nodes[id]["color"] = color

    def updateDistance(self, id:str, dist:int):
        self.graph.nodes[id]["distance"] = dist
                
    async def updateMap(self, packet:ReportPacket):
        """Update the network global view every time a controller receives a report packet

        Args:
            packet (ReportPacket): report packet contains an information of a node and its neighbors

        Returns:
            bool: modified? True/False
        """
        modified = False
        # try:
        # modified = self.checkConsistency()
        addr = packet.getSrc()
        net = packet.getNet()
        pos = (packet.getPosition()[0], packet.getPosition()[1])
        # print(pos)
        intftype = IntfType.fromValue(packet.getIntfType())
        antrange = IntfType.getParams(intftype)['antRange']
        datatype = SensorType.fromValue(packet.getSensorType())
        batt = packet.getBattery()        
        port = packet.getPort()
        distance = packet.getDistance()
        neighbors = packet.getNeighborsSize()
        # renergy = round(2*((batt-0)/(ct.MAX_LEVEL-0))-1, 4) #(b-a)*((x - min_x)/(max_x - min_x))+a; for [a,b]
        # distance = round(2*((packet.getDistance()-1)/(ct.TTL_MAX-1))-1, 4) #(b-a)*((x - min_x)/(max_x - min_x))+a; for [a,b]
        # density = round(2*((packet.getNeighborsSize()-1)/(ct.MAX_NEIG-1))-1, 4) #(b-a)*((x - min_x)/(max_x - min_x))+a; for [a,b]
        nodeId = str(net) + "." + addr.__str__()       
        node = self.getNode(nodeId)
        lastupdate = packet.getTS()
        if nodeId not in self.rmNodes:
            if node:
                self.setupNode(id=nodeId, net=net, pos=pos, intftype=intftype, datatype=datatype, batt=batt, 
                            port=port, antrange=antrange, distance=distance, denisty=neighbors, lastupdate=lastupdate)
                # oldEdges = list(self.graph.edges(nodeId))
                edges = list(self.graph.edges(nbunch=nodeId, data=True, keys=True))
                edgekeys = [edge[2] for edge in edges]
                neighbors = []
                for ng in packet.getNeighbors():
                    ngNodeaddr = ng['addr']
                    ngNodeId = str(net) + "." + ngNodeaddr.__str__()
                    neighbors.append(ngNodeId)
                    # if self.getNode(ngNodeId) is None:
                    #     id = self.addNode(ngNodeId)
                    #     self.setupNode(id, net)
                    
                    # rssi = ct.MAX_BYTE - packet.getLinkQuality(i)
                    edgeKey = nodeId+'-'+ngNodeId
                    rssi = ng['rssi']
                    # color = getColorVal(ng['color'])
                    # self.updateEdgeColor(nodeId, ngNodeId, color)
                    if edgeKey in edgekeys:
                        oldRssi = self.graph.edges[nodeId, ngNodeId, edgeKey]['rssi']                             
                        if rssi != oldRssi:
                            if rssi < ct.RSSI_MIN:
                                self.graph.remove_edge(*(nodeId, ngNodeId))
                                modified = True                        
                            else: 
                                self.graph.edges[nodeId, ngNodeId, edgeKey]['rssi'] = rssi                        
                        
                        edgekeys.remove(edgeKey)
                    else: 
                        # self.addEdge(nodeId, ngNodeId, rssi, color)   
                        self.addEdge(nodeId, ngNodeId, rssi, 'black')   
                        modified = True
                        
                if edgekeys:
                    for edgeKey in edgekeys:
                        self.graph.remove_edge(*(edgeKey.split('-')[0], edgeKey.split('-')[1]))
                    modified = True        
            else:
                self.addNode(nodeId)
                self.setupNode(id=nodeId, net=net, pos=pos, intftype=intftype, datatype=datatype, batt=batt, 
                            port=port, antrange=antrange, distance=distance, denisty=neighbors, lastupdate=lastupdate)

                for ng in packet.getNeighbors():
                    ngNodeaddr = ng['addr']
                    ngNodeId = str(net) + "." + ngNodeaddr.__str__()  
                    # if self.getNode(ngNodeId) is None:
                    #     id = self.addNode(ngNodeId)
                    #     self.setupNode(id, net)
                    
                    # rssi = ct.MAX_BYTE - packet.getLinkQuality(i)
                    rssi = ng['rssi']
                    # color = getColorVal(ng['color'])
                    self.addEdge(nodeId, ngNodeId, rssi, 'black')
                modified = True

            if modified:
                self.lastModification += 1
        # except Exception as e:
        #     error(f'Updating graph error: {e}')
        return modified
             
    def checkConsistency(self):
        """Check the consistency of the controller's network view with the data plane. 
        A node is removed if (now - lastupdate time > consistency timeout in msec).

        Args:
            now (float): current timestamp

        Returns:
            _type_: _description_
        """
        modified = False
        now = round(time.time(), 4)
        if (now - self.lastCheck) * 1000 > ct.REP_IDLE:
            self.lastCheck = now
            toRmvList = []
            for node in self.graph.nodes(data=True):
                id = node[0]
                data = node[1]
                if data["datatype"] != SensorType.sink.name:
                    if data["net"] < ct.THRES and data["lastupdate"] is not None and not self.isAlive(ct.REP_IDLE, data["lastupdate"], now):
                        toRmvList.append(id)
                        modified = True
            for node in toRmvList:
                # remove edges of the node and keep the node just for visualization purpose
                self.removeNodeEdges(node)
                # self.removeNode(node[0])                              
        return modified
    
    def isAlive(self, timeout:int, last:float, now:float):
        return ((now - last) * 1000 < timeout)
    # def updateGraph(self):
    #     asyncio.ensure_future(self.visualizeGraph())    

    # async def visualizeGraph(self):
    #     data = json_graph.node_link_data(self.getGraph())
    #     async with aiohttp.ClientSession(trust_env=True) as session:
    #         async with session.post(url=ct.SIM_URL+'/updategraph', json=data) as response:
    #             await asyncio.sleep(0.5)
    #             # async with aiofiles.open(dir_path, mode='r+') as f:
    #             #     # await response.text()
    #             #     f.seek(0)
    #             #     await f.write(str(data))
    #             #     f.truncate()
    #     return 'Graph is visualized'