import asyncio
from collections import defaultdict, deque
from networkx.algorithms.connectivity import edge_augmentation
from networkx.classes import graph
from networkx.classes.multidigraph import MultiDiGraph
from networkx.generators import directed
import networkx as nx
from ssdwsn.data.addr import Addr
# from ssdwsn.ctrl.networkGraph import dijkstra, shortest_path, Graph

class Routing:
    
    def __init__(self, graph:nx.MultiDiGraph, ctrlGraphLastModif:int=None):
        self.graph = graph
        self.ctrlGraphLastModif = ctrlGraphLastModif
        self.initial = None
        # Computed path Cache dict of calculated route {Addr, list(Addr)}
        self.results = {}
        self.lastSource = ""
        self.lastModification = -1
        
    def getNode(self, id:str):        
        return self.graph.nodes[id]
    
    def setSource(self, src:str):
        self.initial = src
        
    def getResults(self):
        return self.results
                
class Dijkstra(Routing):

    def __init__(self, graph: nx.MultiDiGraph, ctrlGraphLastModif: int = None):
        super().__init__(graph, ctrlGraphLastModif)
        
    def dijkstra(self):
        visited = {self.initial: 0}
        path = {}

        nodes = set(self.graph.nodes(data=False))
        
        while nodes:
            min_node = None
            for node in nodes:
                if node in visited:
                    if min_node is None:
                        min_node = node
                    elif visited[node] < visited[min_node]:
                        min_node = node
            if min_node is None:
                break
            
            nodes.remove(min_node)
            current_weight = visited[min_node]
            for edge in self.graph.edges([min_node]):  
                key = edge[0]+'-'+edge[1]
                weight = self.graph.get_edge_data(edge[0], edge[1])[key]['weight']
                try:
                    weight = current_weight + weight
                except:
                    continue
                if edge[1] not in visited or weight < visited[edge[1]]:
                    visited[edge[1]] = weight
                    path[edge[1]] = min_node

        return visited, path

    def shortest_path(self, origin, destination):
        visited, paths = self.dijkstra()
        full_path = deque()
        _destination = paths[destination]

        while _destination != origin:
            full_path.appendleft(_destination)
            _destination = paths[_destination]

        full_path.appendleft(origin)
        full_path.append(destination)

        return visited[destination], list(full_path)
    
    def getRoute(self, src, dst):
        cost = 0
        p = []
        srcNode = None
        dstNode = None
        if src != dst:
            try:
                srcNode = self.getNode(src)
                dstNode = self.getNode(dst)
            except Exception as ex:
                print(ex)
            
            if not(srcNode is None and dstNode is None):
                try:
                    if self.lastSource != src or self.lastModification != self.ctrlGraphLastModif:
                        self.results.clear()
                        self.setSource(src)
                        self.lastSource = src
                        self.lastModification = self.ctrlGraphLastModif
                    else: p = self.results.get(src+'-'+dst)
                    
                    if not p:
                        cost, path = self.shortest_path(src, dst)
                        p = [Addr(p[2:]) for p in path]
                        self.results[src+'-'+dst] = p.reverse()
                except Exception as e:
                    print(e)
        return cost, p
    
##test
if __name__ == '__main__':
    import time

    graph = nx.MultiDiGraph()
    for node in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        graph.add_node(node)

    graph.add_edge('A', 'B', key="A-B", weight=10)
    graph.add_edge('A', 'C', key="A-C", weight=20)
    graph.add_edge('B', 'D', key="B-D", weight=15)
    graph.add_edge('C', 'D', key="C-D", weight=30)
    graph.add_edge('B', 'E', key="B-E", weight=50)
    graph.add_edge('D', 'E', key="D-E", weight=30)
    graph.add_edge('E', 'F', key="E-F", weight=5)
    graph.add_edge('F', 'G', key="F-G", weight=2)

    dj = Dijkstra(graph)
    dj.initial = 'A'
    _, v = dj.dijkstra()
    print(v)
    # cost, p = dj.getRoute('A', 'G')
    allroutes = nx.all_simple_paths(graph, source='A', target='G')
    print(list(allroutes))
    print(nx.to_numpy_matrix(graph))
    # print(p) # output: (25, ['A', 'B', 'D']) 
    # print(graph.edges.data(data=True, keys=True, nbunch='A'))
    # # print(list(graph.edges(['A'])))