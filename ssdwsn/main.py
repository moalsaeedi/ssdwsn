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

from asyncore import loop
from random import randint
from optparse import OptionParser
from os import environ, kill, getpid, system
from os.path import join as path_join
from sys import path, stdout, exc_info, version_info as py_version_info, stderr

if 'PYTHONPATH' in environ:
    path = environ[ 'PYTHONPATH' ].split(':') + path

import time
import subprocess
from ssdwsn.data.node import Mote, Sink
from ssdwsn.data.addr import Addr
# from ssdwsn.util.log import info, error, debug, output, warn
from ssdwsn.ctrl.graph import Graph
from ssdwsn.ctrl.controller import ATCP_ctrl, NSFP_ctrl
from ssdwsn.util.constants import Constants as ct
from ssdwsn.util.utils import CustomFormatter, customClass, mapRSSI, runCmd
from ssdwsn.data.neighbor import Neighbor
# from ssdwsn.util.log import lg, LEVELS, info, debug, warn, error, output
import networkx as nx
from pathlib import Path
import json
import asyncio, logging
import aiohttp
import re
from subprocess import Popen, PIPE, check_output
from networkx.readwrite import json_graph
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import signal
import requests
import socketio
import psutil
from ssdwsn.data.sensor import SensorType, Temperature, Sound, Camera, Humidity, Pressure, SwitchOnOff, MythingsIoT
from ssdwsn.data.intf import SixLowPan
from ssdwsn.data.prop import LogNormalShadowing, Friis, ITU, TowRayGround
from ssdwsn.data.intf import IntfType
from functools import wraps


from asyncio.proactor_events import _ProactorBasePipeTransport
 
def silence_event_loop_closed(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RuntimeError as e:
            if str(e) != 'Event loop is closed':
                raise
    return wrapper
 
_ProactorBasePipeTransport.__del__ = silence_event_loop_closed(_ProactorBasePipeTransport.__del__)

#logging----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
#-----------------------------------

# loop = asyncio.new_event_loop()
sio = socketio.AsyncClient(
        ssl_verify=False, 
        reconnection=True) #logger=True, engineio_logger=True

requests.adapters.DEFAULT_RETRIES = 5

VERSION = "1.0"

def version(*_args):
    "Print ssdwsn version and exit"
    logging.info( "%s\n" % VERSION )
    exit()
    
# TOPOS = { 'minimal': MinimalTopo,
#           'linear': LinearTopo,
#           'reversed': SingleSwitchReversedTopo,
#           'single': SingleSwitchTopo,
#           'tree': TreeTopo,
#           'torus': TorusTopo }
"""
CONSTANTS
"""
NEXT_MAC_COUNT = 0
NEXT_DPID_COUNT = 0
TASKS = []

MOTES = {'mote':Mote,
}

SINKS = {'sink':Sink,
}

CONTROLLERS = {
    'NSFP-ctrl': NSFP_ctrl,
    'ATCP-ctrl': ATCP_ctrl
}
SENSORS = {
    'temperature': Temperature, 
    'sound': Sound, 
    'camera': Camera, 
    'humidity': Humidity, 
    'pressure': Pressure, 
    'switchonoff': SwitchOnOff, 
    'mythingiot': MythingsIoT
}

PPMS = {
    'logNormalShadowing': LogNormalShadowing,
    'friis': Friis,
    'ITU': ITU,
    'twoRayGround': TowRayGround
}

INTFS = {
    'sixlowpan': SixLowPan
}

TESTS = { name: True
          for name in ( 'pingall', 'pingpair', 'iperf', 'iperfudp' ) }

CLI = None  # Set below if needed

RUNNING_PS = []

# # Locally defined tests
# def allTest( net ):
#     "Run ping and iperf tests"
#     net.waitConnected()
#     net.start()
#     net.ping()
#     net.iperf()

# def nullTest( _net ):
#     "Null 'test' (does nothing)"
#     pass

# TESTS.update( all=allTest, none=nullTest, build=nullTest )

# # Map to alternate spellings of Mininet() methods
# ALTSPELLING = { 'pingall': 'pingAll', 'pingpair': 'pingPair',
#                 'iperfudp': 'iperfUdp' }

# def runTests( mn, options ):
#     """Run tests
#        mn: Mininet object
#        option: list of test optinos """
#     # Split option into test name and parameters
#     for option in options:
#         # Multiple tests may be separated by '+' for now
#         for test in option.split( '+' ):
#             test, args, kwargs = splitArgs( test )
#             test = ALTSPELLING.get( test.lower(), test )
#             testfn = TESTS.get( test, test )
#             if callable( testfn ):
#                 testfn( mn, *args, **kwargs )
#             elif hasattr( mn, test ):
#                 mn.waitConnected()
#                 getattr( mn, test )( *args, **kwargs )
#             else:
#                 raise Exception( 'Test %s is unknown - please specify one of '
#                                  '%s ' % ( test, TESTS.keys() ) )

def isLoaded(module):
    mdl = subprocess.Popen(['lsmod'], stdout=subprocess.PIPE)
    grep= subprocess.Popen(['grep', module],
                                    stdin=mdl.stdout, stdout=subprocess.PIPE)
    grep.communicate() 
    return grep.returncode == 0

def get_intfs(cmd):
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

def reset():    
    logger.info('Reset ...\n')
    pass

def cleanUp():         
    logger.info("Shutting down and cleaning up...") 
    # kill(getpid(), signal.SIGTERM)
    # cmd="kill -9 $(ps -A | grep redis | awk '{print $1}')"
    # runCmd(cmd)
    wpan_list = get_intfs("iwpan dev 2>&1 | grep Interface | awk '{print $2}'")
    for wpan in wpan_list:
        try:            
            # p = runCmd(f'ip link set dev {wpan} down')
            # p.terminate()
            system(f'iwpan dev {wpan} del')
            # p = runCmd(f'iwpan dev {wpan} del')
            # p.terminate()
        except:
            print('delete interface error...')
    try:
        system("ip -all netns del")
    except: pass
    if isLoaded('mac802154_hwsim'):
        system("rmmod mac802154_hwsim")
        # p = runCmd(cmd)    
        # p.terminate() 

    try:
        system("pkill -9 python3 | kill -9 $(ps -A | grep python | awk '{print $1}')")
        # cmd="pkill -9 python3 | kill -9 $(ps -A | grep python | awk '{print $1}')"
        # p = runCmd(cmd)    
        # p.terminate()
    except:
        pass
                
async def shutdownHttpServer():
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.get(url=ct.SIM_URL+'/shutdown') as response:
            await response.text()
    return logger.info(response.status)
          
def addDictOption(opts, choicesDict, default, name, **kwargs):
    """Convenience function to add choices dicts to OptionParser.
    opts: OptionParser instance
    choicesDict: dictionary of valid choices, must include default
    default: default choice key
    name: long option name
    kwargs: additional arguments to add_option"""
    helpStr = ('|'.join(sorted(choicesDict.keys())) +
            '[,param=value...]')
    helpList = [ '%s=%s' % (k, v.__name__)
                for k, v in choicesDict.items() ]
    helpStr += ' ' + (' '.join(helpList))
    params = dict(type='string', default=default, help=helpStr)
    params.update(**kwargs)
    opts.add_option('--' + name, **params)    

def parseArgs():
    """Parse Command-Line"""
    desc = ("The %prog utility creates ssdwsn network from the\n"
            "command line. It can create parametrized topologies,\n"
            "invoke the ssdwsn CLI, and run tests.")

    usage = ('%prog [options]\n'
                '(type %prog -h for details)')
    
    opts = OptionParser(description=desc, usage=usage)
    # addDictOption(opts, MOTES, 'mote', 'mote')
    # addDictOption(opts, SINKS, 'sink', 'sink')
    # addDictOption(opts, CONTROLLERS, [], 'controller', action='append')
    # addDictOption(opts, TOPOS, 'minimal', 'topo')

    # opts.add_option('--ssdwsn', '-s', action='store_true',
    #                 default=False, help='activate ssdwsn')   
    opts.add_option('-c', '--config', type=str, default=ct.CONFIG_FILE, help='Config File')     
    # opts.add_option('-m', '--mote', nargs=2, action='append', default=[], help='Add mote <pos rssi> ex: -m 0,0,0 180')
    # opts.add_option('-s', '--sink', nargs=4, action='append', default=[], help='Add sink <pos rssi ctrAddr ctrlPort> ex: -s 0,0,0 180 127.0.0.1 9990')
    # opts.add_option('-c', '--ctrl', nargs=3, action='append', default=[], help='Ctrl type and addr <type ip port> ex: -c dctrl 127.0.0.1 9990')        
    opts.add_option('-q', '--quit', action='store_true', default=False, help='quit and exit')
    # opts.add_option('-v', '--verbosity', type='choice',
    #                 choices=list(LEVELS.keys()), default='info',
    #                 help='|'.join(list(LEVELS.keys())))
    opts.add_option('--version', action='callback', callback=version,
                    help='prints the version and exits')
    
    options, args = opts.parse_args()
    if args:
        opts.print_help()
        exit()
    return options, args

def configNeighbors(nd, graph):
        """Wireless access medum"""
        """Wireless node can access the signal of nodes in its transmition range"""
        """This function is necessary to identify node's neighbors"""
        try:
            modified = False
            ngTable = [x for x in nd.neighborTable]
            toAdd = {}
            toRemove = []
            inRange = []
            # async with aiohttp.ClientSession(trust_env=True) as session:
            #     async with session.post(url=ct.SIM_URL+'/getgraph') as response:
                    # graph = await response.json()  
            for node in graph["nodes"]:
                node.update(IntfType.getParams(node['intftype']))
                if nd.isInRange(node) and node['id'] != nd.id: 
                    rssi = mapRSSI(nd.getRssi(node)) 
                    inRange.append(node['id'])
                    if  node['id'] in ngTable and rssi == nd.getNeighborTable().get(node['id']).getRssi():
                        continue
                    addr = '{}.{}'.format(node['id'].split('.')[1], node['id'].split('.')[2])
                    toAdd[node['id']] = Neighbor(nd.getDist(), Addr(addr), rssi, node["batt"], int(node["port"]))
                    modified = True
            for node in ngTable:
                if node not in inRange:
                    toRemove.append(node)
                    modified = True
                    # self.lastUpdatedGraphTimeStamp = dir_path.stat().st_mtime
            if modified:
                for key in toRemove:
                    nd.getNeighborTable().pop(key)
                for key in toAdd:
                    if key in ngTable:
                        nd.getNeighborTable().pop(key)
                    nd.getNeighborTable()[key] = toAdd.get(key)
            # print(f'Node: {nd.id} has Neighbors: {nd.getNeighborTable().keys()}\n')
        except Exception:
            logger.warn(f"Graph is empty ...\n")


def handler(signum, frame):
    kill(getpid(), signal.SIGTERM)
    exit(0)

def init():    
    signal.signal(signal.SIGINT, handler)

async def begin(data):
    global TASKS
    topofilename = data['topofilename']
    nodes = data['nodes']
    ctrls = data['controllers']
    settings = data['settings']
    networkGraph = Graph()
    logger.info(f'SIMULATION CONTROLLERS:\n {ctrls}')
    logger.info(f'SIMULATION SETTINGS:\n {settings}\n')
    ct.FT_MAX_ENTRIES = int(settings['ftsize'])
    ct.RL_IDLE = int(settings['idletimeout'])
    ct.BUFFER_SIZE = int(settings['buffersize'])
    ct.MAX_DELAY = int(settings['maxdelay'])
    ct.SIM_TIME = int(settings['simtime'])
    ct.BE_TTI = int(settings['beaconstti'])
    ct.RP_TTI = int(settings['reportstti'])
    ct.RANDOM_SEED = int(settings['randomseed'])
    networkGraph.mxNodes = int(settings['mxnodes'])
    networkGraph.gridWidth = int(settings['gridwidth'])
    networkGraph.gridHeight = int(settings['gridheight'])
    system("modprobe mac802154_hwsim") 
    snds = [] 

    # Remove any previous configured wpan interface and initiate new ones
    for node in nodes:
        snds.append(node['port'])
    snds = set(snds)
    for nd in list(snds):
        phy = check_output(f"iwpan dev | grep -B 1 wpan{nd} | sed -ne 's/wpan\([^ ]\)/\1/p'",
            shell=True, text=True)
        phy = re.findall(r'[0-9]+', phy)
        try:
            if phy:
                check_output(f"iwpan dev wpan{phy[0]} del",
                shell=True, text=True)
            if nd >= 1:
                system("wpan-hwsim add >/dev/null 2>&1") 
        except:
            pass
        
    # Initiate nodes as processes
    for node in nodes:
        datatype = SensorType.sink.name if node['issink'] else node['datatype']
        id = networkGraph.addNode(f"{node['net']}.{node['addr']}")
        intf = IntfType.getParams(IntfType.sixlowpan.value)
        
        networkGraph.setupNode(id, node['net'], pos=(node['pos'][0], node['pos'][1]), color=node['color'], intftype=node['intftype'], datatype=datatype, batt=node['batt'], 
            port=ct.BASE_NODE_PORT+node['port'], antrange=intf['antRange'], distance=node['distance'], denisty=node['denisty'], lastupdate=round(time.time(), 4), active=False)
                         
    expool = ProcessPoolExecutor(max_workers=int(settings['mxnodes']))  
    # expool = ThreadPoolExecutor(max_workers=int(settings['mxnodes']))  
    loop = asyncio.get_running_loop()
    
    for ctrl in ctrls:   
        TASKS.append(loop.run_in_executor(expool, run_loop_in_process, 'ctrl',ctrl, settings, topofilename, networkGraph))

    for node in nodes: 
        if node['issink']:            
            TASKS.append(loop.run_in_executor(expool, run_loop_in_process, 'sink', node, settings, topofilename))
        else:
            TASKS.append(loop.run_in_executor(expool, run_loop_in_process, 'mote', node, settings, topofilename))

    try:    
        await asyncio.gather(*TASKS)
    except:
        for process in expool._processes.values():
            process.kill()

def run_loop_in_process(nodetype, node, settings, topofilename, networkGraph=None):    
    async def subprocess_async_work():
        global NEXT_MAC_COUNT
        global NEXT_DPID_COUNT
        # panid=0xbeef
        if nodetype == 'ctrl':
            controller = customClass(CONTROLLERS, node["ctrltype"])(inetAddress=(node["ip"], ct.BASE_CTRL_PORT+node["port"]), networkGraph=networkGraph)  
            controller.ppm = customClass(PPMS, settings['ppm'])()
            RUNNING_PS.append(getpid())
            # jsondata = json_graph.node_link_data(networkGraph.getGraph())
            # print(jsondata)
            await controller.run()
        else:
            node['pos'].append(0)                  
            mac = "{:012X}".format(ct.BASE_MAC+NEXT_MAC_COUNT)         
            new_mac = ':'.join(mac[i]+mac[i+1] for i in range(0, len(mac), 2))
            node_pos = (*[int(x) for x in node['pos']],) 
            # node['pos'].append(0)
            NEXT_MAC_COUNT +=1
            if nodetype == 'mote':
                # new_mote_addr = moteAddr(stats["NEXT_MOTE_ADDR_C"])  
                #TODO update Addr class to include the subnet id (net) and remove the split from target this will required to update packets structure
                target = node['target'][2:]
                mote = Mote(net=node['net'], addr=Addr(node['addr']), portSeq=node['port'], batt=node['batt'], pos=node_pos, target=target, \
                    topo=topofilename, ctrl=(node["ctrl"].split(':')[0], ct.BASE_CTRL_PORT+int(node["ctrl"].split(':')[1])), cls=SENSORS[node['datatype']] )
                mote.wintf = customClass(INTFS, node['intftype'])(mote.id, mote.ip, port=ct.BASE_NODE_PORT+node['port'], mac=new_mac)
                logger.info(mote.wintf)
                # system("wpan-hwsim add >/dev/null 2>&1")
                mote.ppm = customClass(PPMS, settings['ppm'])() 
                RUNNING_PS.append(getpid())
                # jsondata = json_graph.node_link_data(networkGraph.getGraph())
                # print(jsondata)
                # configNeighbors(mote, data)
                await mote.run()
            if nodetype == 'sink':
                new_sink_dpid = "{:08b}".format(ct.BASE_DPID + NEXT_DPID_COUNT) 
                sink = Sink(net=node['net'], addr=Addr(node['addr']), portSeq=node['port'], dpid=new_sink_dpid, pos=node_pos, \
                    topo=topofilename, ctrl=(node["ctrl"].split(':')[0], ct.BASE_CTRL_PORT+int(node["ctrl"].split(':')[1])))
                sink.wintf = customClass(INTFS, node['intftype'])(sink.id, sink.ip, port=ct.BASE_NODE_PORT+node['port'], mac=new_mac)
                logger.info(sink.wintf)
                sink.ppm = customClass(PPMS, settings['ppm'])()   
    
                # system("wpan-hwsim add >/dev/null 2>&1")   
                RUNNING_PS.append(getpid())
                # jsondata = json_graph.node_link_data(networkGraph.getGraph())
                # print(jsondata)
                NEXT_DPID_COUNT += 1
                # configNeighbors(sink, data)
                await sink.run()

    asyncio.run(subprocess_async_work())

def initGraph():
    # cmd='nohup python3 {} &'.format(ct.SIM_APP)  
    # cmd='python3 {} asgi &'.format(ct.SIM_APP)  
    cmd='python3 {} &'.format(ct.SIM_APP)  
    process = runCmd(cmd)
    # RUNNING_PS.append(process) 
    
async def connect():
    initGraph()
    await asyncio.sleep(2)
    sio.on("connect", _handle_connect)
    await sio.connect(ct.SIM_URL)
    # await asyncio.sleep(2)
    # while not sio.connected:
    #     time.sleep(0.025)
    txt = "\n \
         ___________ _      _______  __  \n \
        / __/ __/ _ \ | /| / / __/ |/ /  \n \
       _\ \_\ \/ // / |/ |/ /\ \/    /   \n \
      /___/___/____/|__/|__/___/_/|_/    \n \
    Click Here!"
    target = f"http://localhost:{ct.SIM_PORT}"
    logger.info(f'Simulation is running ...\n \u001b]8;;{target}\u001b\\{txt}\u001b]8;;\u001b\\\n')
    # await sio.wait()
    while True:
        await asyncio.sleep(5)

async def _handle_connect():
    """send a request to the visualizer to visualize the graph (network view)
    """
    global sio
    sio.on('sr_getresources', _handle_getresources)
    sio.on('sr_start', _handle_start)    
    sio.on('sr_stop', _handle_stop)
    # await sio.wait()
    print('MAIN is connected to socketio server .....')
            
async def _handle_getresources():
    global sio
    if sio.connected:  
        ctrls = [ctrl for ctrl in CONTROLLERS]      
        sensors = [sensor for sensor in SENSORS]
        ppms = [ppm for ppm in PPMS]
        intfs = [intf for intf in INTFS]
        data = json.dumps({"controllers":ctrls, 
                           "sensors":sensors, 
                           "ppms":ppms, 
                           "intfs":intfs,
                           "ftsize": ct.FT_MAX_ENTRIES,
                           "idletimeout": ct.RL_IDLE,
                           "buffersize": ct.BUFFER_SIZE,
                           "maxdelay": ct.MAX_DELAY,
                           "simtime": ct.SIM_TIME,
                           "beaconstti": ct.BE_TTI,
                           "reportstti": ct.RP_TTI,
                           "randomseed": ct.RANDOM_SEED,
                           "mxnodes":ct.MAX_NODES, 
                           "mnport":ct.BASE_NODE_PORT, 
                           "mxport":ct.TOP_MOTE_PORT
                           })
        await sio.emit('resources', data)
        
async def _handle_start(data):    
    await begin(data)

async def _handle_stop():
    global stopTime
    stopTime = time.time()  
    await asyncio.sleep(0)
    raise KeyboardInterrupt
    # cleanUp()

async def main():
    try:
        await connect()        
    except:    
        # try:
        #     await sio.disconnect()
        # except:
        #     pass
        # try:
        #     await shutdownHttpServer()
        # except:
        #     pass
        pass

def exit_gracefully(*args):
    pass

if __name__ == '__main__':
    
    logger.info('*********** start ************')
    options, args = parseArgs()
    signal.signal(signal.SIGINT, exit_gracefully)
    # if options:
    #     setup(options)
    # configNeigborTable(nodes, topo.getGraph())
    # asyncio.ensure_future(bug())
    # loop.create_task(connect())  
    # loop.run_forever()
    try:
        asyncio.run(main())
    except:
        cleanUp()
    finally:
        for pr in RUNNING_PS:
            try:
                kill(pr, signal.SIGINT)
                # psutil.Process(node.pid).terminate()            
            except:
                pass
                
        # session = requests.Session()
        # retry = Retry(connect=3, backoff_factor=0.5)
        # adapter = HTTPAdapter(max_retries=retry)
        # session.mount('http://', adapter)
        # session.mount('https://', adapter)
        # try:
        #     session.get(ct.SIM_URL+'/shutdown')
        # except:
        #     info("Shutdown ... " )
    exit(1)