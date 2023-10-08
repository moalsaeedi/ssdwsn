import asyncio
from typing import Counter
from aiohttp import web
from aiohttp.web_runner import GracefulExit
import aiofiles
from numpy import broadcast
import socketio
from time import sleep
from threading import Lock
import os
from sanic import Sanic
from sanic.response import html
from ssdwsn.util.constants import Constants as ct
import eventlet
import json
import time
from pathlib import Path

eventlet.monkey_patch()  

thr1 = None
# thr2 = None
thread_lock = Lock()

PORT = ct.SIM_PORT
count = 0


# create a new aysnc socket io server
# mgr = socketio.AsyncRedisManager('redis://')
sio = socketio.AsyncServer(async_mod='aiohttp', binary=True, cors_allowed_origins=[], ping_timeout=10, ping_interval=25) # eventlet
# sio = socketio.AsyncServer()

#create a new Aiohttp web application
app = web.Application()
#bind the socket.io server to the web application instance
sio.attach(app)

# sio = socketio.AsyncServer(async_mode='sanic', cors_allowed_origins=[])
# app = Sanic(name='sanic_application')
# app.config['CORS_SUPPORTS_CREDENTIALS'] = True
# sio.attach(app)

# async def background_task():
#     """Example of how to send server generated events to clients."""
#     count = 0
#     while True:
#         await sio.sleep(10)
#         count += 1
#         await sio.emit('my_response', {'data': 'Server generated event'})


# @app.listener('before_server_start')
# def before_server_start(sanic, loop):
#     # sio.start_background_task(background_task)
#     pass


async def index(request):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, './templates/index.html')
    # async with aiofiles.open(filename) as file_obj:
    with open(filename) as file_obj:
        return web.Response(text = file_obj.read(), content_type='text/html')

async def shutdown(request):
    await sio.disconnect()
    # app.shutdown()
    # await app.cleanup()
    raise GracefulExit()

# @app.route('/')
# async def index(request):
#     with open('./ssdwsn/outputs/') as f:
#         return html(f.read())

async def getgraph():
    while True:
        await sio.sleep(5)
        await sio.emit('sr_getgraph', namespace='/ctrl')
        
async def getstate():
    while True:
        # await sio.emit('sr_getenergy')
        await sio.sleep(2)
        # await sio.emit('sr_getstate')
        
async def saveGraph(data):
    async with aiofiles.open('outputs/graph.json', mode='w') as f:
        await f.write(json.dumps(data))
                
# @sio.event
# async def connect(sid, environ):  
    # global thr1
    # with thread_lock:
    #     if thr1 is None:
    #         thr1 = sio.start_background_task(getgraph)
    #         # await sio.emit('none')
       
@sio.event
async def disconnect(sid):
    print('[%s]: disconnected' % sid)
            
@sio.event(namespace='/ctrl')
async def showgraph(sid, data):
    await sio.emit('sr_showgraph', data)
    # await saveGraph(data)
    # await sio.emit('sr_checkgraph', broadcast=True)
    return 'Graph Re-Rendered!'

@sio.event
async def start(sid, data):
    await sio.emit('sr_start', data, broadcast=True)
    global thr1
    # global thr2
    with thread_lock:
        if thr1 is None:
            thr1 = sio.start_background_task(getgraph)
        # if thr2 is None:
        #     thr2 = sio.start_background_task(getstate)
    return 'Start Simulation!'

@sio.event
async def pause(sid):
    await sio.emit('sr_pause', broadcast=True)
    return 'Pause Simulation!'

@sio.event
async def resume(sid):
    await sio.emit('sr_resume', broadcast=True)
    return 'Resume Simulation!'

@sio.event
async def reset(sid):
    await sio.emit('sr_reset', broadcast=True)
    return 'Reset Simulation!'

@sio.event
async def stop(sid):
    await sio.emit('sr_stop', broadcast=True)
    return 'Stop Simulation!'

@sio.event
async def rxpacketin(sid):
    await sio.emit('sr_rxpacketin')

@sio.event
async def txpacketout(sid):
    await sio.emit('sr_txpacketout')

@sio.event
async def removenode(sid, data):
    await sio.emit('sr_removenode', data, namespace='/ctrl')
    
@sio.event
async def txpacket(sid, data):
    await sio.emit('sr_txpacket', data)
  
@sio.event
async def rxpacket(sid, data):
    await sio.emit('sr_rxpacket', data)
    # await sio.emit('sr_rxpacket', data, namespace='/ctrl')
  
@sio.event
async def ndtraffic(sid, data):
    await sio.emit('sr_ndtraffic', data, namespace='/ctrl')

@sio.event
async def droppacket(sid, data):
    await sio.emit('sr_droppacket', data)

@sio.event
async def txbeacon(sid, data):
    await sio.emit('sr_txbeacon', data)

@sio.event
async def getresources(sid):
    await sio.emit('sr_getresources')

@sio.event
async def resources(sid, data):
    await sio.emit('sr_resources', data, broadcast=True)

@sio.event
async def setstate(sid, data):
    await sio.emit('sr_setstate', data, namespace='/ctrl')

@sio.event
async def showstats(sid, data):
    await sio.emit('sr_showstats', data)

@sio.event
async def linkcolor(sid, data):
    await sio.emit('sr_linkcolor', data)

@sio.event
async def nodecolor(sid, data):
    await sio.emit('sr_nodecolor', data, namespace='/ctrl')

@sio.event
async def udneighs(sid, data):
    await sio.emit('sr_udneighs', data['data'], namespace=data['ns'])
    # await sio.emit('sr_udneighs', data, broadcast=True)

@sio.event
async def savetopo(sid, data):
    filename = data['filename']
    data.pop('filename')
    data = json.dumps(data)
    async with aiofiles.open('outputs/topo/'+str(filename)+'.json', mode='w') as f:
        await f.write(data)

@sio.event
async def opentopo(sid, data):
    filename = data['openfilename']
    # data = json.dumps(data)
    async with aiofiles.open('outputs/topo/'+str(filename)) as f:
        data = json.loads(await f.read())
        await sio.emit('sr_opentopo', data)
                
    # async with aiofiles.open('outputs/topo/topo.json') as f:
    #     return json.loads(await f.read())
@sio.event
async def gettopofiles(sid):
    data  = [x for x in os.listdir('./outputs/topo/') if x.endswith('.json')]
    await sio.emit('sr_gettopofiles', data)

app.router.add_static('/static/', path='./ssdwsn/util/plot/static/')
app.router.add_get('/',index)
app.router.add_get('/shutdown/',shutdown)

# app.static('/static', './ssdwsn/util/plot/static')
# start the server
if __name__ == '__main__':    
    web.run_app(app, port=PORT)
    # app.run(port=PORT, debug=True, access_log=True)