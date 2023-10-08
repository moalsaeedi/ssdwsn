#!/usr/bin/env python3
"""Example for aiohttp.web.Application.on_startup signal handler
"""

import aiohttp
from aiohttp import web
import asyncio
import contextlib
import os
from aiohttp.web import Application, run_app, Response
from ssdwsn.util.constants import Constants as ct

PORT = ct.SIM_PORT

async def index(request):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, './templates/index.html')
    # async with aiofiles.open(filename) as file_obj:
    with open(filename) as file_obj:
        return web.Response(text = file_obj.read(), content_type='text/html')

async def get_message(where):
    async with aiohttp.ClientSession(loop=app.loop) as session:
        res = await session.get('http://127.0.0.1:4455/{}'.format(where))
        # async for msg in res.iter(encoding='utf-8'):
        #     # Forward message to all connected websockets:
        #     for ws in app['websockets']:
        #         ws.send_str('{}: {}'.format(res.name, msg))
        #     print("message in {}: {}".format(res.name, msg))
        return await res.text()

async def quick_1(app):
    for x in range(5):
        print('quck_1')

async def quick_2(app):
    await asyncio.sleep(0.05)
    for x in range(5):
        print('quck_2')

async def listen_to_zeromq(app):
    try:
        await asyncio.sleep(0.01)
        while True:
            print('Listening to ZeroMQ...')
            print(await get_message('showgraph'))
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    finally:
        print('Cancel Redis listener: close connection...')
        await cleanup_background_tasks(app)
        print('Zmq connection closed.')

async def start_background_tasks(app):
    app['zeromq_listener'] = app.loop.create_task(listen_to_zeromq(app))


async def cleanup_background_tasks(app):
    print('cleanup background tasks...')
    app['zeromq_listener'].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await app['zeromq_listener']
    print('cleanup background tasks done...')


async def init():
    app = Application()
    app.router.add_static('/static/', path='./ssdwsn/util/plot/static/')
    app.router.add_get('/', index)
    app.on_startup.append(quick_1)
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    app.on_startup.append(quick_2)
    return app

loop = asyncio.get_event_loop()
app = loop.run_until_complete(init())
run_app(app, port=PORT)