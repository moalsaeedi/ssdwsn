#!/usr/bin/env python3
"""Example for aiohttp.web.Application.on_startup signal handler
"""

import aiohttp
import asyncio
import contextlib

from aiohttp.web import Application, run_app, Response


async def fake_redis_msg(request):
    return Response(text='Received fake Redis message...')


async def fake_zmq_msg(request):
    return Response(text='Received fake ZeroMQ message...')


async def get_fake_message(where):
    async with aiohttp.ClientSession(loop=app.loop) as session:
        res = await session.get('http://127.0.0.1:8080/{}'.format(where))
        return await res.text()


async def quick_1(app):
    for x in range(5):
        print('quck_1')


async def quick_2(app):
    await asyncio.sleep(0.05)
    for x in range(5):
        print('quck_2')


async def listen_to_redis(app):
    await asyncio.sleep(0.01)
    while True:
        print('Listening to Redis...')
        print(await get_fake_message('redis'))
        await asyncio.sleep(0.5)


async def listen_to_zeromq(app):
    await asyncio.sleep(0.01)
    while True:
        print('Listening to ZeroMQ...')
        print(await get_fake_message('zeromq'))
        await asyncio.sleep(0.5)

async def start_background_tasks(app):
    app['redis_listener'] = app.loop.create_task(listen_to_redis(app))
    app['zeromq_listener'] = app.loop.create_task(listen_to_zeromq(app))


async def cleanup_background_tasks(app):
    print('cleanup background tasks...')
    app['redis_listener'].cancel()
    app['zeromq_listener'].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await app['redis_listener']
        await app['zeromq_listener']
    print('cleanup background tasks done...')


async def init():
    app = Application()
    app.router.add_get('/redis', fake_redis_msg)
    app.router.add_get('/zeromq', fake_zmq_msg)
    app.on_startup.append(quick_1)
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    app.on_startup.append(quick_2)
    return app

loop = asyncio.get_event_loop()
app = loop.run_until_complete(init())
run_app(app)