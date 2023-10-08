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

import asyncio, logging
import time
import random
import wave
from os import urandom
from ssdwsn.util.constants import Constants as ct
from ssdwsn.util.utils import CustomFormatter
from ssdwsn.openflow.packet import DataPacket
from enum import Enum

#logging----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
#-----------------------------------

class SensorType(Enum):
    """Sensor type

    Args:
        Enum (humidity): humidity sensor
        Enum (sink): sink (gateway) node without sensor
        Enum (pressure): pressure sensor
        Enum (temperature): temperature sensor
        Enum (switchonoff): switch on/off sensor
        Enum (sound): sound receiver sensor
        Enum (camera): camera sensor
        Enum (mythingsiot): mulit-purpose sensor

    Returns:
        value: int sensor value
        name: name of the sensor
    """
    sink = 0
    humidity = 1
    pressure = 2
    temperature = 3
    switchonoff = 4
    sound = 5
    camera = 6
    mythingsiot = 7
    
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
    
    @staticmethod
    def fromStringToInt(str):
        switcher = {
            SensorType.humidity.name: SensorType.humidity.value,
            SensorType.sink.name: SensorType.sink.value,
            SensorType.pressure.name: SensorType.pressure.value,
            SensorType.temperature.name: SensorType.temperature.value,
            SensorType.switchonoff.name: SensorType.switchonoff.value,
            SensorType.sound.name: SensorType.sound.value,
            SensorType.camera.name: SensorType.camera.value,
            SensorType.mythingsiot.name: SensorType.mythingsiot.value
        }
        return switcher.get(str, -1)
    
class Sensor:
    """Abstract Sensor Object"""
    def __init__(self, sensorType:SensorType, node):
        """Initiate a sensor object

        Args:
            loop (AbstractEventLoop): asyncio event loop of the parent process
            sensorType (SensorType): sensor enum type
        """
        self.type = sensorType.getValue()  
        self.loop = asyncio.get_running_loop()
        # asyncio Queue consists of tuple of sensing data, seq No and time
        # self.sensingQueue = asyncio.Queue(ct.BUFFER_SIZE, loop=self.loop)
        self.node = node
        # random.seed(ct.RANDOM_SEED)
        random.seed(time.time())

    def random_delta(size, start_value, min_delta, max_delta, precision=1):
        """
        * Generate rando data
        * Ex: list(random_delta(200, 10, -1, 1))
        """
        value_range = max_delta - min_delta
        half_range = value_range / 2
        for _ in range(size):
            delta = random.random() * value_range - half_range
            yield round(start_value, precision)
            start_value += delta
    
    def getType(self):
        self.sensorType.getValue()
        
class Humidity(Sensor):
    """Humidity Sensor"""
    def __init__(self):
        """Initiate a humidity sensor

        Args:
            loop (AbstractEventLoop): asyncio event loop of the parent process
        """
        super().__init__(sensorType=SensorType.humidity)
        
class Pressure(Sensor):
    """Pressure Sensor"""
    def __init__(self):
        """Initiate a pressure sensor

        Args:
            loop (AbstractEventLoop): asyncio event loop of the parent process
        """
        super().__init__(sensorType=SensorType.pressure)
        
class Temperature(Sensor):
    """Temperature Sensor"""
    def __init__(self, node):
        """Initiate a temperature sensor
        - val: is a temperature values is a normal distribution of a mean (mu) of a 
        uniform distribution of random degree values between -30C and 50C and sigma = 10
        - timeout: inter transsmition time is 5 seconds (timeout = 5.0 sec)
        Args:
            loop (AbstractEventLoop): asyncio event loop of the parent process
        """
        super().__init__(sensorType=SensorType.temperature, node=node)
        self.meanTemp = random.uniform(-30, 50)
        
    def run(self):
        async def async_thread():
            sqNo = 0
            while True:
                val= str(round(random.normalvariate(self.meanTemp, 10),1))+" C"
                timeout= 5
                timeout= float(random.uniform(5,15))
                data = bytearray((val+' sq'+str(sqNo)).encode()) # adding seq to the payload to get a uniqe hashed value for calculating the delay of received data packet.
                try:
                    self.node.sensingQueue.put_nowait((data, sqNo, time.time()))
                except asyncio.QueueFull as e:
                    logger.warn(e)
                await asyncio.sleep(timeout)
                sqNo += 1
        asyncio.run(async_thread())
        
# class Proximity(Sensor):
     
#      def __init__(self):
#          super().__init__()
         
# class Level(Sensor):
    
#     def __init__(self):
#         super().__init__()        
        
# class Accelerometer(Sensor):
    
#     def __init__(self):
#         super().__init__()
        
# class Optical(Sensor):
    
#     def __init__(self):
#         super().__init__()
        
class SwitchOnOff(Sensor):
    """Switch on/off Sensor"""
    def __init__(self):
        """Initiate an on/off switch sensor
        - Generated sensing data is a uniform distribution 
        over two values OFF/ON
        - inter transsmition time is of uniform distribution between 0.1 and 15 seconds
        Args:
            loop (AbstractEventLoop): asyncio event loop of the parent process
        """
        super().__init__(sensorType=SensorType.switchonoff)
        
    def run(self):
        async def async_thread():
            sqNo = 0      
            while True:
                val=random.choice(["OFF","ON"])
                timeout= float(random.uniform(0.1,15)) 
                data = bytearray((val+' sq'+str(sqNo)).encode()) # adding seq to the payload to get a uniqe hashed value for calculating the delay of received data packet.
                try:
                    self.node.sensingQueue.put_nowait((data, sqNo, time.time()))
                except asyncio.QueueFull as e:
                    logger.warn(e)
                await asyncio.sleep(timeout)
                sqNo += 1
        asyncio.run(async_thread())

class Sound(Sensor):
    """Ambient Sound Sensor"""
    def __init__(self):
        """Generate sound traffic 
        sound specification:
        48 KHz 16 bit stereo
        sps: samples per seconds
        spts: number of frames aggregated in each timestamp (spts = framerate/sps)
        waveData: list of sound waves
        waveFile: sample sound (nchannels=2, sampwidth=2, framerate=48000, nframes=2880000, comptype='NONE', compname='not compressed')
        timeout: inter transmition time (packetization) = 1/sps

        Args:
            loop (AbstractEventLoop): asyncio event loop of the parent process
        """
        super().__init__(sensorType=SensorType.sound)
        
    def run(self):
        async def async_thread():
            waveData = []
            sps = 100 #1/sps = packetization ts
            spts = 1
            asdWrapLimit = 1
            
            waveFile = wave.open('ssdwsn/util/outputs/sample.wav', 'rb')
            params = waveFile.getparams()
            frameRate = float(params[2])
            frames = waveFile.getnframes()
            spts = frameRate/sps
            asdWrapLimit = int(params[3]/spts)
            for _ in range(0,asdWrapLimit):
                data = waveFile.readframes(int(spts))
                waveData.append(data)
            sqNo = 0
            while True:
                val = waveData[i%asdWrapLimit]
                timeout=float(1.0/sps)
                data = bytearray(val)
                try:
                    self.node.sensingQueue.put_nowait((data, sqNo, time.time()))
                except asyncio.QueueFull as e:
                    logger.warn(e)
                await asyncio.sleep(timeout)
                sqNo += 1
        asyncio.run(async_thread())
            
class Camera(Sensor):
    """Camera Sensor"""
    def __init__(self):
        """Generate surveillance video traffic
        - Video is generated when sensing a motion. 
        - The motion is a binomial random variable of two possible values [0,1].
        - Generating random bytes to simulate MPEG2 video payload. In MPEG2 all frames are equal sized
        - motionTime: is a random variable of uniform distribution for values between 1sec and 5sec
        - stopTime: is the current time plus the motionTime
        - fps: frames per second
        - bitRate: is value from a uniform distribution for values between 50 kbps to 200 kbps (low quality (<64 kbps))
        - timeout: inter transmition time (is random variable of uniform distribution for values between 1sec and 10sec and represented as the time between consequtive motion stream recording)
        Args:
            loop (AbstractEventLoop): asyncio event loop of the parent process
        """
        super().__init__(sensorType=SensorType.camera)
        
    def run(self):
        async def async_thread():
            sqNo = 0
            while True:
                motion=random.choice([0,1])
                if (motion):
                    fps=15
                    bitRate= int(random.uniform(50000, 200000))
                    motionTime=float(random.uniform(1,5))
                    stopTime=time.time()+motionTime
                    while time.time() < stopTime:
                        val= urandom(int(bitRate/8/fps))
                        timeout=float(1.0/fps)
                        data = bytearray(val)
                        try:
                            self.node.sensingQueue.put_nowait((data, sqNo, time.time()))
                        except asyncio.QueueFull as e:
                            logger.warn(e)
                        await asyncio.sleep(timeout)
                else:
                    #no motion, no data, sleep random time
                    val="NO_MOTION"
                    timeout=float(random.uniform(1,10))
                    await asyncio.sleep(timeout)
                sqNo += 1
        asyncio.run(async_thread())
         
class MythingsIoT(Sensor):
    """
    The MYTHINGS Smart Sensor is a self-contained, battery-powered multi-purpose IoT sensor 
    that allows you to capture critical data points like acceleration, temperature, humidity, 
    pressure and GPS. The smart sensor is integrated with the MYTHINGS Library – a hardware independent, 
    small-footprint and power-optimized library of code, featuring the MIOTY (TS-UNB) low-power wide area 
    """
    def __init__(self):
        super().__init__(sensorType=SensorType.mythingsiot)
        
if __name__== "__main__" :
        waveData = []
        sps = 100 #1/sps = packetization ts
        spts = 1
        asdWrapLimit = 1
        
        waveFile = wave.open('ssdwsn/util/outputs/sample.wav', 'rb')
        params = waveFile.getparams()
        frameRate = float(params[2])
        frames = waveFile.getnframes()
        spts = frameRate/sps
        asdWrapLimit = int(params[3]/spts)
        for _ in range(0,asdWrapLimit):
            data = waveFile.readframes(int(spts))
            waveData.append(data)
        seq = 0
        i = 0
        while True:
            val = waveData[i%asdWrapLimit]
            i +=1
            timeout=float(1.0/sps)
            data = bytearray(val)
            # print(timeout)
            # print(data)
            # self.node.sensingQueue.put_nowait((data, seq, time.time()))
            time.sleep(timeout)
            seq += 1
    
        