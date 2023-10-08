from ssdwsn.util.constants import Constants as ct
from binascii import unhexlify
from os import environ, kill
from os.path import join as path_join
import re
import asyncio
import webbrowser
import time

def getOperandFromString(val:str):

    tmp = []
    strVal = val.split(".")
    switcherLh = {
        "P": ct.PACKET,
        "R": ct.STATUS
    }
    tmp.append(switcherLh.get(strVal[0], ct.CONST))
    if tmp[0] == ct.PACKET:
        tmp.append(getNetworkPacketByteFromName(strVal[1]))
    elif tmp[0] == ct.CONST:
        tmp.append(strVal[0])
    else: tmp.append(strVal[1])
    
    return tmp

def getColorInt(val):
    return {
        "black": 0,
        "red": 1,
        "green": 2,
        "orange": 3,
        "blue": 4,
        "gray": 5          
    }.get(val, val)

def getColorVal(val):
    return {
        0 : "black",
        1: "red",
        2 : "green",
        3 : "orange",
        4 : "blue",
        5 : "gray"         
    }.get(val, val)

def getNetworkPacketByteFromName(val):
    return {
        "LEN": ct.LEN_INDEX,
        "NET": ct.NET_INDEX,
        "SRC": ct.SRC_INDEX,
        "DST": ct.DST_INDEX,
        "TYP": ct.TYP_INDEX,
        "TTL": ct.TTL_INDEX           
    }.get(val, val)

def getNetworkPacketByteName(val):
    return {
        ct.LEN_INDEX : "LEN",
        ct.NET_INDEX : "NET",
        ct.SRC_INDEX : "SRC",
        ct.DST_INDEX : "DST",
        ct.TYP_INDEX : "TYP",
        ct.TTL_INDEX : "TTL"           
    }.get(val, val)
            
def getCompOperatorFromString(val):
    return {
        "==" : ct.EQUAL,
        "!=" : ct.NOT_EQUAL,
        ">"  : ct.GREATER,
        "<"  : ct.LESS,
        ">=" : ct.GREATER_OR_EQUAL,
        "<=" : ct.LESS_OR_EQUAL
    }.get(val, 'error')

def getMathOperatorFromString(val):
    return {
        "+"  : ct.ADD,
        "-"  : ct.SUB,
        "*"  : ct.MUL,
        "/"  : ct.DIV,
        "%"  : ct.MOD,
        "&"  : ct.AND,
        "|"  : ct.OR,
        "^"  : ct.XOR
    }.get(val, 'error')
    
def getCompOperatorToString(val):
    return {
        ct.EQUAL : "==",
        ct.NOT_EQUAL : "!=",
        ct.GREATER : ">",
        ct.LESS : "<",
        ct.GREATER_OR_EQUAL : ">=",
        ct.LESS_OR_EQUAL : "<="
    }.get(val, "error")
        
def getMathOperatorToString(val):
    return {
        ct.ADD : " + ",
        ct.SUB : " - ",
        ct.MUL : " * ",
        ct.DIV : " / ",
        ct.MOD : " % ",
        ct.AND : " & ",
        ct.OR : " | ",
        ct.XOR : " ^ "
    }.get(val, "error")
        
def compare(op, val1, val2):
    if val1 == -1 or val2 == -1:
        return False
    else:
        return{
        ct.EQUAL : True if val1 == val2 else False,
        ct.NOT_EQUAL : True if val1 != val2 else False,
        ct.GREATER : True if val1 > val2 else False,
        ct.LESS : True if val1 < val2 else False,
        ct.GREATER_OR_EQUAL : True if val1 >= val2 else False,
        ct.LESS_OR_EQUAL : True if val1 <= val2 else False
        }.get(op, False)
        
def setBitRange(val:int=None, start:int=None, len:int=None, newVal:int=None):
    mask = ((1 << len) - 1) << start
    return (val & ~mask) | ((newVal << start) & mask)

def getBitRange(b:int=None, s:int=None, n:int=None):
    mask = ((1 << n) - 1) << s
    return (b & mask) >> s
    # return (((b & ct.MASK) >> (s & ct.MASK))
    #             & ((1 << (n & ct.MASK)) - 1)) & ct.MASK

def mergeBytes(high=None, low=None):
    return (high << 8) | low

def fromIntArrayToByteArray(val:list):
    res = bytearray()
    for i in len(val):
        res.append(val[i])
    return res

def byteToStr(val:bytearray):
    return ''.join(chr(x).format('utf-8') for x in val if x < 256)     
    
def bitsToBytes(val:str):
    return bytearray(int(val[i : i + 8], 2) for i in range(0, len(val), 8))
    
def _colonHex( val, bytecount ):
    """Generate colon-hex string.
       val: input as unsigned int
       bytecount: number of bytes to convert
       returns: chStr colon-hex string"""
    pieces = []
    for i in range( bytecount - 1, -1, -1 ):
        piece = ( ( 0xff << ( i * 8 ) ) & val[i] ) >> ( i * 8 )
        pieces.append( '%02x' % piece )
    chStr = ':'.join( pieces )
    return chStr

def macColonHex( mac ):
    """Generate MAC colon-hex string from unsigned int.
       mac: MAC address as unsigned int
       returns: macStr MAC colon-hex string"""
    return _colonHex( mac, 6 )

def autoGenMac(nextMac:int):    
    return macColonHex(nextMac)

def portToAddrStr(portseq:int):
    prefixLen = 4
    baseNum = 0x0001
    imax = 0xffff >> prefixLen
    assert(portseq <= imax)
    mask = 0xffff ^ imax
    addr = ( baseNum & mask ) + portseq
    w = ( addr >> 8 ) & 0xff
    x = addr & 0xff
    return str(w)+'.'+str(x)

def ipStr( ip ):
    """Generate IP address string from an unsigned int.
       ip: unsigned int of form w << 24 | x << 16 | y << 8 | z
       returns: ip address string w.x.y.z"""
    w = ( ip >> 24 ) & 0xff
    x = ( ip >> 16 ) & 0xff
    y = ( ip >> 8 ) & 0xff
    z = ip & 0xff
    return "%i.%i.%i.%i" % ( w, x, y, z )

def netParse( ipstr ):
    """Parse an IP network specification, returning
       address and prefix len as unsigned ints"""
    prefixLen = 0
    if '/' in ipstr:
        ip, pf = ipstr.split( '/' )
        prefixLen = int( pf )
    # if no prefix is specified, set the prefix to 24
    else:
        ip = ipstr
        prefixLen = 24
    return ipParse( ip ), prefixLen

def ipParse( ip ):
    "Parse an IP address and return an unsigned int."
    args = [ int( arg ) for arg in ip.split( '.' ) ]
    while len(args) < 4:
        args.insert( len(args) - 1, 0 )
    return ipNum( *args )

def ipNum( w, x, y, z ):
    """Generate unsigned int from components of IP address
       returns: w << 24 | x << 16 | y << 8 | z"""
    return ( w << 24 ) | ( x << 16 ) | ( y << 8 ) | z

def ipAdd6(i, prefixLen=32, ipBaseNum=0xff020000000000000000000000000000):
    """Return IP address string from ints
       i: int to be added to ipbase
       prefixLen: optional IP prefix length
       ipBaseNum: option base IP address as int
       returns IP address as string"""
    MAX_128 = 0xffffffffffffffffffffffffffffffff
    ipv6_max = MAX_128 >> prefixLen
    assert i <= ipv6_max, 'Not enough IPv6 addresses in the subnet'
    mask = MAX_128 ^ ipv6_max
    ipnum = ( ipBaseNum & mask ) + i
    return ip6Str(ipnum)


def ip6Str(ip):
    """Generate IP address string from an unsigned int.
       ip: unsigned int of form w << 24 | x << 16 | y << 8 | z
       returns: ip address string w.x.y.z"""
    x1 = (ip >> 112) & 0xffff
    x2 = (ip >> 96) & 0xffff
    x3 = (ip >> 80) & 0xffff
    x4 = (ip >> 64) & 0xffff
    x5 = (ip >> 48) & 0xffff
    x6 = (ip >> 32) & 0xffff
    x7 = (ip >> 16) & 0xffff
    x8 = ip & 0xffff
    return "%s:%s:%s:%s:%s:%s:%s:%s" % (format(x1, 'x'), format(x2, 'x'), format(x3, 'x'), format(x4, 'x'), format(x5, 'x'), format(x6, 'x'), format(x7, 'x'), format(x8, 'x'))

def netParse6(ipstr):
    """Parse an IP network specification, returning
       address and prefix len as unsigned ints"""
    prefixLen = 0
    if '/' in ipstr:
        ip, pf = ipstr.split('/')
        prefixLen = int(pf)
    # if no prefix is specified, set the prefix to 24
    else:
        ip = ipstr
        prefixLen = 24
    return ip6Parse(ip), prefixLen

def long_to_bytes (val, endianness='big'):
    """
    Use :ref:`string formatting` and :func:`~binascii.unhexlify` to
    convert ``val``, a :func:`long`, to a byte :func:`str`.

    :param long val: The value to pack

    :param str endianness: The endianness of the result. ``'big'`` for
      big-endian, ``'little'`` for little-endian.

    If you want byte- and word-ordering to differ, you're on your own.

    Using :ref:`string formatting` lets us use Python's C innards.
    """

    # one (1) hex digit per four (4) bits
    width = val.bit_length()

    # unhexlify wants an even multiple of eight (8) bits, but we don't
    # want more digits than we need (hence the ternary-ish 'or')
    width += 8 - ((width % 8) or 8)

    # format width specifier: four (4) bits per hex digit
    fmt = '%%0%dx' % (width // 4)

    # prepend zero (0) to the width, to zero-pad the output
    s = unhexlify(fmt % val)

    if endianness == 'little':
        # see http://stackoverflow.com/a/931095/309233
        s = s[::-1]

    return s
def checkInt( s ):
    "Check if input string is an int"
    try:
        int( s )
        return True
    except ValueError:
        return False

def checkFloat( s ):
    "Check if input string is a float"
    try:
        float( s )
        return True
    except ValueError:
        return False

def makeNumeric( s ):
    "Convert string to int or float if numeric."
    if checkInt( s ):
        return int( s )
    elif checkFloat( s ):
        return float( s )
    else:
        return s
    
def splitArgs( argstr ):
    """Split argument string into usable python arguments
       argstr: argument string with format fn,arg2,kw1=arg3...
       returns: fn, args, kwargs"""
    split = argstr.split( ',' )
    fn = split[ 0 ]
    params = split[ 1: ]
    # Convert int and float args; removes the need for function
    # to be flexible with input arg formats.
    args = [ makeNumeric( s ) for s in params if '=' not in s ]
    kwargs = {}
    for s in [ p for p in params if '=' in p ]:
        key, val = s.split( '=', 1 )
        kwargs[ key ] = makeNumeric( val )
    return fn, args, kwargs

def customClass( classes, argStr ):
    """Return customized class based on argStr
    The args and key/val pairs in argStr will be automatically applied
    when the generated class is later used.
    """
    cname, args, kwargs = splitArgs( argStr )
    cls = classes.get( cname, None )
    if not cls:
        raise Exception( "error: %s is unknown - please specify one of %s" %
                         ( cname, classes.keys() ) )
    if not args and not kwargs:
        return cls

    return specialClass( cls, append=args, defaults=kwargs )

def specialClass( cls, prepend=None, append=None,
                  defaults=None, override=None ):
    """Like functools.partial, but it returns a class
       prepend: arguments to prepend to argument list
       append: arguments to append to argument list
       defaults: default values for keyword arguments
       override: keyword arguments to override"""

    if prepend is None:
        prepend = []

    if append is None:
        append = []

    if defaults is None:
        defaults = {}

    if override is None:
        override = {}

    class CustomClass( cls ):
        "Customized subclass with preset args/params"
        def __init__( self, *args, **params ):
            newparams = defaults.copy()
            newparams.update( params )
            newparams.update( override )
            cls.__init__( self, *( list( prepend ) + list( args ) +
                                   list( append ) ),
                          **newparams )

    CustomClass.__name__ = '%s%s' % ( cls.__name__, defaults )
    return CustomClass

def addrStr( val ):
    """Generate Addr string from an unsigned int.
       val: unsigned int of form w << 8 | x
       returns: val address string w.x"""
    w = ( val >> 8 ) & 0xff
    x = val & 0xff
    return "%i.%i" % ( w, x )

def addrNum( w, x):
    """Generate unsigned int from components of Addr
       returns: w << 8 | x"""
    return ( w << 8 ) | x

def moteAddr( i, prefixLen=4, baseNum=0xa001 ):
    """Return Addr string from ints
       i: int to be added to mote base addr
       prefixLen: optional prefix length
       baseNum: option base Addr as int
       returns Addr as string"""
    imax = 0xffff >> prefixLen
    assert i <= imax, 'Not enough Addres in the subnet'
    mask = 0xffff ^ imax
    addr = ( baseNum & mask ) + i
    return addrStr( addr )

def sinkAddr( i, prefixLen=4, baseNum=0x0001 ):
    """Return Addr string from ints
       i: int to be added to sink base addr
       prefixLen: optional prefix length
       baseNum: option base Addr as int
       returns Addr as string"""
    imax = 0xffff >> prefixLen
    assert i <= imax, 'Not enough Addres in the subnet'
    mask = 0xffff ^ imax
    addr = ( baseNum & mask ) + i
    return addrStr( addr )

def addrParse( ip ):
    "Parse an Addr and return an unsigned int."
    args = [ int( arg ) for arg in ip.split( '.' ) ]
    while len(args) < 2:
        args.insert( len(args) - 1, 0 )
    return addrNum( *args )

def setPos(node):
    nums = re.findall(r'\d+', node.name)
    if nums:
        id = int(hex(int(nums[0]))[2:])
        node.position = (10, round(id, 2), 0)
        
def mapRSSI(val):
    val = 0 if val > 0 else val
    rssiRealSpane = 0 - (-100)
    rssiSimSpane = ct.RSSI_MAX - ct.RSSI_MIN
    rssiScaled = int(val - (-100))/ int(rssiRealSpane)
    return int(ct.RSSI_MIN + (rssiScaled*rssiSimSpane))

def openURL(url):    
    # MacOS
    # chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
    # Windows
    # chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    # Linux
    chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
    webbrowser.get(chrome_path).open(url)

def ipParse(ip):
    "Parse an IP address and return an unsigned int."
    args = [ int(arg) for arg in ip.split(':') ]
    while len(args) < 8:
        args.append(0)
    return ipNum(*args)

def netParse(ipstr):
    """Parse an IP network specification, returning
       address and prefix len as unsigned ints"""
    prefixLen = 0
    if '/' in ipstr:
        ip, pf = ipstr.split('/')
        prefixLen = int(pf)
    # if no prefix is specified, set the prefix to 24
    else:
        ip = ipstr
        prefixLen = 24
    return ipParse(ip), prefixLen

def ip6Num(x1, x2, x3, x4, x5, x6, x7, x8):
    "Generate unsigned int from components of IP address"
    return (x1 << 112) | (x2 << 96) | (x3 << 80) | (x4 << 64) | (x5 << 48) | (x6 << 32) | (x7 << 16) | x8

def ip6Parse(ip):
    "Parse an IP address and return an unsigned int."
    args = [ int(arg, 16) for arg in ip.split(':') ]
    while len(args) < 8:
        args.append(0)
    return ip6Num(*args)


def netParse6(ipstr):
    """Parse an IP network specification, returning
       address and prefix len as unsigned ints"""
    prefixLen = 0
    if '/' in ipstr:
        ip, pf = ipstr.split('/')
        prefixLen = int(pf)
    # if no prefix is specified, set the prefix to 24
    else:
        ip = ipstr
        prefixLen = 24
    return ip6Parse(ip), prefixLen

def ipAdd(i, prefixLen=64, ipBaseNum=0x0a000000):
    """Return IP address string from ints
       i: int to be added to ipbase
       prefixLen: optional IP prefix length
       ipBaseNum: option base IP address as int
       returns IP address as string"""
    #imax = 0xffffffff >> prefixLen
    #assert i <= imax, 'Not enough IP addresses in the subnet'
    #mask = 0xffffffff ^ imax
    mask = 64
    ipnum = (ipBaseNum & mask) + i
    return ipStr(ipnum)

def makeIntfPair( intf1, intf2, addr1=None, addr2=None, node1=None, node2=None,
                  deleteIntfs=True, runCmd=None ):
    """Make a veth pair connnecting new interfaces intf1 and intf2
       intf1: name for interface 1
       intf2: name for interface 2
       addr1: MAC address for interface 1 (optional)
       addr2: MAC address for interface 2 (optional)
       node1: home node for interface 1 (optional)
       node2: home node for interface 2 (optional)
       deleteIntfs: delete intfs before creating them
       runCmd: function to run shell commands (quietRun)
       raises Exception on failure"""
    if not runCmd:
        runCmd = quietRun if not node1 else node1.cmd
        runCmd2 = quietRun if not node2 else node2.cmd
    if deleteIntfs:
        # Delete any old interfaces with the same names
        runCmd( 'ip link del ' + intf1 )
        runCmd2( 'ip link del ' + intf2 )
    # Create new pair
    netns = 1 if not node2 else node2.pid
    if addr1 is None and addr2 is None:
        cmdOutput = runCmd( 'ip link add name %s '
                            'type veth peer name %s '
                            'netns %s' % ( intf1, intf2, netns ) )
    else:
        cmdOutput = runCmd( 'ip link add name %s '
                            'address %s '
                            'type veth peer name %s '
                            'address %s '
                            'netns %s' %
                            (  intf1, addr1, intf2, addr2, netns ) )
    if cmdOutput:
        raise Exception( "Error creating interface pair (%s,%s): %s " %
                         ( intf1, intf2, cmdOutput ) )

def runCmd(cmd):
    myenv = environ.copy()
    if 'VIRTUAL_ENV' in environ:
        myenv['PATH'] = ':'.join(
            [x for x in environ['PATH'].split(':')
                if x != path_join(environ['VIRTUAL_ENV'], 'bin')])
    return Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, env=myenv) 

import sys
import codecs
Python3 = sys.version_info[0] == 3
BaseString = str if Python3 else getattr( str, '__base__' )
Encoding = 'utf-8' if Python3 else None
class NullCodec( object ):
    "Null codec for Python 2"
    @staticmethod
    def decode( buf ):
        "Null decode"
        return buf

    @staticmethod
    def encode( buf ):
        "Null encode"
        return buf


if Python3:
    def decode( buf ):
        "Decode buffer for Python 3"
        return buf.decode( Encoding )

    def encode( buf ):
        "Encode buffer for Python 3"
        return buf.encode( Encoding )
    getincrementaldecoder = codecs.getincrementaldecoder( Encoding )
else:
    decode, encode = NullCodec.decode, NullCodec.encode

    def getincrementaldecoder():
        "Return null codec for Python 2"
        return NullCodec

try:
    # pylint: disable=import-error
    oldpexpect = None
    import pexpect as oldpexpect

    # pylint: enable=import-error
    class Pexpect( object ):
        "Custom pexpect that is compatible with str"
        @staticmethod
        def spawn( *args, **kwargs):
            "pexpect.spawn that is compatible with str"
            if Python3 and 'encoding' not in kwargs:
                kwargs.update( encoding='utf-8'  )
            return oldpexpect.spawn( *args, **kwargs )

        def __getattr__( self, name ):
            return getattr( oldpexpect, name )
    pexpect = Pexpect()
except ImportError:
    pass

def isShellBuiltin( cmd ):
    "Return True if cmd is a bash builtin."
    if isShellBuiltin.builtIns is None:
        isShellBuiltin.builtIns = set(quietRun( 'bash -c enable' ).split())
    space = cmd.find( ' ' )
    if space > 0:
        cmd = cmd[ :space]
    return cmd in isShellBuiltin.builtIns

isShellBuiltin.builtIns = None

from subprocess import call, check_call, Popen, PIPE, STDOUT

def quietRun( cmd, **kwargs ):
    "Run a command and return merged stdout and stderr"
    return errRun( cmd, stderr=STDOUT, **kwargs )[ 0 ]

# This is a bit complicated, but it enables us to
# monitor command output as it is happening

from select import poll, POLLIN, POLLHUP

# pylint: disable=too-many-branches,too-many-statements
def errRun( *cmd, **kwargs ):
    """Run a command and return stdout, stderr and return code
       cmd: string or list of command and args
       stderr: STDOUT to merge stderr with stdout
       shell: run command using shell
       echo: monitor output to console"""
    # By default we separate stderr, don't run in a shell, and don't echo
    stderr = kwargs.get( 'stderr', PIPE )
    shell = kwargs.get( 'shell', False )
    echo = kwargs.get( 'echo', False )
    if echo:
        # cmd goes to stderr, output goes to stdout
        print( cmd, '\n' )
    if len( cmd ) == 1:
        cmd = cmd[ 0 ]
    # Allow passing in a list or a string
    if isinstance( cmd, BaseString ) and not shell:
        cmd = cmd.split( ' ' )
        cmd = [ str( arg ) for arg in cmd ]
    elif isinstance( cmd, list ) and shell:
        cmd = " ".join( arg for arg in cmd )
    print( '*** errRun:', cmd, '\n' )
    popen = Popen( cmd, stdout=PIPE, stderr=stderr, shell=shell )
    # We use poll() because select() doesn't work with large fd numbers,
    # and thus communicate() doesn't work either
    out, err = '', ''
    poller = poll()
    poller.register( popen.stdout, POLLIN )
    fdToFile = { popen.stdout.fileno(): popen.stdout }
    fdToDecoder = { popen.stdout.fileno(): getincrementaldecoder() }
    outDone, errDone = False, True
    if popen.stderr:
        fdToFile[ popen.stderr.fileno() ] = popen.stderr
        fdToDecoder[ popen.stderr.fileno() ] = getincrementaldecoder()
        poller.register( popen.stderr, POLLIN )
        errDone = False
    while not outDone or not errDone:
        readable = poller.poll()
        for fd, event in readable:
            f = fdToFile[ fd ]
            decoder = fdToDecoder[ fd ]
            if event & ( POLLIN | POLLHUP ):
                data = decoder.decode( f.read( 1024 ) )
                if echo:
                    print( data )
                if f == popen.stdout:
                    out += data
                    if data == '':
                        outDone = True
                elif f == popen.stderr:
                    err += data
                    if data == '':
                        errDone = True
            else:  # something unexpected
                if f == popen.stdout:
                    outDone = True
                elif f == popen.stderr:
                    errDone = True
                poller.unregister( fd )

    returncode = popen.wait()
    # Python 3 complains if we don't explicitly close these
    popen.stdout.close()
    if stderr == PIPE:
        popen.stderr.close()
    print( out, err, returncode )
    return out, err, returncode

def getInterArrivalTime(pt:float=0.0, pat:float=0.0, alpha:float=0.3):
    """Exponentially Weighted Moving Average (EWMA)
    To calculate Flow Requests Inter-Arrival Time
    EWMA_{t} = alpha * r_{t} + (1-alpha) * r_{t-1}
    pt: previous request time
    ct: current request time
    pat: previous inter-arrival time (r_{t-1})
    cat: current inter-arrival time (r_{t})
    alpha: is a constant weight decided by the user
    """
    cat = time.time() - pt
    return (alpha * cat) + ((1-alpha) * pat), time.time()

def zero_division(n, d):
    return n / d if d else 0

class Suspendable:
    def __init__(self, target, loop):
        self._target = target
        self._can_run = asyncio.Event()
        self._can_run.set()
        self._task = asyncio.ensure_future(self, loop=loop)

    def __await__(self):
        target_iter = self._target.__await__()
        iter_send, iter_throw = target_iter.send, target_iter.throw
        send, message = iter_send, None
        # This "while" emulates yield from.
        while True:
            # wait for can_run before resuming execution of self._target
            try:
                while not self._can_run.is_set():
                    yield from self._can_run.wait().__await__()
            except BaseException as err:
                send, message = iter_throw, err

            # continue with our regular program
            try:
                signal = send(message)
            except StopIteration as err:
                return err.value
            else:
                send = iter_send
            try:
                message = yield signal
            except BaseException as err:
                send, message = iter_throw, err

    def suspend(self):
        self._can_run.clear()

    def is_suspended(self):
        return not self._can_run.is_set()

    def resume(self):
        self._can_run.set()

    def get_task(self):
        return self._task

import logging
class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)