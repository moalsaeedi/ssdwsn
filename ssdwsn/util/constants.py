import os.path as path

class Constants:
    
    # Flow Table Entry Constans
    CONST = 1 
    NULL = 0
    PACKET = 2 
    STATUS = 3
    
    # Window Constants
    #locations    
    W_SIZE = 5 #in bytes
    
    W_SIZE_0 = 0 #no. of windows
    W_SIZE_1 = 1 #no. of windows
    #indexes
    W_LEFT_BIT = 3    
    W_LEFT_INDEX_H = 1 
    W_LEFT_INDEX_L = 2
    W_LEFT_LEN = 2 
    W_OP_BIT = 5 
    W_OP_INDEX = 0 
    W_OP_LEN = 3
    W_RIGHT_BIT = 1 
    W_RIGHT_INDEX_H = 3 
    W_RIGHT_INDEX_L = 4
    W_RIGHT_LEN = W_LEFT_LEN
    W_SIZE_BIT = 0
    W_SIZE_LEN = 1 
    W_LEN = 3
    W_LHS_OPTIONS = ['P.SRC', 'P.DST', 'P.NXH']
    W_RHS_OPTIONS = ['P.SRC', 'P.DST', 'P.NXH']
    
        
    # CONSTANTS 
    RANDOM_SEED = 5
    CNT_DATA_MAX = 10
    CNT_BEACON_MAX = 10
    CNT_REPORT_MAX = 2 * CNT_BEACON_MAX
    CNT_UPDTABLE_MAX = 6
    STATUS_LEN = 10000
    COM_START_BYTE = 0x7A
    COM_STOP_BYTE = 0x7E
    MAC_SEND_UNICAST = False
    MAC_SEND_BROADCAST = True
    # Packet Constants
    # Packet
    DFLT_PAYLOAD_LEN = 109
    DFLT_HDR_LEN = 18 
    MTU = DFLT_PAYLOAD_LEN + DFLT_HDR_LEN
    # Buffering
    #Queue size.
    BUFFER_SIZE = 100 #queue size (packets)   
    AGGR_BUFFER_SIZE = 50 #aggregation max buffer size 
    # Flow Table
    FT_MAX_ENTRIES = 50 #number of entries
    # Indexes
    NET_INDEX = 0
    LEN_INDEX = 1
    DST_INDEX = 2
    DST_LEN = 2
    SRC_INDEX = 4
    SRC_LEN = 2
    TYP_INDEX = 6
    TTL_INDEX = 7
    NXH_INDEX = 8
    NXH_LEN = 2
    PRH_INDEX = 10
    PRH_LEN = 2
    TS_INDEX = 12
    TS_LEN = 6
    PLD_INDEX = 18 
    # Types
    DATA = 0
    BEACON = 1
    REPORT = 2
    REQUEST = 3
    RESPONSE = 4
    OPEN_PATH = 5
    CONFIG = 6
    REG_PROXY = 7  
    AGGR = 8    
    TTL_MAX = 100 # packets max time to live value (number of hops)  
    DIST_MAX = 100 # max distance (number of hops) to the sink
    THRES = 63
    SIM_TIME = 3600
    # Transmission Time Intervals (Data TTI, Report TTI, Beacon TTI)
    DA_TTI = 1000 # in msec
    MAX_DA_TTI = 2000 # in msec
    MIN_DA_TTI = 1000 # in msec
    BE_TTI = 5000 # in msec
    MAX_BE_TTI = 5000 # in msec
    MIN_BE_TTI = 250 # in msec
    RP_TTI = 1500  # transmittion time interval for reporting a node to the controller (in msec)
    MAX_RP_TTI = 5000 # in msec
    MIN_RP_TTI = 1500 # in msec
    RP_IDLE = RP_TTI * 20 # Idle-timeout to remove non-active nodes from the controller's network global view (in msec)
    # Beacon Packet
    BEACON_HDR_LEN = 12
    REPORT_HDR_LEN = 13
    DIST_INDEX = 0
    BATT_INDEX = 1
    BATT_LEN = 4
    POS_INDEX = 5
    POS_LEN = 4
    INFO_INDEX= 9
    INFO_LEN = 1
    TYP_BIT_INDEX = 0
    TYP_BIT_LEN = 4
    INTF_BIT_INDEX = 4
    INTF_BIT_LEN = 4
    PORT_INDEX= 10
    PORT_LEN = 2
    # Config Packet
    CNF_PATH_INDEX = 0
    CNF_WRITE = 1
    CNF_MASK_POS = 7
    CNF_MASK = 0x7F
    # Open Path Packet
    OP_WINS_SIZE_INDEX = 0
    # Reg Proxy Packet
    REG_HDR_LEN = 22
    REG_DPID_INDEX = 0
    DPID_LEN = 8
    IP_LEN = 4
    REG_MAC_INDEX = REG_DPID_INDEX + DPID_LEN
    MAC_LEN = 6
    MAC_STR_LEN = 18
    REG_PORT_INDEX = REG_MAC_INDEX + MAC_LEN
    REG_IP_INDEX = REG_PORT_INDEX + PORT_LEN
    RADIX = 16
    REG_TCP_INDEX = REG_IP_INDEX + IP_LEN
    # Report Packet
    NEIGH_INDEX = BEACON_HDR_LEN
    NEIGH_LEN = 3
    MAX_NEIG = round((DFLT_PAYLOAD_LEN - (DFLT_HDR_LEN + REPORT_HDR_LEN))/NEIGH_LEN) #TODO what is the practical max number of neighbors
    # Request Packet 
    ID_INDEX = 0
    PART_INDEX = 1
    TOTAL_INDEX = 2
    REQUEST_HDR_LEN = 3
    REQUEST_PAYLOAD_SIZE = DFLT_PAYLOAD_LEN \
            - (DFLT_HDR_LEN + REQUEST_HDR_LEN)
    # Aggregate Packet
    AGGR_HDR_LEN = 1
    AGGR_PLHDR_LEN = 9
    AGGR_TYPE_INDEX = 0
    AGGR_SRC_INDEX = 1
    AGGR_SRC_LEN = 2
    AGGR_TS_INDEX = 3
    AGGR_TS_LEN = 6
    AGGR_PAYLOAD_SIZE = DFLT_PAYLOAD_LEN \
            - (DFLT_HDR_LEN + AGGR_HDR_LEN)
    # AGGR_CACHE = 2 #number of aggregated packets
    AGGR_MAX_LEVEL = 1
    # Stats Constans
    RL_IDLE_PERM = 255
    RL_IDLE = 4 #max value is 254
    MAX_DELAY = 200 #max accepted delay (can be updated based on the required QoS)
    MAX_BANDWIDTH = 250 #max network bandwidth (QoS)
    ST_SIZE = 12
    ST_TTL_INDEX = 0
    ST_TTL_LEN = 4
    ST_IDLE_INDEX = 4
    ST_PCOUNT_INDEX = 5
    ST_PCOUNT_LEN = 4
    ST_BCOUNT_INDEX = ST_PCOUNT_INDEX+ST_PCOUNT_LEN
    ST_BCOUNT_LEN = ST_PCOUNT_LEN
    
    # Actions Constants
    AC_TYPE_INDEX = 0
    AC_VALUE_INDEX = 1

    # Set Action Constans
    # indexes
    SET_LEFT_BIT = 1
    SET_LEFT_INDEX_H = 3
    SET_LEFT_INDEX_L = 4
    SET_LEFT_LEN = 2
    SET_OP_BIT = 3
    SET_OP_INDEX = 0 
    SET_OP_LEN = 3
    SET_RES_BIT = 0
    SET_RES_INDEX_H = 1
    SET_RES_INDEX_L = 2
    SET_RES_LEN = 1
    SET_RIGHT_BIT = 6
    SET_RIGHT_INDEX_H = 5
    SET_RIGHT_INDEX_L = 6
    SET_RIGHT_LEN = SET_LEFT_LEN 
    # string parsing
    SET_FULL_SET = 6
    SET_HALF_SET = 4
    SET_RES = 1
    SET_LHS = 3
    SET_RHS = 5
    SET_OP = 4
    # size
    SET_SIZE = 7  
    
    # Match Action Constants
    MATCH_SIZE = 0 
    
    # Drop Action Constants
    DROP_SIZE = 0
    
    # Ask Action Constants
    ASK_SIZE = 0
    
    # Function Action Constants
    FN_ARGS_INDEX = 1
    FN_ID_INDEX = 0 
    # size
    FN_SIZE = 0
    
    # Forward Action Constants
    FD_NXH_INDEX = 0
    FD_SIZE = 2
    
    # DRL Agent Actions Constants
    DRL_AG_INDEX = 1
    DRL_AG_LEN = 1
    DRL_NH_INDEX = 2
    DRL_NH_LEN = 2
    DRL_RT_INDEX = 3
    DRL_RT_LEN = 2
    DRL_DT_INDEX = 4
    DRL_DT_LEN = 2
    DRL_DR_INDEX = 5
    DRL_DR_LEN = 1
    # Forward Unicast Action Constants
    
    # Forward Multicast Action Constants
    
    # Controller Constants
    # UTF8_CHARSET = Charset.forName("UTF-8")
    CTRL_BUFF_SIZE = 16384
    SINK_BUFF_SIZE = 16384
    CTRL_DELAY = 200
    CONFIG_HEADER_LEN = 1
    CTRL_PARTS_MAX = 256
    CACHE_MAX_SIZE = 1000
    CACHE_EXP_TIME = 5
    RESPONSE_TIMEOUT = 300
    SLOW_NET = 0.25 #sec slow factor: to slow the network transmission (lower the CPU load)
    # Battery Constants
    # 1 AA = 9000 Columb (C)
    # 1 AA = 9000 * 1000 = 9000000 MilliColumb (mC)
    # 1 AA = 9000 * 1000 * 1.5v = 13500000 Millijoules (mJ)
    # 1 AA has nearly 2.5 Ah --> 2.5 * 60 * 60 = 9000 C = 9000000 mC
    # 1 ampere = 1 C    
    # ampere (A)*(h)*1000 =(mAh)
    # 1 mA = 0.001 C = 1 mC
    # 1 mAh = 3.6 C
    # 1 mAh = (1 / 1000) * 60 * 60 = 3.6 A = 3.6 C

    # 6lowpan protocol on MicaZ mote

    #MICAz (Crossbow Technology Inc. 2012) with CC2420 radio transceiver
    # The MICAz radio energy model is used for
    # low power wireless sensor network. The data rate of MICAz
    # is 250 Kbps. The transmission and receiving currents are 19
    # mA and 17 mA respectively. The MICA Motes radio energy
    # model is pre-configured with the specification of power
    # consumption of MICA motes. It can transmit approximately
    # 40 kbits per second. In sleep mode, the radio consumes less
    # than one micro ampere. Receiving and transmission powers
    # are 10 and 25 mA respectively.
    # To validate our system, we configure our simulation by
    # considering the energy consumption parameters of MicaZ
    # [7], a well-known platform for sensor networks applica-
    # tions. Note, however, that any other platform with similar
    # energy requirements can be used instead. MicaZ is com-
    # posed of a 7 Mhz l controller, a Zigbee-compliant CC2420
    # radio, 3 LEDs, and an external flash memory. To provide
    # capacity of sensing, it may be integrated with different
    # sensor boards. MicaZ is powered by 2-AA batteries, which
    # should provide a voltage between 2.7 and 3.3 for the cor-
    # rect working of the sensor node. According to the docu-
    # mentation, the energy requirements of MicaZ are shown in
    # Table 2. The power demand of MicaZ depends on the
    # components that need to be on in the application and in
    # which mode. In the worst case, if we take into account the
    # maximum current draw of each component (whose sum is
    # 64.7 mA according to Table 2), the power required by
    # MicaZ is 64.7 9 3.3 = 213.5 mW. However, it is typically
    # sufficient to guarantee 100 mW to power the device.
    # Current Draw 
    # 19.7 mA Receive mode
    # 11 mA TX, -10 dBm
    # 14 mA TX, -5 dBm
    # 17.4 mA TX, 0 dBm
    # 20 μA Idle mode, voltage regular on
    # 1 μA Sleep mode, voltage regulator off
    # Electromechanical
    # Battery 2X AA batteries Attached pack
    # External Power 2.7 V - 3.3 V Molex connector provided
    MAX_LEVEL = 9000000 # in Millicoulomb (mC)
    KEEP_ALIVE = 8 # mC spent every 1 sec (Mote Active mode)
    KEEP_SLEEP = 0.015 # mC spent every 1 sec (Mote Sleep mode)
    RADIO_TX = 17.4 # mC spent every 1 sec (Radio Tx per second)
    RADIO_RX = 19.7 # mC spent every 1 sec (Radio Rx per second)
    BATT_LEVEL = 5000000 # default simulation initial energy
    MAX_ENRCONS = 5 * RADIO_RX # maximum energy consumption every 1 sec
    # 50000 mC energy, which is 50000 * 1.5 = 75000 mJ (75000/13500000 = 0.0055% mJ remaining energy of 1 AA battery)
    # The energy consumption can be calculated in Joules as follows:

    # Transmit-Energy-Consumed = Transmit-Current * Voltage * Time-for-which-node-transmits-packets
    # Receive-Energy-Consumed = Receive-Current * Voltage * Time -for-which-node-receives-packets
    # IdleMode-Energy-Consumed = IdleMode-Current * Voltage * Time-in-Idle-Mode
    # SleepMode-Energy-Consumed = SleepMode-Current * Voltage * Time-in-sleep-mode

    # Example Calculation

    # Let's assume the following parameters for a MICAz mote based on typical datasheet values:

    # Power consumed during transmission (PTXPTX​) = 52.2 mW (milliwatts)
    # Power consumed during reception (PRXPRX​) = 59.1 mW
    # Power consumed during idle (PidlePidle​) = 0.96 mW
    # Power consumed during sleep (PsleepPsleep​) = 0.003 mW
    # Duration of transmission (TTXTTX​) = 1 second
    # Duration of reception (TRXTRX​) = 1 second
    # Duration in idle (TidleTidle​) = 10 seconds
    # Duration in sleep (TsleepTsleep​) = 100 seconds

    # Calculate Transmit Energy Consumption:
    # ETX=PTX⋅TTX=52.2 mW⋅1 s=52.2 mJ
    # ETX​=PTX​⋅TTX​=52.2mW⋅1s=52.2mJ

    # where 1 milliwatt (mW) = 1 millijoule per second (mJ/s).

    # Calculate Receive Energy Consumption:
    # ERX=PRX⋅TRX=59.1 mW⋅1 s=59.1 mJ
    # ERX​=PRX​⋅TRX​=59.1mW⋅1s=59.1mJ

    # Calculate Idle Energy Consumption:
    # Eidle=Pidle⋅Tidle=0.96 mW⋅10 s=9.6 mJ
    # Eidle​=Pidle​⋅Tidle​=0.96mW⋅10s=9.6mJ

    # Calculate Sleep Energy Consumption:
    # Esleep=Psleep⋅Tsleep=0.003 mW⋅100 s=0.3 mJ
    # Esleep​=Psleep​⋅Tsleep​=0.003mW⋅100s=0.3mJ

    # Calculate Total Energy Consumption:
    # Etotal=ETX+ERX+Eidle+Esleep=52.2 mJ+59.1 mJ+9.6 mJ+0.3 mJ=121.2 mJ
    # Etotal​=ETX​+ERX​+Eidle​+Esleep​=52.2mJ+59.1mJ+9.6mJ+0.3mJ=121.2mJ

    # Given the current draw values from the MICAz datasheet and assuming a voltage of 3V, the energy consumption for each state is calculated as follows:

    # Transmit Energy Consumption (ETXETX​):
    # ETX=PTX⋅TTX⋅V=17.4 mA×1 s×3 V=0.0522 J
    # ETX​=PTX​⋅TTX​⋅V=17.4mA×1s×3V=0.0522J

    # Receive Energy Consumption (ERXERX​):
    # ERX=PRX⋅TRX⋅V=19.7 mA×1 s×3 V=0.0591 J
    # ERX​=PRX​⋅TRX​⋅V=19.7mA×1s×3V=0.0591J

    # Idle Energy Consumption (EidleEidle​):
    # Eidle=Pidle⋅Tidle⋅V=20 μA×10 s×3 V=0.0006 J
    # Eidle​=Pidle​⋅Tidle​⋅V=20μA×10s×3V=0.0006J

    # Sleep Energy Consumption (EsleepEsleep​):
    # Esleep=Psleep⋅Tsleep⋅V=1 μA×100 s×3 V=0.0003 J
    # Esleep​=Psleep​⋅Tsleep​⋅V=1μA×100s×3V=0.0003J

    # Total Energy Consumption (EtotalEtotal​):
    # Etotal=ETX+ERX+Eidle+Esleep=0.0522 J+0.0591 J+0.0006 J+0.0003 J=0.1122 J
    # Etotal​=ETX​+ERX​+Eidle​+Esleep​=0.0522J+0.0591J+0.0006J+0.0003J=0.1122J

    # Neighbor Constants
    DEFAULT = 0xFF
    
    # Node Constants
    #Max RSSI value. RSSI value range is [-100, 0] the closer the value is to 0, the stronger the received signal has been. 
    RSSI_MAX = 255 #map to 0
    RSSI_MIN = 180 #map to -80 minimum boor acceptable signal quality
    RSSI_RESOLUTION = 30
    SNR_DEFAULT = 30

    # Sink Core
    CTRL_RSSI = 255
    
    # Adapter Com Constants
    TIMEOUT = 2000
    
    # Network Graph Constants
    MAX_BYTE = 255
    MILLIS_IN_SECOND = 1000
    LINK_DFL_COLOR = 'black'
    # Util Constants
    DIGITS = "0123456789abcdef"
    MASK = 0xFF
    MASK_1 = 4
    MASK_2 = 0xf   
    # operators
    EQUAL = 0
    GREATER = 2
    GREATER_OR_EQUAL = 4
    LESS = 3
    LESS_OR_EQUAL = 5
    NOT_EQUAL = 1
    
    ADD = 0
    AND = 5
    DIV = 3
    MOD = 4
    MUL = 2
    OR = 6
    SUB = 1
    XOR = 7
    BROADCAST_ADDR = "255.255"
    
    # NODES
    MAX_NODES = 1000
    
    # PORTS
    BASE_NODE_PORT = 6660
    TOP_MOTE_PORT = 9660
    BASE_CTRL_PORT = 9990
    CTRL_NET_VIEW_PUB = 5550
    CTRL_NET_VIEW_SUB = 5556
    
    # ADDR
    BASE_SINK_ADDR = 0x0001
    BASE_MOTE_ADDR = 0xa001
    BASE_IP6 = 0x20020000000000000000000000000000
    # DPID
    BASE_DPID = 0X0001
    
    # MAC
    BASE_MAC = 0x000000000001
    # INTERFACES TYPES
    INTF_PARAMS = {
        'sixlowpan': {
            'antHeight': 1, #cm
            'antGain': 4, #dBm 
            'antRange': 100, #m A pixel equal to 1m  
            'txPower': 12, #dBm Ex. 36 dBm (4 watt)
            'freq': 2.48, #GHz
        },
        'lorawan':{
            'antHeight': 1, #cm
            'antGain': 4, #dBm 
            'antRange': 12000, #m A pixel equal to 1m  
            'txPower': 12, #dBm Ex. 36 dBm (4 watt)
            'freq': 2.48, #GHz    
        }
    }
    ROOT_DIR = path.dirname(path.abspath(__file__))
    # CONFIG FILE
    CONFIG_FILE = 'ssdwsn/util/plot/static/json/config.json'
    # GRAPH FILE
    GRAPH_FILE = 'ssdwsn/util/plot/static/json/graph.json'
    # VISUALIZATION APP
    SIM_APP = 'ssdwsn/util/plot/app.py'
    SIM_PORT = 4455
    SIM_URL = "http://localhost:"+str(SIM_PORT)
    
    def __init__(self) -> None:            
        pass

if __name__ == '__main__':
    ct = Constants()
    print(ct.SIM_APP)