import struct
from abc import ABC, abstractmethod
from enum import Enum
from scapy.all import sniff, Ether, IP, UDP, TCP
import os
import socket
import zmq
import time
from .e3_logging import e3_logger, LOG_DIR, dapp_logger


class E3LinkLayer(Enum):
    ZMQ = "zmq"
    POSIX = "posix"
    SCAPY = "scapy"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(link_layer_str: str):
        try:
            return E3LinkLayer(link_layer_str.lower())
        except ValueError:
            raise ValueError(f"Invalid link layer: '{link_layer_str}'. Must be one of {[e.value for e in E3LinkLayer]}")

class E3TransportLayer(Enum):
    SCTP = "sctp"
    TCP = "tcp"
    IPC = "ipc"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(transport_layer_str: str):
        try:
            return E3TransportLayer(transport_layer_str.lower())
        except ValueError:
            raise ValueError(f"Invalid transport layer: '{transport_layer_str}'. Must be one of {[e.value for e in E3TransportLayer]}")

class E3Connector(ABC):
    # List of known valid configurations
    VALID_CONFIGURATIONS = [
        (E3LinkLayer.ZMQ, E3TransportLayer.IPC),
        (E3LinkLayer.ZMQ, E3TransportLayer.TCP),
        (E3LinkLayer.POSIX, E3TransportLayer.TCP),
        (E3LinkLayer.POSIX, E3TransportLayer.SCTP),
        (E3LinkLayer.POSIX, E3TransportLayer.IPC),
        (E3LinkLayer.SCAPY, E3TransportLayer.SCTP)
    ]
    
    IPC_BASE_DIR = "/tmp/dapps"
    E3_IPC_SETUP_PATH = f"{IPC_BASE_DIR}/setup"
    E3_IPC_SOCKET_PATH = f"{IPC_BASE_DIR}/e3_socket" # inbound
    DAPP_IPC_SOCKET_PATH = f"{IPC_BASE_DIR}/dapp_socket" # outbound

    @staticmethod
    def setup_connector(link_layer: str, transport_layer: str, id: int):
        link_layer = E3LinkLayer.from_string(link_layer)
        transport_layer = E3TransportLayer.from_string(transport_layer)
        
        # Check if the configuration is part of the valid configurations
        if (link_layer, transport_layer) not in E3Connector.VALID_CONFIGURATIONS:
            raise ValueError(
                f"Invalid configuration: Link Layer={link_layer}, Transport Layer={transport_layer}. "
                f"Must be one of {E3Connector.VALID_CONFIGURATIONS}"
            )
            
        print("SETUP CONNECTOR")
        if link_layer == E3LinkLayer.POSIX:
            return POSIXConnector(transport_layer, id)
        elif link_layer == E3LinkLayer.SCAPY:
            return SCAPYConnector(transport_layer, id)
        else:
            return ZMQConnector(transport_layer, id)
    
    @abstractmethod
    def send_setup_request(self, payload):
        pass
    
    @abstractmethod
    def setup_inbound_connection(self):
        pass
    
    @abstractmethod
    def receive(self) -> bytes:
        """Receive a byte-encoded payload based on the configuration."""
        pass
    
    @abstractmethod
    def setup_outbound_connection(self):
        pass

    @abstractmethod
    def send(self, payload: bytes, seq_number: int = None):
        pass
    
    @abstractmethod 
    def dispose(self):
        pass
     
class ZMQConnector(E3Connector):
    setup_context: zmq.Context
    inbound_context: zmq.Context
    outbound_context: zmq.Context

    def __init__(self, transport_layer: E3TransportLayer, id: int):
        self.setup_context = zmq.Context()
        
        match transport_layer:
            case E3TransportLayer.SCTP | E3TransportLayer.TCP:
                self.setup_endpoint = f"{transport_layer}://127.0.0.1:9990"
                self.inbound_endpoint = f"{transport_layer}://127.0.0.1:9991"
                self.outbound_endpoint = f"{transport_layer}://127.0.0.1:9999"  
            
            case E3TransportLayer.IPC:
                self.setup_endpoint = f"{transport_layer}://{self.E3_IPC_SETUP_PATH}"
                self.inbound_endpoint = f"{transport_layer}://{self.E3_IPC_SOCKET_PATH}"
                self.outbound_endpoint = f"{transport_layer}://{self.DAPP_IPC_SOCKET_PATH}"

            case _:
                raise ValueError(f'Unknown/Unsupported value for transport layer {transport_layer}')

        self.transport_layer = transport_layer
        self.id = id
        
    def send_setup_request(self, payload):
        # Lazy pirate pattern, i.e., reliability on the client side
        request_timeout = 1000
        request_retries = 5

        while request_retries > 0:
            setup_socket = self.setup_context.socket(zmq.REQ)
            setup_socket.connect(self.setup_endpoint)
            #e3_logger.debug("Send E3 Setup request")
            setup_socket.send(payload)

            if (setup_socket.poll(request_timeout) & zmq.POLLIN) != 0:
                reply = setup_socket.recv()
                #e3_logger.debug('ZMQ setup socket replied')
                return reply
            
            request_retries -= 1
            #e3_logger.error("ZMQ setup did not reply")
            setup_socket.setsockopt(zmq.LINGER, 0)
            setup_socket.close()
            #e3_logger.debug('Retrying to connect')
        
        raise ConnectionRefusedError('E3 Setup request procedure did not went through')
        

    def setup_inbound_connection(self):
        self.inbound_context = zmq.Context()
        self.inbound_socket = self.inbound_context.socket(zmq.SUB)        
        self.inbound_socket.setsockopt_string(zmq.SUBSCRIBE, "") # subscribe to all the messages
        self.inbound_socket.setsockopt(zmq.CONFLATE, 1)  # Keep only last message
        self.inbound_socket.connect(self.inbound_endpoint)

        self.poller = zmq.Poller()
        self.poller.register(self.inbound_socket, zmq.POLLIN)

    def receive(self) -> bytes:
        # The commented part should be decommented for measuring the effective performance of the control loop
        #start_time = time.perf_counter()
        #while True:
        #    
        #     # Poll the socket without blocking
        #    events = dict(self.poller.poll(0))
        #    
        #    if self.inbound_socket in events and events[self.inbound_socket] == zmq.POLLIN:
        #        with open(f"{LOG_DIR}/busy.txt", "a") as f:
        #            print(f"{time.perf_counter() - start_time}", file=f)
                rec = self.inbound_socket.recv()
                seq_number = int.from_bytes(bytes.fromhex(rec.hex()[14:18]), byteorder="little")
                rec = rec[:7] + rec[9:]
                dapp_logger.info(f"RECEIVED IQs | Thread {self.id} | Sequence Number {seq_number}")
                return rec,seq_number
        

    def setup_outbound_connection(self):
        self.outbound_context = zmq.Context()
        #self.outbound_socket = self.outbound_context.socket(zmq.PUB)
        #self.outbound_socket.bind(self.outbound_endpoint)
        ## Changes to allow for multiple dApps communicating simultaneously
        self.outbound_socket = self.outbound_context.socket(zmq.PUSH)
        self.outbound_socket.connect(self.outbound_endpoint)
        # ------
        if self.transport_layer == E3TransportLayer.IPC:
            os.chmod(self.DAPP_IPC_SOCKET_PATH, 0o666)
    
    def send(self, payload: bytes, seq_number: int = None):
        dapp_logger.info(f"SEND CONTROL | Thread {self.id}")
        seq_bytes = struct.pack('<I', seq_number if seq_number is not None else 0)
        self.outbound_socket.send(seq_bytes + payload)

    def dispose(self):
        if hasattr(self, "setup_context"):    
            self.setup_context.destroy()  
        if hasattr(self, "inbound_context"):   
            self.inbound_context.destroy()
        if hasattr(self, "outbound_context"):   
            self.outbound_context.destroy()

class POSIXConnector(E3Connector):
    CHUNK_SIZE = 8192
    
    def __init__(self, transport_layer: E3TransportLayer, id: int):   
        match transport_layer:
            case E3TransportLayer.SCTP | E3TransportLayer.TCP:
                self.setup_endpoint = ("192.168.100.1", 9990)
                self.inbound_endpoint = ("0.0.0.0", 9991)
                self.outbound_endpoint = ("192.168.100.1", 9999)
            
            case E3TransportLayer.IPC: 
                self.setup_endpoint = self.E3_IPC_SETUP_PATH
                self.inbound_endpoint = self.E3_IPC_SOCKET_PATH
                self.outbound_endpoint = self.DAPP_IPC_SOCKET_PATH

            case _:
                raise ValueError(f'Unknown/Unsupported value for transport layer {transport_layer}')
        
        print("INITIALIZING POSIX")
        print(self.setup_endpoint)
        self.transport_layer = transport_layer
        self.id = id
    
    def _create_socket(self):
        match self.transport_layer:
            case E3TransportLayer.SCTP:
                try:
                    import sctp
                except ModuleNotFoundError:
                    e3_logger.critical(
                        "SCTP selected as transport layer, but the optional dependency 'pysctp' is not installed.\n"
                        "Fix this by running:\n\n"
                        "    pip install 'dApps[network]'  # OR\n"
                        "    pip install 'dApps[all]'\n",
                        exc_info=True
                    )
                    exit(-1)
                sock = sctp.sctpsocket_tcp(socket.AF_INET)
            case E3TransportLayer.TCP:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            case E3TransportLayer.IPC:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            case _:
                raise ValueError(f'Unknown/Unsupported value for transport layer {self.transport_layer}')
        return sock
    
    def send_setup_request(self, payload):
        setup_socket = self._create_socket()

        try:
            setup_socket.connect(self.setup_endpoint)
            setup_socket.send(payload)
            reply, _ = self.receive_in_chunks(setup_socket)
        finally:
            setup_socket.close()
            
        return reply
    
    def setup_inbound_connection(self):
        self.inbound_socket = self._create_socket()
        self.inbound_socket.bind(self.inbound_endpoint)
        self.inbound_socket.listen(5)        
        self.inbound_connection, _ = self.inbound_socket.accept()

    def receive_in_chunks(self, conn):
        data = bytearray()
        chunks = 0 
        seq_number = -1

        # Receive the size of the buffer first
        raw_size = conn.recv(4)
        if len(raw_size) < 4:
            e3_logger.error("Failed to receive buffer size")
            return None
        buffer_size = struct.unpack("!I", raw_size)[0]
        #e3_logger.debug(f"buffer size is {buffer_size}")

        # Receive the buffer in chunks
        while len(data) < buffer_size-2:
            remaining = buffer_size - len(data)
            chunk = conn.recv(min(self.CHUNK_SIZE, remaining))
            if(chunks == 0 and len(chunk) > 4): 
                seq_number = int.from_bytes(bytes.fromhex(chunk.hex()[14:18]), byteorder="little")
                chunk = chunk[:7] + chunk[9:]
            if not chunk:
                e3_logger.error("Connection closed unexpectedly")
                return None  # Connection closed unexpectedly
            data.extend(chunk)
            chunks += 1

        #e3_logger.debug(f"Chunks recv {chunks}")
        #e3_logger.debug(f"Total size recv {len(data)}")
        return bytes(data), seq_number
    
    def receive(self) -> bytes:
        data,seq_number = self.receive_in_chunks(self.inbound_connection) 
        dapp_logger.info(f"RECEIVED IQs | Thread {self.id} | Sequence Number {seq_number}")
        return data,seq_number

    def setup_outbound_connection(self):
        self.outbound_socket = self._create_socket()
        self.outbound_socket.connect(self.outbound_endpoint)

    def send(self, payload: bytes, seq_number: int = None):
        dapp_logger.info(f"SEND CONTROL | Thread {self.id}")
        seq_bytes = struct.pack('<I', seq_number if seq_number is not None else 0)
        self.outbound_socket.send(seq_bytes + payload)
    
    def dispose(self):
        if hasattr(self, "outbound_socket"):
            self.outbound_socket.close()
        if hasattr(self, "inbound_connection"):
            self.inbound_connection.close()
        if hasattr(self, "inbound_socket"):
            self.inbound_socket.close()
        
        if self.transport_layer == E3TransportLayer.IPC:
            os.remove(self.E3_IPC_SETUP_PATH)    
            os.remove(self.E3_IPC_SOCKET_PATH)    
            os.remove(self.DAPP_IPC_SOCKET_PATH)             


class SCAPYConnector(POSIXConnector):
    CHUNK_SIZE = 8192
    def __init__(self, transport_layer, id):
        match transport_layer:
            case E3TransportLayer.SCTP | E3TransportLayer.TCP:
                self.setup_endpoint = ("192.168.100.1", 9990)
                self.inbound_endpoint = ("0.0.0.0", 9991)
                self.outbound_endpoint = ("192.168.100.1", 9999)
            
            case E3TransportLayer.IPC: 
                self.setup_endpoint = self.E3_IPC_SETUP_PATH
                self.inbound_endpoint = self.E3_IPC_SOCKET_PATH
                self.outbound_endpoint = self.DAPP_IPC_SOCKET_PATH

            case _:
                raise ValueError(f'Unknown/Unsupported value for transport layer {transport_layer}')
        
        self.transport_layer = transport_layer
        self.id = id
        self.interface = "p0" 

    def _create_socket(self):
        match self.transport_layer:
            case E3TransportLayer.SCTP:
                try:
                    import sctp
                except ModuleNotFoundError:
                    e3_logger.critical(
                        "SCTP selected as transport layer, but the optional dependency 'pysctp' is not installed.\n"
                        "Fix this by running:\n\n"
                        "    pip install 'dApps[network]'  # OR\n"
                        "    pip install 'dApps[all]'\n",
                        exc_info=True
                    )
                    exit(-1)
                sock = sctp.sctpsocket_tcp(socket.AF_INET)
            case E3TransportLayer.TCP:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            case E3TransportLayer.IPC:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            case _:
                raise ValueError(f'Unknown/Unsupported value for transport layer {self.transport_layer}')
        return sock
    
    def send_setup_request(self, payload):
        setup_socket = self._create_socket()

        try:
            setup_socket.connect(self.setup_endpoint)
            setup_socket.send(payload)
            reply, _ = self.receive_in_chunks(setup_socket)
        finally:
            setup_socket.close()
            
        return reply
    
    def setup_inbound_connection(self):
        self.inbound_socket = self._create_socket()
        self.inbound_socket.bind(self.inbound_endpoint)
        self.inbound_socket.listen(5)        
        self.inbound_connection, _ = self.inbound_socket.accept()

    def receive_in_chunks(self, conn):
        data = bytearray()
        chunks = 0 
        seq_number = -1

        # Receive the size of the buffer first
        raw_size = conn.recv(4)
        if len(raw_size) < 4:
            e3_logger.error("Failed to receive buffer size")
            return None
        buffer_size = struct.unpack("!I", raw_size)[0]
        #e3_logger.debug(f"buffer size is {buffer_size}")

        # Receive the buffer in chunks
        while len(data) < buffer_size-2:
            remaining = buffer_size - len(data)
            chunk = conn.recv(min(self.CHUNK_SIZE, remaining))
            if(chunks == 0 and len(chunk) > 4): 
                seq_number = int.from_bytes(bytes.fromhex(chunk.hex()[14:18]), byteorder="little")
                chunk = chunk[:7] + chunk[9:]
            if not chunk:
                e3_logger.error("Connection closed unexpectedly")
                return None  # Connection closed unexpectedly
            data.extend(chunk)
            chunks += 1

        #e3_logger.debug(f"Chunks recv {chunks}")
        #e3_logger.debug(f"Total size recv {len(data)}")
        return bytes(data), seq_number
    
    #def receive(self) -> bytes:
    #    data,seq_number = self.receive_in_chunks(self.inbound_connection) 
    #    dapp_logger.info(f"RECEIVED IQs | Thread {self.id} | Sequence Number {seq_number}")
    #    return data,seq_number

    def receive(self):
        src_host = "10.50.1.2"
        # Sniff packets from the specified interface
        packets = sniff(iface=self.interface, count=1, filter=f"host {src_host}")
        for packet in packets:
            if Ether in packet:
                # Process the packet as needed
                if(TCP in packet): 
                    payload_size = len(bytes(packet[TCP].payload))
                    print(f"Payload_size {payload_size}")
                    if(payload_size == 0):
                        return bytes(10),1
                    return bytes(packet[TCP].payload),1  # Return the raw bytes of the packet
            return None

    def setup_outbound_connection(self):
        self.outbound_socket = self._create_socket()
        self.outbound_socket.connect(self.outbound_endpoint)

    def send(self, payload: bytes, seq_number: int = None):
        dapp_logger.info(f"SEND CONTROL | Thread {self.id}")
        seq_bytes = struct.pack('<I', seq_number if seq_number is not None else 0)
        self.outbound_socket.send(seq_bytes + payload)
    
    def dispose(self):
        if hasattr(self, "outbound_socket"):
            self.outbound_socket.close()
        if hasattr(self, "inbound_connection"):
            self.inbound_connection.close()
        if hasattr(self, "inbound_socket"):
            self.inbound_socket.close()
        
        if self.transport_layer == E3TransportLayer.IPC:
            os.remove(self.E3_IPC_SETUP_PATH)    
            os.remove(self.E3_IPC_SOCKET_PATH)    
            os.remove(self.DAPP_IPC_SOCKET_PATH)
