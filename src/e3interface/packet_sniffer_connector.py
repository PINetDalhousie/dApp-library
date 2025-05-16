from .posix_connector import POSIXConnector
from scapy.all import sniff, Ether

class PacketSnifferConnector(POSIXConnector):
    def __init__(self, transport_layer, id, interface):
        super().__init__(transport_layer, id)
        self.interface = interface

    def setup_inbound_connection(self):
        # Instead of a traditional socket, we will set up packet sniffing
        self.inbound_socket = None  # Placeholder, as we won't use a socket here

    def receive(self):
        # Sniff packets from the specified interface
        packets = sniff(iface=self.interface, count=1)
        for packet in packets:
            if Ether in packet:
                # Process the packet as needed
                return bytes(packet)  # Return the raw bytes of the packet

    def dispose(self):
        # No specific disposal needed for packet sniffing
        pass

    def send(self, payload: bytes, seq_number: int = None):
        # Use the POSIXConnector's send method
        super().send(payload, seq_number)
