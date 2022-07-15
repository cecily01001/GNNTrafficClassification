from pathlib import Path
from scapy.layers.dns import DNS
from scapy.layers.inet import TCP
from scapy.packet import Padding
from scapy.utils import rdpcap
# for app identification
PREFIX_TO_ENTROPY_ID = {
    'AIMchat2': 0,
    'email1b': 1,
    'facebook_audio2a': 2,
    # 'ftps_up_2a': 3,
    'ftps_up_2b': 3,
    'hangouts_audio3': 4,
    # 'hangouts_audio4': 4,
    # 'scpdown1': 5,
    'scpdown5': 5,
    'sftp_up_2a': 6,
    # 'skype_chat1a': 7,
    'skype_video1a': 7,
    'toryoutube2': 8,
    'vimeo': 9,
    'youtube2': 10,
}
ID_TO_ENTROPY = {
    0: 'AIM Chat',
    1: 'Email',
    2: 'Facebook',
    3: 'FTPS',
    4: 'Hangouts',
    5: 'SCP',
    6: 'SFTP',
    7: 'Skype',
    8: 'Tor',
    9: 'Vimeo',
    10: 'Youtube',
}
def read_pcap(path: Path):
    packets = rdpcap(str(path))

    return packets


def should_omit_packet(packet):
    # SYN, ACK or FIN flags set to 1 and no payload
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    # DNS segment
    if DNS in packet:
        return True

    return False
