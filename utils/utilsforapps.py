from pathlib import Path
from scapy.layers.dns import DNS
from scapy.layers.inet import TCP
from scapy.packet import Padding
from scapy.utils import rdpcap

PREFIX_TO_APP_ID = {
    'ppstream': 0,
    'qqmusic': 1,
    'sbs_gorealra': 2,
    'scribd': 3,
    'tunein_radio': 4,
    'youku': 5,
    'vsee': 6,
    'icq': 7,
    'imo': 8,
    'whatsapp': 9,
    'alibaba': 10,
    'any': 11,
    'espn_star_sports': 12,
    'firefox': 13,
    'flipboard': 14,
    'google': 15,
    'hupu': 16,
    'speedtest': 17,
    'yahoo': 18,
    'yandex': 19,
    'teamviewer': 20,
    'fubar': 21,
    'pinterest': 22,
    'travelzoo': 23,
    'nbc': 24,
    'startv': 25,
    'moneycontrol': 26,
    'mlb': 27,
    'nhl_gamecenter':28,
    'chrome':29,
    'dictionary':30,
    'yahoo_messenger':31,
    'itunes':32,
    'kik':33,
    'afreecatv':34,
    'skype':35,
    'usa_today':36,
    'mega':37,
    'tagged':38,
    'camfrog':39,
    'amazon_instant_video':40,
}

ID_TO_APP = {
    0:'ppstream',
    1:'qqmusic',
    2:'sbs_gorealra',
    3:'scribd',
    4:'tunein_radio',
    5:'youku',
    6:'vsee',
    7:'icq',
    8:'imo',
    9:'whatsapp',
    10:'alibaba',
    11:'any',
    12:'espn_star_sports',
    13:'firefox',
    14:'flipboard',
    15:'google',
    16:'hupu',
    17:'speedtest',
    18:'yahoo',
    19:'yandex',
    20:'teamviewer',
    21:'fubar',
    22:'pinterest',
    23:'travelzoo',
    24:'nbc',
    25:'startv',
    26:'moneycontrol',
    27:'mlb',
    28:'nhl_gamecenter',
    29:'chrome',
    30:'dictionary',
    31:'yahoo_messenger',
    32:'itunes',
    33:'kik',
    34:'afreecatv',
    35:'skype',
    36:'usa_today',
    37:'mega',
    38:'tagged',
    39:'camfrog',
    40:'amazon_instant_video',
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
