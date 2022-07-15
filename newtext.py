import os
from pathlib import Path
paths=['/home/user1/PangBo/GNNTrafficClassification/Dataset/Splite_Session/ENTROPY2/Train/']
# paths=['/home/user1/PangBo/GNNTrafficClassification/Dataset/Splite_Session/en/']

dst=['/home/user1/PangBo/GNNTrafficClassification/Dataset/Text/ENTROPY2/Train/']
data_dir_path = paths[0]
data_dir_path=Path(data_dir_path)
for pcap_file in sorted(data_dir_path.iterdir()):
    name=pcap_file.name
    print(name)
    p = os.system('tshark -T text -V -x -r '+paths[0]+name+' > '+dst[0]+name+'.txt')
    # p = os.system('tshark -T text -V -r ' + paths[0] + name + ' > ' + dst[0] + name + '.txt')

