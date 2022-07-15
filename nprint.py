import subprocess
from pathlib import Path
paths=['/home/pcl/PangBo/pro/GNNTrafficClassification/Dataset/Splite_Session/app/Test/']
data_dir_path = paths[0]
data_dir_path=Path(data_dir_path)
for pcap_file in sorted(data_dir_path.iterdir()):
    name=pcap_file.name
    p = subprocess.Popen('nprint -P /home/pcl/PangBo/pro/GNNTrafficClassification/Dataset/Splite_Session/app/Test/'+name+' -p 20 ', shell=True, stdout=subprocess.PIPE)
out, err = p.communicate()

for line in out.splitlines():
    print(line)
