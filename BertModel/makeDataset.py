import csv
from pathlib import Path

train_add = "/home/user1/PangBo/GNNTrafficClassification/Dataset/Text/app/Train/"
test_add = "/home/user1/PangBo/GNNTrafficClassification/Dataset/Text/app/Test/"
dev_add = "/home/user1/PangBo/GNNTrafficClassification/Dataset/Text/app/valid/"
train_tsv = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/app/train.tsv"
test_tsv = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/app/test.tsv"
dev_tsv = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/app/dev.tsv"

train_add_en = "/home/user1/PangBo/GNNTrafficClassification/Dataset/Text/en/train/"
test_add_en = "/home/user1/PangBo/GNNTrafficClassification/Dataset/Text/en/test/"
dev_add_en = "/home/user1/PangBo/GNNTrafficClassification/Dataset/Text/en/valid/"
train_tsv_en = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/en/train.tsv"
test_tsv_en = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/en/test.tsv"
dev_tsv_en = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/en/dev.tsv"
# train_tsv_bar= open(train_tsv, 'w', encoding='utf-8', newline="")
# tsv_train = csv.writer(train_tsv_bar, delimiter='\t')

test_tsv_bar= open(dev_tsv_en, 'w', encoding='utf-8', newline="")
tsv_test = csv.writer(test_tsv_bar, delimiter='\t')
def makeTSVdataset(data_dir_path):

    # app_label = PREFIX_TO_TRAFFIC_ID.get(prefix)

    for pcap_path in sorted(data_dir_path.iterdir()):
        app_prefix = pcap_path.name.split('.')[0]
        print(app_prefix)
        try:
            f = open(pcap_path, "r", encoding="utf-8")
            l = f.read()
            text_packets = l.split(' Encapsulation type:')
            text_packets = text_packets[1:]
            for i in range(len(text_packets)):
                text_packets[i] = text_packets[i][0:5000]
                tsv_test.writerow([text_packets[i], app_prefix])
        except:
            print('error')
train_add_en = Path(train_add_en)
test_add_en = Path(test_add_en)
dev_add_en = Path(dev_add_en)
makeTSVdataset(dev_add_en)