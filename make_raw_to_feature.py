import csv
import click
import torch
from pathlib import Path
import numpy as np
from scapy.all import *
from scapy.layers.inet import TCP
from tqdm import tqdm
from BertModel.Utils.utils import classifiction_metric
from utils.utilsformalware import read_pcap, PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID
from utils.utilsforentropy import PREFIX_TO_ENTROPY_ID, en_list
from utils.processPcapHeader import remove_ether_header, mask_ip, pad_udp, packet_to_sparse_array
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from BertModel.Utils.Classifier_utils import InputExample, convert_examples_to_features, convert_features_to_tensors
from BertModel.BertCNN.BertCNN import BertCNN
from BertModel.BertCNN import args

bertModelPath = '/home/user1/PangBo/GNNTrafficClassification/BertModelResult'
output_model_file = '/home/user1/PangBo/GNNTrafficClassification/BertModelResult/pytorch_model.bin'
output_config_file = '/home/user1/PangBo/GNNTrafficClassification/BertModelResult/config.json'

data_dir = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/app"
entropy_dir = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/en"
output_dir = "BertModelResult/"
cache_dir = ".sst_cache/"
log_dir = ".sst_log/"
bert_vocab_file = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/pytorch_Bert/bert-large-uncased-vocab.txt"
bert_model_dir = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/pytorch_Bert/bert-large-uncased"

bertConfig = args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir)

tokenizer = BertTokenizer.from_pretrained(
    bertConfig.bert_vocab_file, do_lower_case=bertConfig.do_lower_case)  # 分词器选择

label_list = ['aim',
                  'facebook',
                  'gmailchat',
                  'hangouts',
                  'icq',
                  'netflix',
                  'skype',
                  'spotify',
                  'thunderbird',
                  'vimeo',
                  'youtube']

bert_config = BertConfig(output_config_file)
filter_sizes = [int(val) for val in bertConfig.filter_sizes.split()]
model = BertCNN(bert_config, num_labels=len(label_list),n_filters=bertConfig.filter_num, filter_sizes=filter_sizes)
model.load_state_dict(torch.load(output_model_file))
model.cuda()

def load_data(pcapList, label_list, label):
    examples = []
    for (i, line) in enumerate(pcapList):
        # print(line)
        # if i == 0:
        #     continue
        guid = i
        text_a = line
        label = label
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    features = convert_examples_to_features(
        examples, label_list, bertConfig.max_seq_length, tokenizer)
    examples_len = len(examples)
    dataloader = convert_features_to_tensors(features, examples_len, "test")
    # print('load data')
    # print(len(dataloader))
    return dataloader, examples_len


def load_model(label_list):
    bert_config = BertConfig(output_config_file)
    filter_sizes = [int(val) for val in bertConfig.filter_sizes.split()]
    model = BertCNN(bert_config, num_labels=len(label_list),
                    n_filters=bertConfig.filter_num, filter_sizes=filter_sizes)
    model.load_state_dict(torch.load(output_model_file))
    return model


def transform_packet(packet):
    # if should_omit_packet(packet):
    #     return None
    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)
    arr = packet_to_sparse_array(packet)
    return arr


def evaluate_save(model, dataloader, criterion, label_list):
    model.eval()

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    all_idxs = np.array([], dtype=int)
    all_preds_text=[]
    epoch_loss = 0

    for idxs, input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Eval"):
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        label_ids = label_ids.cuda()

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

        preds = logits.detach().cpu().numpy()
        # all_preds_text.append(list(preds.reshape(len(preds[0]))))
        # print(preds)
        all_preds_text.append(list(preds))
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

        idxs = idxs.detach().cpu().numpy()
        all_idxs = np.append(all_idxs, idxs)

        epoch_loss += loss.mean().item()

    acc, report, auc = classifiction_metric(all_preds, all_labels, label_list)
    return epoch_loss / len(dataloader), acc, report, auc, all_idxs, all_labels, all_preds, all_preds_text


def get_text_feature(pcapList, label_list, label):

    test_dataloader, _ = load_data(pcapList, label_list, label)
    # model = load_model(label_list)
    # model.cuda()
    res_path = '/home/user1/PangBo/GNNTrafficClassification/BertModel/result/app_precision_recall.csv'
    res_file_temp = open(res_path, 'w', encoding='utf-8', newline="")
    graph_file = csv.writer(res_file_temp)
    # 损失函数准备
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # test the model
    test_loss, test_acc, test_report, test_auc, all_idx, all_labels, all_preds, all_preds_text = evaluate_save(
        model, test_dataloader, criterion, label_list)
    # print('text feature')
    # print(all_preds_text)
    return all_preds_text


def transform_pcap(path, label_dic, list, kind, datakind):
    rows = []
    direcs = []
    prefix = path.name.split('.')[0]
    app_label = label_dic.get(prefix)
    # 查看是否能够成功读取原始pcap文件
    try:
        packets_list = read_pcap(path)
    except:
        print('read error')
        return

    save_dir = os.path.join('Dataset/Processed_data/' + kind + '/', datakind)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    graph_path = os.path.join(save_dir + '/' + str(len(packets_list)) + '_' + str(app_label) + '_' + path.name + '.csv')
    graph_file_temp = open(graph_path, 'w', encoding='utf-8', newline="")
    graph_file = csv.writer(graph_file_temp)

    p1 = int(path.name.split('.')[2].split('_')[2])
    text_path = str(path).split('Splite_Session')[0] + 'Text' + str(path).split('Splite_Session')[1] + '.txt'
    print(text_path)
    try:
        f = open(text_path, "r", encoding="utf-8")
        l = f.read()
        text_packets = l.split(' Encapsulation type:')
        text_packets = text_packets[1:]
        for i in range(len(text_packets)):
            text_packets[i] = text_packets[i][0:5000]
    except:
        print('splite error')
        return
    for i, packet in enumerate(packets_list):
        direc = -1
        if TCP in packet:
            sport = packet[TCP].sport
            if sport == p1:
                direc = 0
            else:
                direc = 1
        direcs.append(direc)
        arr = transform_packet(packet)
        # print(arr)
        # if arr is not None:
        arr_feature = arr.todense().tolist()[0]
        rows.append(arr_feature)
    texts = get_text_feature(text_packets, label_list, prefix)

    if len(texts) == len(rows):
        if len(texts) >= 1:
            for i in range(len(texts)):
                graph_file.writerow([','.join(map(str, rows[i])), ','.join(map(str, texts[i])), direcs[i]])
        if len(texts) != len(packets_list):
            graph_path_fix = os.path.join(
                save_dir + '/' + str(len(texts)) + '_' + str(app_label) + '_' + path.name + '.csv')
            os.rename(graph_path, graph_path_fix)
        print(path, 'Done')
    else:
        print(len(texts))
        print(len(rows))
        print(path, 'Error')


def make_feature(data_dir_path, valid_data_dir_path, testdata_dir_path, kind):
    testdata_dir_path = Path(testdata_dir_path)
    valid_data_dir_path = Path(valid_data_dir_path)
    data_dir_path = Path(data_dir_path)
    if kind == 'app':
        label_dic = PREFIX_TO_APP_ID
        label_list = []
    elif kind == 'malware':
        label_dic = PREFIX_TO_Malware_ID
        label_list = []
    else:
        label_dic = PREFIX_TO_ENTROPY_ID
        label_list = en_list
    os.makedirs('Dataset/Processed_data/' + kind)
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap(pcap_path, label_dic, label_list, kind, 'train')
    for pcap_path in sorted(valid_data_dir_path.iterdir()):
        transform_pcap(pcap_path, label_dic, label_list, kind, 'valid')
    for pcap_path in sorted(testdata_dir_path.iterdir()):
        transform_pcap(pcap_path, label_dic, label_list, kind, 'test')


@click.command()
@click.option('-k', '--kind', help='object to be classificated', required=True)
def main(kind):
    if kind == 'app':
        make_feature("Dataset/Splite_Session/app/Train", "Dataset/Splite_Session/app/valid",
                     "Dataset/Splite_Session/app/Test", kind)

    elif kind == 'malware':
        make_feature("Dataset/Splite_Session/Malware/trainset", "Dataset/Splite_Session/Malware/validset",
                     "Dataset/Splite_Session/Malware/testset", kind)

    else:
        make_feature("Dataset/Splite_Session/ENTROPY2/Train", "Dataset/Splite_Session/ENTROPY2/valid",
                     "Dataset/Splite_Session/ENTROPY2/Test", kind)


if __name__ == '__main__':
    main()
