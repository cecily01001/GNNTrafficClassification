import click
import dgl
import numpy
import torch
from pathlib import Path
import numpy as np
from scapy.compat import raw
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
from utils.utilsformalware import should_omit_packet, read_pcap, PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID
from utils.utilsforentropy import PREFIX_TO_ENTROPY_ID
from dgl.data.utils import save_graphs

from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
cnames = {
# 'aliceblue':            '#F0F8FF',
# 'antiquewhite':         '#FAEBD7',
# 'aqua':                 '#00FFFF',
# 'aquamarine':           '#7FFFD4',
# 'azure':                '#F0FFFF',
# 'beige':                '#F5F5DC',
# 'bisque':               '#FFE4C4',
# 'black':                '#000000',
# 'blanchedalmond':       '#FFEBCD',
# 'blue':                 '#0000FF',
# 'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
# 'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
# 'coral':                '#FF7F50',
# 'cornflowerblue':       '#6495ED',
# 'cornsilk':             '#FFF8DC',
# 'crimson':              '#DC143C',
# 'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
# 'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

def plot_embedding_2D(data, label):
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    vis_x = data[:, 0]  # 0维
    vis_y = data[:, 1]  # 1维
    father_list = []
    color_list=[]
    fig = plt.figure()
    # for i in range(41):
    #     temp_list = []
    #     father_list.append(temp_list)
    # for j in range(41):
    #     father_list[j]= [i for i in range(len(label)) if label == j]
    # for i in range(41):
    #     color = plt.cm.Set1(i)
    #     color_list.append(color)
    # print(color_list)
    # for i in range(41):
    #     plt.scatter(vis_x[father_list[i]], vis_y[father_list[i]], c=color_list[i], cmap='brg', label=i)

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    color_list=[]
    for i in cnames:
        color_list.append(i)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 # color=plt.cm.Set1(label[i]),
                 color=color_list[label[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    return fig
def get_sne_data(label, feature_array):  # Input_path为你自己原始数据存储路径，我的路径就是上面的'./Images'

    data = np.zeros((len(label), 1500))  # 初始化一个np.array数组用于存数据
    label_np = np.zeros((len(label),))  # 初始化一个np.array数组用于存数据
    for k in range(len(label)):
        label_np[k] = label[k]
    print(label_np)
    for i in range(len(label)):
        data[i] = feature_array[i]
        n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'

    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = '\x00' * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet


def packet_to_sparse_array(packet, max_length=1500):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length]
    # arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    # print(arr)
    arr = sparse.csr_matrix(arr)
    return arr


def transform_packet(packet):
    if should_omit_packet(packet):
        return None

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)

    arr = packet_to_sparse_array(packet)

    return arr


def transform_pcap_malware(path, listofnodenum_test, listofkind_test, feature_matrix_test):
    # output_path = 'processdata/' + path.name
    # if Path(output_path + '_SUCCESS').exists():
    #     return
    # with Path(output_path + '_SUCCESS').open('w') as f:
    #     f.write('')
    # print(path,"Done")
    # print(path)
    # try:
    #     read_pcap(path)
    # except Scapy_Exception:
    #     os.remove(path)
    #     return
    rows = []
    j = 0
    for i, packet in enumerate(read_pcap(path)):
        arr = transform_packet(packet)
        if arr is not None:
            # get labels for app identification
            prefix = path.name.split('.')[0]
            app_label = PREFIX_TO_Malware_ID.get(prefix)
            j = j + 1
            rows.append(arr.todense().tolist()[0])

    listofnodenum_test.append(j)
    listofkind_test.append(app_label)
    feature_matrix_test.append(rows)
    print(path, 'Done')

def transform_pcap(path, listofnodenum, listofkind, feature_matrix):
    # try:
    #     read_pcap(path)
    # except Scapy_Exception:
    #     os.remove(path)
    #     print(path)
    #     return
    # else:
    #     with Path(output_path + '_SUCCESS').open('w') as f:
    #         f.write('')
    rows = []
    j = 0
    for i, packet in enumerate(read_pcap(path)):
        arr = transform_packet(packet)
        if arr is not None:
            # get labels for app identification
            prefix = path.name.split('.')[0]
            # app_label = PREFIX_TO_APP_ID.get(prefix)
            app_label = PREFIX_TO_ENTROPY_ID.get(prefix)
            j = j + 1
            rows.append(arr.todense().tolist()[0])
    if j>1:
        if app_label:
            rows=np.array(rows)
            mean_row=rows.mean(axis=0)  # 列
            # mean_row = np.array(mean_row)
            listofnodenum.append(j)
            listofkind.append(app_label)
            feature_matrix.append(mean_row)
            print(path, 'Done')
        else:
            print(prefix)
            print(app_label)
            rows=np.array(rows)
            mean_row=rows.mean(axis=0)  # 列
            # mean_row = np.array(mean_row)
            listofnodenum.append(j)
            listofkind.append(app_label)
            feature_matrix.append(mean_row)
            print(path, 'Done')


class graphdataset(object):

    def __init__(self, num_of_graphnode, listofkind, feature_matrix,num_classes):
        super(graphdataset, self).__init__()
        graphlist = []
        k = 0
        for i in num_of_graphnode:
            list_point = []
            list_point_2 = []
            for j in range(i - 1):
                list_point.append(j)
                list_point_2.append(j + 1)
            for j in range(i - 1):
                list_point.append(j + 1)
                list_point_2.append(j)
            g = dgl.graph((list_point, list_point_2))
            g = dgl.add_self_loop(g)
            # print(i)
            g.ndata['h'] = torch.tensor(feature_matrix[k])
            # print(g)
            k = k + 1
            graphlist.append(g)
        # print(list_point)
        self.graphs = graphlist
        self.num_classes = num_classes
        self.labels = listofkind
        # self.feature_matrix=feature_matrix

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]


def make_app_graph(data_dir_path, valid_data_dir_path, testdata_dir_path):
    testdata_dir_path = Path(testdata_dir_path)
    valid_data_dir_path = Path(valid_data_dir_path)
    data_dir_path = Path(data_dir_path)
    listofkind = []
    listofnodenum = []
    feature_matrix = []
    listofkind_valid = []
    listofnodenum_valid = []
    feature_matrix_valid = []
    listofkind_test = []
    listofnodenum_test = []
    feature_matrix_test = []
    num_classes=len(PREFIX_TO_ENTROPY_ID)
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap(pcap_path, listofnodenum, listofkind, feature_matrix)

    data, label, n_samples, n_features = get_sne_data(listofkind,feature_matrix)
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)  # 调用TSNE
    result_2D = tsne_2D.fit_transform(data)
    fig1 = plot_embedding_2D(result_2D, label)
    # plt.show(fig1)
    fig1.savefig('entropy_t_sne.png')
    # result=[0]*41
    # num=[0]*41
    # length=len(listofnodenum)
    # dif_num=0
    # dif_sum=0
    # dif_num0 = 0
    # dif_sum0 = 0
    # for i in range(length):
    #     label1=listofkind[i]
    #     num[label1]=num[label1]+1
    #     vec1 = feature_matrix[i]
    #     for j in range(length):
    #         label2=listofkind[j]
    #         vec2=feature_matrix[j]
    #         dist = numpy.linalg.norm(vec1 - vec2)
    #         if label1==label2:
    #             if dist!=0:
    #                 result[label1]=result[label1]+dist
    #                 dif_num0 = dif_num0 + 1
    #                 dif_sum0 = dif_sum0 + dist
    #
    #         else:
    #             if dist != 0:
    #                 dif_num = dif_num + 1
    #                 dif_sum = dif_sum + dist
    # print(result)
    # print(num)
    # for i in range(41):
    #     result[i]=result[i]/num[i]
    # print(result)
    # print(dif_sum0)
    # print(dif_num0)
    # print(dif_sum0 / dif_num0)
    # print(dif_sum)
    # print(dif_num)
    # print(dif_sum/dif_num)

    # for pcap_path in sorted(valid_data_dir_path.iterdir()):
    #     transform_pcap(pcap_path, listofnodenum_valid, listofkind_valid, feature_matrix_valid)
    # for pcap_path in sorted(testdata_dir_path.iterdir()):
    #     transform_pcap(pcap_path, listofnodenum_test, listofkind_test, feature_matrix_test)

    # trainset = graphdataset(listofnodenum, listofkind, feature_matrix,num_classes)
    # validset = graphdataset(listofnodenum_valid, listofkind_valid, feature_matrix_valid,num_classes)
    # testset = graphdataset(listofnodenum_test, listofkind_test, feature_matrix_test,num_classes)
    # graph_labels = {"graph_sizes": torch.tensor()}
    # save_graphs("Graphs/APP/trainset.bin", trainset.graphs,
    #             {'labels': torch.tensor(trainset.labels)})
    # save_graphs("Graphs/APP/validset.bin", validset.graphs,
    #             {'labels': torch.tensor(validset.labels)})
    # save_graphs("Graphs/APP/testset.bin", testset.graphs,
    #             {'labels': torch.tensor(testset.labels)})

def make_malware_graph(data_dir_path, valid_data_dir_path, testdata_dir_path):
    testdata_dir_path = Path(testdata_dir_path)
    valid_data_dir_path = Path(valid_data_dir_path)
    data_dir_path = Path(data_dir_path)
    listofkind = []
    listofnodenum = []
    feature_matrix = []
    listofkind_valid = []
    listofnodenum_valid = []
    feature_matrix_valid = []
    listofkind_test = []
    listofnodenum_test = []
    feature_matrix_test = []
    num_classes = len(PREFIX_TO_ENTROPY_ID)
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap_malware(pcap_path, listofnodenum, listofkind, feature_matrix)
    for pcap_path in sorted(valid_data_dir_path.iterdir()):
        transform_pcap_malware(pcap_path, listofnodenum_valid, listofkind_valid, feature_matrix_valid)
    for pcap_path in sorted(testdata_dir_path.iterdir()):
        transform_pcap_malware(pcap_path, listofnodenum_test, listofkind_test, feature_matrix_test)
    trainset = graphdataset(listofnodenum, listofkind, feature_matrix,num_classes)
    validset = graphdataset(listofnodenum_valid, listofkind_valid, feature_matrix_valid,num_classes)
    testset = graphdataset(listofnodenum_test, listofkind_test, feature_matrix_test,num_classes)
    # graph_labels = {"graph_sizes": torch.tensor()}
    save_graphs("Graphs/Malware/trainset.bin", trainset.graphs,
                {'labels': torch.tensor(trainset.labels)})
    save_graphs("Graphs/Malware/validset.bin", validset.graphs,
                {'labels': torch.tensor(validset.labels)})
    save_graphs("Graphs/Malware/testset.bin", testset.graphs,
                {'labels': torch.tensor(testset.labels)})


@click.command()
@click.option('-k', '--kind', help='object to be classificated', required=True)
def main(kind):
    if kind == 'app':
        make_app_graph("Dataset/Splite_Session/Entropy/train", "Dataset/Splite_Session/app/valid", "Dataset/Splite_Session/app/Test")
    elif kind == 'malware':
        make_malware_graph("Dataset/Malware/trainset", "Dataset/Malware/validset", "Dataset/Malware/testset")


if __name__ == '__main__':
    main()
