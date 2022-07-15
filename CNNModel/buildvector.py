import os
from torchtext.legacy import data
import re
from torchtext.vocab import Vectors
import dill
import torch
from spacy.lang.en import English
spacy_en = English()

def tokenizeren(text): # create a tokenizer function
    # regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    # text = regex.sub(' ', text)
    # return [word for word in jieba.cut(text) if word.strip()]
    return [tok.text for tok in spacy_en.tokenizer(text)]

def dump_examples(train, dev, test):
    # 保存examples
    if not os.path.exists('Dataset/textSplitDataset/train_examples'):
        with open('Dataset/textSplitDataset/train_examples', 'wb')as f:
            dill.dump(train.examples, f)
    if not os.path.exists('Dataset/textSplitDataset/dev_examples'):
        with open('Dataset/textSplitDataset/dev_examples', 'wb')as f:
            dill.dump(dev.examples, f)
    if not os.path.exists('Dataset/textSplitDataset/test_examples'):
        with open('Dataset/textSplitDataset/test_examples', 'wb')as f:
            dill.dump(test.examples, f)
snli_split_path_lst=['Dataset/textSplitDataset/train_examples','Dataset/textSplitDataset/dev_examples','Dataset/textSplitDataset/test_examples']
def if_split_already():
    for path in snli_split_path_lst:
        if not os.path.exists(path):
            return False
    return True

def load_split_datasets(fields):
    # 加载examples
    with open('Dataset/textSplitDataset/train_examples', 'rb')as f:
        train_examples = dill.load(f)
    with open('Dataset/textSplitDataset/dev_examples', 'rb')as f:
        dev_examples = dill.load(f)
    # with open('Dataset/textSplitDataset/test_examples', 'rb')as f:
    #     test_examples = dill.load(f)

    # 恢复数据集
    train = data.Dataset(examples=train_examples, fields=fields)
    dev = data.Dataset(examples=dev_examples, fields=fields)
    # test = data.Dataset(examples=test_examples, fields=fields)
    return train, dev
def load_data(args):
    # print('加载数据中...')
    '''
    如果需要设置文本的长度，则设置fix_length,否则torchtext自动将文本长度处理为最大样本长度
    text = data.Field(sequential=True, tokenize=tokenizer, fix_length=args.max_len, stop_words=stop_words)
    '''
    text = data.Field(sequential=True, lower=True, tokenize=tokenizeren)
    label = data.Field(sequential=False)

    text.tokenize = tokenizeren
    if if_split_already()==False:
        print('saving trainset')
        train = data.TabularDataset(
                path='sampletrain.csv',
                skip_header=True,
                # train='sampletrain.tsv',
                format='csv',
                fields=[('label', label), ('text', text)],
            )
        print('saving validset')
        valid = data.TabularDataset(
            path='samplevalid.csv',
            skip_header=True,
            # train='sampletrain.tsv',
            format='csv',
            fields=[('label', label), ('text', text)],
        )
        print('saving testset')
        test= data.TabularDataset(
            path='sampletest.csv',
            skip_header=True,
            # train='sampletrain.tsv',
            format='csv',
            fields=[('label', label), ('text', text)],
        )
        dump_examples(train,valid,test)
    else:
        print('loading exist dataset')
        fields = [('text', text),('label', label)]
        train, valid=load_split_datasets(fields)
    print('已经搭载了train：')
    print(train[0].text[0:14])
    print(train[1].text[0:14])
    print(train[2].text[0:14])
    print('\n')
    # print(train[1].label, train[1].text)
    print("build vocab")
    # 创建vocab
    text.build_vocab(train)
    label.build_vocab(train)

    # 加载已有的词向量
    # with open("data/pcap.vector", 'wb')as f:
    #     dill.dump(text.vocab, f)
    # with open("data/label.vector", 'wb')as f:
    #     dill.dump(label.vocab, f)
    # f1 = open("data/pcap.vector", 'rb')
    # f2 = open("data/label.vector", 'rb')
    # pcap = dill.load(f1)
    # pcap_label = dill.load(f2)
    # text.vocab= pcap
    # label.vocab = pcap_label

    # text.vocab.vectors=temp
    # label.build_vocab(train)
    # print(text.vocab.itos[12])
    # print(text.vocab.itos[340])
    # print(text.vocab.itos[1])
    # print(text.vocab.itos[11])
    print(label.vocab.stoi)
    # for i in label.vocab:
    #     print(i)
    # args.embedding_dim = text.vocab.vectors.size()[-1]
    # args.embedding_dim = 128
    # args.vectors = text.vocab.vectors
    #
    # else:
    #     text.build_vocab(train, val)
    #     label.build_vocab(train, val)
    #     with open("data/pcap.vector", 'wb')as f:
    #         dill.dump(text.vocab, f)
    #     with open("data/label.vector", 'wb')as f:
    #         dill.dump(label.vocab, f)

    # print(text.vocab.vectors)
    train_iter = data.Iterator(
            train,
            sort_key=lambda x: len(x.text),
            # batch_size=len(train), # 训练集设置batch_size,验证集整个集合用于测试
            batch_size=256,  # 训练集设置batch_size,验证集整个集合用于测试
            device=torch.device('cuda')
    )
    valid_iter = data.Iterator(
        valid,
        sort_key=lambda x: len(x.text),
        # batch_size=len(valid)/2, # 训练集设置batch_size,验证集整个集合用于测试
        batch_size=256,  # 训练集设置batch_size,验证集整个集合用于测试
        device=torch.device('cuda')
    )
    # test_iter = data.Iterator(
    #     test,
    #     sort_key=lambda x: len(x.text),
    #     # batch_size=len(test),  # 训练集设置batch_size,验证集整个集合用于测试
    #     batch_size=len(test)/2,  # 训练集设置batch_size,验证集整个集合用于测试
    #     device=torch.device('cuda')
    # )
    print('建立词表后：')
    print('train batch number:')
    print(len(train_iter))
    j=0
    for batch in train_iter:
        j=j+1
        if j%100==0:
            print(j,'输出text嵌入结果：')
            print(batch.text[13])
            print(len(batch.text[0]))
            print(batch.text.shape)
    args.vocab_size = len(text.vocab)
    args.label_num = len(label.vocab)
    print(len(text.vocab))
    print(len(label.vocab))
    return train_iter,valid_iter