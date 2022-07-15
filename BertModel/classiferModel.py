from main import *

if __name__ == "__main__":

    model_name = "BertCNN"
    label_list = ['ppstream',
    'qqmusic',
    'sbs_gorealra',
    'scribd',
    'tunein_radio',
    'youku',
    'vsee',
    'icq',
    'imo',
    'whatsapp',
    'alibaba',
    'any',
    'espn_star_sports',
    'firefox',
    'flipboard',
    'google',
    'hupu',
    'speedtest',
    'yahoo',
    'yandex',
    'teamviewer',
    'fubar',
    'pinterest',
    'travelzoo',
    'nbc',
    'startv',
    'moneycontrol',
    'mlb',
    'nhl_gamecenter',
    'chrome',
    'dictionary',
    'yahoo_messenger',
    'itunes',
    'kik',
    'afreecatv',
    'skype',
    'usa_today',
    'mega',
    'tagged',
    'camfrog',
    'amazon_instant_video']
    en_list = ['aim',
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
    data_dir = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/app"
    entropy_dir = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/en"
    output_dir = ".sst_output/"
    cache_dir = ".sst_cache/"
    log_dir = ".sst_log/"

    # bert-base
    # bert_vocab_file = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/pytorch_Bert/bert-base-uncased-vocab.txt"
    # bert_model_dir = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/pytorch_Bert/bert-base-uncased"

    # # bert-large
    bert_vocab_file = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/pytorch_Bert/bert-large-uncased-vocab.txt"
    bert_model_dir = "/home/user1/PangBo/GNNTrafficClassification/Dataset/BertDataset/pytorch_Bert/bert-large-uncased"

    if model_name == "BertOrigin":
        from BertOrigin import args

    elif model_name == "BertCNN":
        from BertCNN import args

    elif model_name == 'BertLSTM':
        from BertLSTM import args

    elif model_name == "BertATT":
        from BertATT import args

    elif model_name == "BertRCNN":
        from BertRCNN import args

    elif model_name == "BertCNNPlus":
        from BertCNNPlus import args

    elif model_name == "BertDPCNN":
        from BertDPCNN import args

    config = args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir)

    main(config, config.save_name, label_list)