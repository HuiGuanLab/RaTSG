import os
import codecs
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from util.data_util import load_json, load_lines, load_pickle, save_pickle, time_to_index

PAD, UNK = "<PAD>", "<UNK>"


class Charades_RFProcessor:#train has answer teat and val has no answer set
    def __init__(self):
        super(Charades_RFProcessor, self).__init__()
        self.idx_counter = 0

    def reset_idx_counter(self):
        self.idx_counter = 0
    
    def process_train_data(self, data, charades, scope):
        results = []
        for line in tqdm(data, total=len(data), desc='process charades-sta {}'.format(scope)):
            line = line.lstrip().rstrip()
            if len(line) == 0:
                continue
            video_info, sentence = line.split('##')
            vid, start_time, end_time = video_info.split(' ')
            duration = float(charades[vid]['duration'])
            start_time = max(0.0, float(start_time))
            end_time = min(float(end_time), duration)
            words = word_tokenize(sentence.strip().lower(), language="english")
            record = {'sample_id': scope+'-%05d'%self.idx_counter, 'vid': str(vid), 's_time': start_time, 'e_time': end_time,
                      'duration': duration, 'words': words,'noanswer':False,'sentence_id' : None}
            results.append(record)
            self.idx_counter += 1
        return results
    
    def process_test_data(self, data, scope):
        results = []
        for vid, data_item in tqdm(data.items(), total=len(data), desc='process new-charades-sta {}'.format(scope)):
            duration = float(data_item['duration'])
            for idx, item in enumerate(data_item["sts"]):
                timestamp = item["timestamp"]
                sentence = item["sentence"]
                noanswer = item["no_answer"]
                sentence_id = item["id"]

                start_time = max(0.0, float(timestamp[0])) if not noanswer else None
                end_time = min(float(timestamp[1]), duration) if not noanswer else None
                words = word_tokenize(sentence.strip().lower(), language="english")
                record = {'sample_id': scope+'-%05d'%self.idx_counter, 'vid': str(vid), 's_time': start_time, 'e_time': end_time,
                          'duration': duration, 'words': words, 'noanswer':noanswer,'sentence_id' : str(sentence_id)}
                results.append(record)
                self.idx_counter += 1
        return results
    
    def convert(self, data_dir, val_name = 'charades_sta_val2.0.json', test_name = 'charades_sta_test2.0.json'):
        if not os.path.exists(data_dir):
            raise ValueError('data dir {} does not exist'.format(data_dir))
        
         # load raw data
        charades = load_json(os.path.join(data_dir, 'charades.json'))
        train_data = load_lines(os.path.join(data_dir, 'charades_sta_train.txt'))
        val_data = load_json(os.path.join(data_dir, val_name))
        test_data = load_json(os.path.join(data_dir, test_name))


        # process data
        self.reset_idx_counter()
        train_set = self.process_train_data(train_data, charades, scope='train')

        self.reset_idx_counter()
        val_set = self.process_test_data(val_data, scope='val')

        self.reset_idx_counter()
        test_set = self.process_test_data(test_data, scope='test')

        return train_set, val_set, test_set  # train/val/test



class ActivityNet_RFProcessor:#train has answer teat and val has no answer set
    def __init__(self):
        super(ActivityNet_RFProcessor, self).__init__()
        self.idx_counter = 0

    def reset_idx_counter(self):
        self.idx_counter = 0
    
    def process_test_data(self, data, scope):
        results = []
        for vid, data_item in tqdm(data.items(), total=len(data), desc='process new-activitynet {}'.format(scope)):
            duration = float(data_item['duration'])
            for idx, item in enumerate(data_item["sts"]):
                timestamp = item["timestamp"]
                sentence = item["sentence"]
                noanswer = item["no_answer"]
                sentence_id = item["id"]
                start_time = max(0.0, float(timestamp[0])) if not noanswer else None
                end_time = min(float(timestamp[1]), duration) if not noanswer else None
                words = word_tokenize(sentence.strip().lower(), language="english")
                record = {'sample_id': scope+'-%05d'%self.idx_counter, 'vid': str(vid), 's_time': start_time, 'e_time': end_time,
                          'duration': duration, 'words': words, 'noanswer':noanswer,'sentence_id' : str(sentence_id)}
                results.append(record)
                self.idx_counter += 1
        return results
    
    def process_train_data(self, data, scope):
        results = []
        for vid, data_item in tqdm(data.items(), total=len(data), desc='process activitynet {}'.format(scope)):
            duration = float(data_item['duration'])
            for timestamp, sentence in zip(data_item["timestamps"], data_item["sentences"]):
                start_time = max(0.0, float(timestamp[0]))
                end_time = min(float(timestamp[1]), duration)
                words = word_tokenize(sentence.strip().lower(), language="english")
                record = {'sample_id': scope+'-%05d'%self.idx_counter, 'vid': str(vid), 's_time': start_time, 'e_time': end_time,
                          'duration': duration, 'words': words,'noanswer':False,'sentence_id' : None}
                results.append(record)
                self.idx_counter += 1
        return results
    def convert(self, data_dir, val_name ='activitynet_val2.0.json', test_name ='activitynet_test2.0.json'):
        if not os.path.exists(data_dir):
            raise ValueError('data dir {} does not exist'.format(data_dir))
        
        # load raw data
        train_data = load_json(os.path.join(data_dir, 'train.json'))
        val_data = load_json(os.path.join(data_dir, val_name))
        test_data = load_json(os.path.join(data_dir, test_name))

        # process data
        self.reset_idx_counter()
        train_set = self.process_train_data(train_data, scope='train')

        self.reset_idx_counter()
        val_set = self.process_test_data(val_data, scope='val')

        self.reset_idx_counter()
        test_set = self.process_test_data(test_data, scope='test')

        return train_set, val_set, test_set  # train/val/test


def load_glove(glove_path):
    vocab = list()
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, desc="load glove vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            vocab.append(word)
    return set(vocab)


def filter_glove_embedding(word_dict, glove_path):
    '''
    glove-> small glove
    '''
    vectors = np.zeros(shape=[len(word_dict), 300], dtype=np.float32)
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, desc="load glove embeddings"):#
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]#
            if word in word_dict: #
                vector = [float(x) for x in line[1:]]
                word_index = word_dict[word]
                vectors[word_index] = np.asarray(vector)
    print(vectors.shape)
    return np.asarray(vectors)


def vocab_emb_gen(datasets, emb_path):
    '''
    datasets: [train, val, test]
    emb_path: glove_path
    '''
    # generate word dict and vectors
    emb_vocab = load_glove(emb_path)
    word_counter, char_counter = Counter(), Counter()#
    for data in datasets:
        for record in data:
            for word in record['words']:
                word_counter[word] += 1
                for char in list(word):
                    char_counter[char] += 1
    word_vocab = list()#

    for word, _ in word_counter.most_common():
        if word in emb_vocab:
            word_vocab.append(word)
    tmp_word_dict = dict([(word, index) for index, word in enumerate(word_vocab)])
    # print(len(tmp_word_dict))
    vectors = filter_glove_embedding(tmp_word_dict, emb_path)# [word_nums, 300]
    # print(len(vectors))
    word_vocab = [PAD, UNK] + word_vocab
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])#key word// value index in small glove
    # generate character dict
    char_vocab = [PAD, UNK] + [char for char, _ in char_counter.most_common()]#key char //value rank
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    return word_dict, char_dict, vectors


def dataset_gen(data, vfeat_lens, word_dict, char_dict,  max_desc_len, scope):
    '''
    vfeat_lens{vid:length}
    word_dict{word:index}
    char_dict{char:index}
    max_desc_len:max query lengths and max, Usually, this parameter is ineffective, cause the length of text is short
    '''
    # noansdataset = list()
    # hasansdataset = list()
    dataset = list()
    for record in tqdm(data, total=len(data), desc='process {} data'.format(scope)):
        vid = record['vid']
        if vid not in vfeat_lens:
            print(vid)
            continue
        s_ind, e_ind, _ = time_to_index(record['s_time'], record['e_time'], vfeat_lens[vid], record['duration'])#The moments are mapped to the 128 scale
        word_ids, char_ids = [], []
        for word in record['words'][0:max_desc_len]:#The words of the sentence are also taken only the first max desc len
            word_id = word_dict[word] if word in word_dict else word_dict[UNK]
            char_id = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            word_ids.append(word_id)#Put the word index in
            char_ids.append(char_id)#Put the char index in there
        result = {'sample_id': record['sample_id'], 'vid': record['vid'], 's_time': record['s_time'],
                  'e_time': record['e_time'], 'duration': record['duration'], 'words': record['words'],
                  's_ind': int(s_ind), 'e_ind': int(e_ind), 'v_len': vfeat_lens[vid], 'w_ids': word_ids,
                  'c_ids': char_ids, 'noanswer': record['noanswer'],"sentence_id":record['sentence_id']}
        dataset.append(result)
    return dataset


def gen_or_load_dataset(configs):
    if not os.path.exists(configs.save_dir):
        os.makedirs(configs.save_dir)

    dataset_name = configs.task.split('_')[0]
    data_dir = os.path.join('data', 'dataset', configs.task)
    feature_dir = os.path.join('data', 'features', dataset_name, configs.fv)
    if configs.suffix is None:
        save_path = os.path.join(configs.save_dir, '_'.join([configs.task, configs.fv, str(configs.max_pos_len)]) +
                                 '.pkl')
    else:
        save_path = os.path.join(configs.save_dir, '_'.join([configs.task, configs.fv, str(configs.max_pos_len),
                                                             configs.suffix]) + '.pkl')
    if os.path.exists(save_path):
        print(save_path)
        dataset = load_pickle(save_path)
        return dataset
    
    feat_len_path = os.path.join(feature_dir, 'feature_shapes.json')
    print(feat_len_path)
    emb_path = os.path.join('data', 'features', 'glove.840B.300d.txt')
    # load video feature length
    vfeat_lens = load_json(feat_len_path)
    for vid, vfeat_len in vfeat_lens.items():
        vfeat_lens[vid] = min(configs.max_pos_len, vfeat_len)#The video length is max pos len at most
    # load data
    if configs.task == 'charades_RF':
        processor = Charades_RFProcessor()
    elif configs.task == 'activitynet_RF':
        processor = ActivityNet_RFProcessor()
    else:
        raise ValueError('Unknown task {}!!!'.format(configs.task))
    
    train_data, val_data, test_data = processor.convert(data_dir,configs.val_name,configs.test_name)
    # generate dataset
    data_list = [train_data, val_data, test_data]
    word_dict, char_dict, vectors = vocab_emb_gen(data_list, emb_path)
    
    train_set = dataset_gen(train_data, vfeat_lens, word_dict, char_dict, configs.max_desc_len, 'train')
    val_set =  dataset_gen(val_data, vfeat_lens, word_dict, char_dict,
                                                        configs.max_desc_len,'val')
    test_set = dataset_gen(test_data, vfeat_lens, word_dict, char_dict,  configs.max_desc_len, 'test')

    
    # save dataset
    dataset = {'train_set': train_set, 'val_set': val_set, 'test_set': test_set, 'word_dict': word_dict,
               'char_dict': char_dict, 'word_vector': vectors, 'n_train': len(train_set), 'n_val': len(val_set),
               'n_test': len(test_set), 'n_words': len(word_dict), 'n_chars': len(char_dict)}
    save_pickle(dataset, save_path)
    return dataset



