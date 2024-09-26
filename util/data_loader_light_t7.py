import numpy as np
import torch
import torch.utils.data
from util.data_util import pad_seq, pad_char_seq, pad_video_seq

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_features):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.video_features = video_features
       
    def __getitem__(self, index):
        record = self.dataset[index]
        v_id = record['vid']
        video_feature = self.video_features[record['vid']]
        s_ind, e_ind = record['s_ind'], record['e_ind']
        word_ids, char_ids = record['w_ids'], record['c_ids']
        noanswer = record["noanswer"]
        return record, video_feature, word_ids, char_ids, noanswer, s_ind, e_ind, v_id

    def __len__(self):
        return len(self.dataset)
    
def train_collate_fn(data, max_pos_length, max_desc_length):
    records, video_features, word_ids, char_ids, _, s_inds, e_inds, vid= zip(*data)
    # process word ids
    word_ids, _ = pad_seq(word_ids,max_length=max_desc_length)
    word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids,max_length=max_desc_length)
    char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features, max_length=max_pos_length)

    vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # process labels
    batch_size = vfeat_lens.shape[0]
    # gennerate_vid_mask
    vid = np.asarray(vid)
    

    h_labels = np.zeros(shape=[batch_size, max_pos_length], dtype=np.float32) 
    extend = 0.
    for idx in range(batch_size):
        st, et = s_inds[idx]-1, e_inds[idx]-1
        cur_max_len = vfeat_lens[idx]
        extend_len = round(extend * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, cur_max_len - 1)
            h_labels[idx][st_:(et_ + 1)] = 1.
        else:
            h_labels[idx][st:(et + 1)] = 1.
    f_labels = torch.cat([torch.zeros(batch_size),torch.ones(batch_size)],dim=0) #[B+B] hasans=0, noans=1 [00000000000000000001111111111111111]
    s_labels = torch.cat([torch.tensor(np.asarray(s_inds)),torch.zeros(batch_size)],dim=0)#[2,4,0,8...]
    e_labels = torch.cat([torch.tensor(np.asarray(e_inds)),torch.zeros(batch_size)],dim=0)#[2,4,0,8...]

    # convert to torch tensor
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)

    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    f_labels = torch.tensor(f_labels, dtype=torch.float32)
    h_labels = torch.tensor(h_labels, dtype=torch.float32)
    return records, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, f_labels, h_labels


def test_collate_fn(data, max_pos_length, max_desc_length):
    records, video_features, word_ids, char_ids, *_= zip(*data)
    # process word ids
    word_ids, _ = pad_seq(word_ids,max_length=max_desc_length)
    word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids,max_length=max_desc_length)
    char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features,max_length=max_pos_length)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)

    return records, vfeats, vfeat_lens, word_ids, char_ids 


def get_train_loader(dataset, video_features, configs):
    train_set = Dataset(dataset=dataset, video_features=video_features)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=True,
                                                collate_fn= lambda data: train_collate_fn(data, max_pos_length = configs.max_pos_len, max_desc_length= configs.max_desc_len))
    
    return train_loader

def get_test_loader(dataset, video_features, configs):
    test_set = Dataset(dataset=dataset, video_features=video_features)
   
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=configs.batch_size, shuffle=False,
                                              collate_fn=lambda data: test_collate_fn(data, max_pos_length = configs.max_pos_len, max_desc_length= configs.max_desc_len))
    return test_loader



