import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from model.model import RaTSG, build_optimizer_and_scheduler
from util.data_util import load_video_features, save_json, load_json
from util.data_gen_light import gen_or_load_dataset
from util.data_loader_light_t7 import get_train_loader, get_test_loader
from util.runner_utils_light_t7 import set_th_config, convert_length_to_mask, filter_checkpoints, get_last_checkpoint, model_eval
from torch.utils.tensorboard import SummaryWriter
import random

parser = argparse.ArgumentParser()
# data parameters
parser.add_argument('--save_dir', type=str, default='datasets_bert_t7', help='path to save processed dataset')
parser.add_argument('--task', type=str, default='charades_RF', help='[charades_RF,activityney_RF]  task')
parser.add_argument('--fv', type=str, default='i3d', help='[i3d] for visual features')
parser.add_argument('--max_pos_len', type=int, default=128, help='maximal position sequence length allowed')
parser.add_argument('--max_desc_len', type=int, default=50, help='maximal sentencese length allowed')

# model parameters
parser.add_argument("--word_size", type=int, default=None, help="number of words")
parser.add_argument("--char_size", type=int, default=None, help="number of characters")
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--video_feature_dim", type=int, default=1024, help="video feature input dimension")
parser.add_argument("--char_dim", type=int, default=50, help="character dimension, set to 100 for activitynet")
parser.add_argument("--dim", type=int, default=128, help="hidden size")
parser.add_argument("--intermediate_size", type=int, default=256, help="intermediate_size")
parser.add_argument("--n_heads", type=int, default=8, help="num_attention_heads")
parser.add_argument("--input_drop_rate", type=float, default=0.1, help="dropout rate")
parser.add_argument("--drop_rate", type=float, default=0.2, help="dropout rate")
parser.add_argument("--num_hidden_layers", type=int, default=4, help="num_hidden_layers")
parser.add_argument('--hidden_act', type=str, default='gelu', help='')
parser.add_argument("--initializer_range", type=float, default=0.02, help="initializer_range")

# training/evaluation parameters
parser.add_argument("--gama", type=float, default=10.0, help="gama for bce loss(relation loss)")
parser.add_argument("--beta", type=float, default=10.0, help="beta for bf-loss(fore loss)")
parser.add_argument("--threshold", type=float, default=0.6, help="threshold for case discrimination")
parser.add_argument("--gpu_idx", type=str, default="0", help="GPU index")
parser.add_argument("--seed", type=int, default=12345, help="random seed")
parser.add_argument("--mode", type=str, default="train", help="[train | test]")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--num_train_steps", type=int, default=None, help="number of training steps")
parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--low_lr", type=float, default=0.01, help="initial learning rate")
parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
parser.add_argument("--period", type=int, default=100, help="training loss print period")
parser.add_argument('--model_dir', type=str, default='ckpt_t7', help='path to save trained model weights')
parser.add_argument('--model_name', type=str, default='vslnet', help='model name')
parser.add_argument('--suffix', type=str, default=None, help='set to the last `_xxx` in ckpt repo to eval results')
parser.add_argument('--test_name', type=str, default='newcharades_sta_test.json', help='test_file name')
parser.add_argument('--val_name', type=str, default='newcharades_sta_test.json', help='val_file name')
parser.add_argument('--pretrain', type=str, default=None, help='test_file name')
parser.add_argument('--train_nums_rate', type=float, default=1., help='train datasets rate')

configs = parser.parse_args()

#solve the cpu problem
torch.set_num_threads(1)
# set tensorflow configs
set_th_config(configs.seed)

# prepare or load dataset
dataset = gen_or_load_dataset(configs)
configs.char_size = dataset['n_chars']
configs.word_size = dataset['n_words']

dataset_name = configs.task.split("_")[0]
visual_features = load_video_features(os.path.join('data', 'features', dataset_name, configs.fv), configs.max_pos_len)#Load video features, if the length is greater than 128 to reintegrate the clip
train_dataset = dataset['train_set'] if configs.train_nums_rate == 1. else random.sample(dataset['train_set'], int(len(dataset['train_set'])*configs.train_nums_rate))

train_loader = get_train_loader(dataset=train_dataset, video_features=visual_features, configs=configs)
val_loader =  get_test_loader(dataset=dataset['val_set'], video_features=visual_features, configs=configs)
test_loader = get_test_loader(dataset=dataset['test_set'], video_features=visual_features, configs=configs)


configs.num_train_steps = len(train_loader) * configs.epochs
num_train_batches = len(train_loader)
print(num_train_batches)
num_val_batches = 0 if val_loader is None else len(val_loader)
num_test_batches = len(test_loader)

# Device configuration
cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

# create model dir
home_dir = os.path.join(configs.model_dir, '_'.join([configs.model_name, configs.task, configs.fv, str(configs.max_pos_len)]))
# set tensor_board
tbwriter = SummaryWriter('tensorboard/'+'_'.join([configs.model_name, configs.task, configs.fv, str(configs.max_pos_len)]))

if configs.suffix is not None:
    home_dir = home_dir + '_' + configs.suffix
model_dir = os.path.join(home_dir, "model")

# train and test
if configs.mode.lower() == 'train':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    eval_period = num_train_batches
    print(eval_period)
    save_json(vars(configs), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)
    
    # build model
    model = RaTSG(configs=configs, word_vectors=dataset['word_vector']).to(device)
    if configs.pretrain is not None:
        filename = get_last_checkpoint(configs.pretrain, suffix='t7')
        model.load_state_dict(torch.load(filename),strict= False)

    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
    # start training
    best_acc = .0
    best_mi_ANS = .0
    best_score = .0
    score_writer = open(os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8")
    print('start training...', flush=True)
    global_step = 0
    for epoch in range(configs.epochs):
        model.train()
        for data in tqdm(train_loader, total=num_train_batches, desc='Epoch %3d / %3d' % (epoch + 1, configs.epochs)):
            global_step += 1
            records, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels,  f_labels, h_labels = data
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            s_labels, e_labels,  f_labels, h_labels = s_labels.to(device), e_labels.to(device),f_labels.to(device),h_labels.to(device)
            # generate mask
            query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)# Anything other than 0 is true; otherwise, it is false because index 0 points to [PAD].
            video_mask = convert_length_to_mask(vfeat_lens,max_len=configs.max_pos_len).to(device)

            ce_loss, bce_loss, bf_loss = model(word_ids, char_ids, vfeats, video_mask, query_mask, s_labels, e_labels, f_labels, h_labels, is_train=True)
            tbwriter.add_scalar('Train_l/boundary_loss', ce_loss, global_step)
            tbwriter.add_scalar('Train_l/relation_loss', bce_loss, global_step)
            tbwriter.add_scalar('Train_l/fore_loss', bf_loss, global_step)
            total_loss = ce_loss + configs.gama* bce_loss + configs.beta* bf_loss

            tbwriter.add_scalar('Train_l/loss Total', total_loss, global_step)
            # compute and apply gradients
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), configs.clip_norm)  # clip gradient
            optimizer.step()
            scheduler.step()
            # evaluate
            if global_step % eval_period == 0 or global_step % num_train_batches == 0:
                model.eval()
                r1i3_ans, r1i5_ans, r1i7_ans, mi_ans, mi_accs, score_str = model_eval(model=model, data_loader=val_loader, device=device,
                                                                        max_pos_len = configs.max_pos_len, threshold = configs.threshold,
                                                                        mode='test', epoch=epoch + 1, global_step=global_step)
                
                print('\nEpoch: %2d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f | acc: %.2f' % (epoch + 1, global_step, r1i3_ans, r1i5_ans, r1i7_ans, mi_ans, mi_accs), flush=True)
                
                tbwriter.add_scalar('Test/miou', mi_ans, global_step)
                tbwriter.add_scalar('Test/acc', mi_accs, global_step)
                score_writer.write(score_str)
                score_writer.flush()

                if  mi_ans >= best_score:
                    best_score = mi_ans
                    torch.save(model.state_dict(), os.path.join(model_dir, '{}_{}.t7'.format(configs.model_name, global_step)))
                    # only keep the top-3 model checkpoints
                    filter_checkpoints(model_dir, suffix='t7', max_to_keep=3)
                
                model.train()
    score_writer.close()
    tbwriter.close()

elif configs.mode.lower() == 'test':
    print(model_dir)
    if not os.path.exists(model_dir):
        raise ValueError('No pre-trained weights exist')
    # load previous configs
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    # build model
    model = RaTSG(configs=configs, word_vectors=dataset['word_vector']).to(device)
    # get last checkpoint file
    filename = get_last_checkpoint(model_dir, suffix='t7')
    model.load_state_dict(torch.load(filename))

    model.eval()
    r1i3_ans, r1i5_ans, r1i7_ans, mi_ans, mi_accs, _ = model_eval(model=model, data_loader=test_loader, device=device, max_pos_len = configs.max_pos_len,  threshold = configs.threshold, mode='test', epoch=None, global_step=None )
   
    print("\n" + "\x1b[1;31m" + "Rank@1, IoU_ans=0.3:\t{:.2f}".format(r1i3_ans) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "Rank@1, IoU_ans=0.5:\t{:.2f}".format(r1i5_ans) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "Rank@1, IoU_ans=0.7:\t{:.2f}".format(r1i7_ans) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "{}:\t{:.2f}".format("mean IoU_ans".ljust(15), mi_ans) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "acc:\t{:.2f}".format(mi_accs) + "\x1b[0m", flush=True)
    
