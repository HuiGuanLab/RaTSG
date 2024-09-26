import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from model.model_components import FusionModule, QueryEncoder,TrainablePositionalEncoding, FeatureEncoder,WeightedPool,DynamicRNN,BCE_loss,Back_forward_ground_loss, VisualProjection, CQAttention

def build_optimizer_and_scheduler(model, configs):
    no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
  
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
     
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                configs.num_train_steps)
    return optimizer, scheduler

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class RaTSG(nn.Module):
    def __init__(self, configs, word_vectors):
        super(RaTSG, self).__init__()
        self.configs = configs

        self.word_char_encoder = QueryEncoder(num_words=configs.word_size, num_chars=configs.char_size, out_dim=configs.dim, 
                                          word_dim=configs.word_dim, char_dim=configs.char_dim, word_vectors=word_vectors, drop_rate=configs.drop_rate)
        self.video_affine = VisualProjection(visual_dim=configs.video_feature_dim, dim=configs.dim, drop_rate=configs.drop_rate)
        # position embedding
        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=configs.max_desc_len,
                                                           hidden_size=configs.dim, dropout=configs.input_drop_rate)
        self.video_pos_embed = TrainablePositionalEncoding(max_position_embeddings=configs.max_pos_len,
                                                           hidden_size=configs.dim, dropout=configs.input_drop_rate)
        # transformer encoder
        self.feature_encoder = FeatureEncoder(dim=configs.dim, num_heads=configs.n_heads, kernel_size=7, num_layers=4, drop_rate=configs.drop_rate)

        #text-video fusion
        self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.fc = nn.Linear(configs.dim, configs.dim)# no using


        #self-attention pooling
        self.sentence_gennerator = WeightedPool(configs.dim)
        
        #signal-genneration
        self.signal_gennerator = nn.Sequential(
            nn.Linear(in_features= configs.dim * 2, out_features=configs.dim),
            nn.ReLU(),
            nn.Dropout(configs.drop_rate),
            nn.Linear(in_features= configs.dim , out_features=configs.dim)
        )
       

        #score-genneration
        self.score_metrics = nn.Linear(configs.dim , 1)

        #binary_mask_gennerator
        self.feature_fusion = FusionModule(configs.dim)
        
         #Predictor-layer
        self.start_lstm = DynamicRNN(dim=configs.dim)
        self.end_lstm = DynamicRNN(dim=configs.dim)
     
        self.start_layer = nn.Sequential(
            nn.Linear(in_features= configs.dim * 2, out_features=configs.dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features= configs.dim, out_features=1, bias=True)
        )
        self.end_layer = nn.Sequential(
            nn.Linear(in_features= configs.dim * 2, out_features=configs.dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features= configs.dim, out_features=1, bias=True)
        )

        #inference and loss
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.bce_loss = BCE_loss(reduction="mean")
        self.bf_loss = Back_forward_ground_loss(reduction='mean')
        self.init_parameters()
        self.use_great_negative = False

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()
        self.apply(init_weights)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask, loc_starts = None, loc_ends = None, f_labels = None, h_labels = None, return_signal = False, is_train=False):
        #encoding text query to [B,Lq,d]
        raw_query_features = self.word_char_encoder(word_ids, char_ids)
        #encoding video to [B,Lv,d]
        raw_video_features = self.video_affine(video_features)
        #position-embedding
        add_pos_query_features = self.query_pos_embed(raw_query_features)
        add_pos_video_features = self.video_pos_embed(raw_video_features)
        #encoding-highlevel
        query_features = self.feature_encoder(add_pos_query_features, q_mask)
        video_features = self.feature_encoder(add_pos_video_features, v_mask)

        #fusion text-video
        pos_features = self.cq_attention(video_features, query_features, v_mask, q_mask)

        #sentence genneration, maybe a mean pooling is enough
        sentence_feature = self.sentence_gennerator(query_features,q_mask)
        
        #background - forground - modual
        pos_sigmoid_scores, pos_softmax_scores = self.feature_fusion(pos_features, sentence_feature,v_mask)

        #clip feature
        pos_clip_feature = torch.sum(pos_features * pos_softmax_scores.unsqueeze(2),dim=1) # B,L,C

        #increase forground
        pos_features = pos_features * pos_sigmoid_scores.unsqueeze(2)
        pos_bf_max_score, _ = torch.max(pos_sigmoid_scores,dim=1)

        #signal genneration
        pos_signal_feature = self.signal_gennerator(torch.cat([sentence_feature, pos_clip_feature],dim=-1))
        
        # pos_signal_feature = pos_clip_feature
        pos_score = self.score_metrics(pos_signal_feature).squeeze(1) # no sigmoid
        pos_score_max = 1. - pos_bf_max_score #with sigmoid


        #concat the signal at first token, and throughout Transformer encoders
        B = v_mask.shape[0]
        
        pos_sp_token = pos_signal_feature.clone()
        pos_v2t_features_with_signal = torch.cat([pos_sp_token.unsqueeze(1), pos_features],dim=1)

        extend_vmask = torch.cat([torch.ones((B,1)).to(v_mask.device), v_mask],dim=1)
       
        #predict
        start_features = self.start_lstm(pos_v2t_features_with_signal, extend_vmask)  # (batch_size, seq_len, dim) 
        end_features = self.end_lstm(start_features, extend_vmask) #
        pos_start_logits_nomasked = self.start_layer(torch.cat([start_features, pos_v2t_features_with_signal], dim=2)) 
        pos_start_logits = mask_logits(pos_start_logits_nomasked.squeeze(2), mask=extend_vmask)
        pos_end_logits_nomasked = self.end_layer(torch.cat([end_features, pos_v2t_features_with_signal], dim=2))
        pos_end_logits = mask_logits(pos_end_logits_nomasked.squeeze(2), mask=extend_vmask)

        #ce-loss/bce-loss/bf-loss
        # neg sample in train stage
        if is_train:
            neg_video_features = torch.cat([video_features[1:], video_features[0:1]], dim=0)
            neg_v_mask = torch.cat([v_mask[1:], v_mask[0:1]], dim=0)

            #fusion text-video
            neg_features = self.cq_attention(neg_video_features, query_features, neg_v_mask, q_mask)
           
            #background - forground - modual
            neg_sigmoid_scores, neg_softmax_scores = self.feature_fusion(neg_features, sentence_feature,neg_v_mask)

            neg_clip_feature = torch.sum(neg_features * neg_softmax_scores.unsqueeze(2),dim=1) # B,L,C

            neg_features = neg_features * neg_sigmoid_scores.unsqueeze(2)

            neg_bf_max_score, _ = torch.max(neg_sigmoid_scores,dim=1)

            neg_signal_feature = self.signal_gennerator(torch.cat([sentence_feature,neg_clip_feature],dim=-1))

            neg_score = self.score_metrics(neg_signal_feature).squeeze(1) 

            neg_score_max = 1. - neg_bf_max_score

            #concat the signal at first token, and throughout Transformer encoders
            neg_sp_token = neg_signal_feature.clone()
            neg_v2t_features_with_signal = torch.cat([neg_sp_token.unsqueeze(1), neg_features],dim=1)
            extend_neg_vmask = torch.cat([torch.ones((B,1)).to(neg_v_mask.device), neg_v_mask],dim=1)

            #predict
            neg_start_features = self.start_lstm(neg_v2t_features_with_signal, extend_neg_vmask)  # (batch_size, seq_len, dim) 
            neg_end_features = self.end_lstm(neg_start_features, extend_neg_vmask) 
            neg_start_logits_nomasked = self.start_layer(torch.cat([neg_start_features, neg_v2t_features_with_signal], dim=2))
            neg_start_logits = mask_logits(neg_start_logits_nomasked.squeeze(2), mask=extend_neg_vmask)
            neg_end_logits_nomasked = self.end_layer(torch.cat([neg_end_features, neg_v2t_features_with_signal], dim=2))
            neg_end_logits = mask_logits(neg_end_logits_nomasked.squeeze(2), mask=extend_neg_vmask)

            #compute ce-loss
            start_logits = torch.cat([pos_start_logits, neg_start_logits],dim=0)
            end_logits = torch.cat([pos_end_logits, neg_end_logits],dim=0)
            start_loss = self.ce_loss(start_logits, loc_starts)
            end_loss = self.ce_loss(end_logits, loc_ends)
            ce_loss = start_loss + end_loss

            #compute bce-loss
            score_after = torch.cat([pos_score,neg_score],dim=0)
            score_after_max = torch.cat([pos_score_max,neg_score_max],dim=0)
            sa_bce_loss =  self.bce_loss(score_after, f_labels, weight=1.0)
            bce_loss = sa_bce_loss

            #compute back_forwad_loss
            h_scores = torch.cat([pos_sigmoid_scores,neg_sigmoid_scores],dim=0)
            neg_h_labels = torch.zeros_like(h_labels,device=h_labels.device)
            h_labels = torch.cat([h_labels,neg_h_labels],dim=0)
            all_v_mask = torch.cat([v_mask,neg_v_mask],dim=0)
            bf_loss = self.bf_loss(h_scores,h_labels,all_v_mask,weight=2.0)

            return ce_loss, bce_loss, bf_loss

       
        score_after = (self.sigmoid(pos_score) + pos_score_max)/2


        if return_signal:
            return pos_signal_feature

        
        return self.softmax(pos_start_logits), self.softmax(pos_end_logits), score_after
    

    @staticmethod
    def extract_start_end_index_with_case_score(pos_start_logits, pos_end_logits, score_after, threshold):
        # Time points are considered discrete
        outer = pos_start_logits.unsqueeze(dim=2) + pos_end_logits.unsqueeze(dim=1)
        outer = torch.triu(outer,)

        outer_small = outer[:,1:,1:]
        _, start_index = torch.max(torch.max(outer_small, dim=2)[0], dim=1)  # (batch_size, )
        _, end_index = torch.max(torch.max(outer_small, dim=1)[0], dim=1)  # (batch_size, )
       
        mask = (score_after < threshold)
      
        start_index = (start_index + 1) * mask
        end_index = (end_index + 1) * mask

        return start_index, end_index

   