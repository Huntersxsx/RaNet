""" Dataset loader for the TACoS dataset """
import os
import json

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from . import average_to_fixed_length
from core.eval import iou
from core.config import config
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import math

class TACoS(data.Dataset):

    # GloVe
    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    # BERT
    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bert_model = BertModel.from_pretrained('bert-base-uncased')
    # print("Loaded BERT...")
    # bert_model.eval()



    def __init__(self, split):
        super(TACoS, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        self.temporal_scale = 128
        self.temporal_gap = 1. / self.temporal_scale
        self.boundary_ratio = 0.1
        self.anchor_xmin = [self.temporal_gap * i for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * i for i in range(1, self.temporal_scale + 1)]

        # val_1.json is renamed as val.json, val_2.json is renamed as test.json
        with open(os.path.join(self.data_dir, '{}.json'.format(split)),'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        vid_dict = {}
        for vid, video_anno in annotations.items():
            duration = video_anno['num_frames']/video_anno['fps']
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                vid_list = []
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append(          
                        {
                            'video': vid,
                            'duration': duration,
                            'times':[max(timestamp[0]/video_anno['fps'],0),min(timestamp[1]/video_anno['fps'],duration)],
                            'description':sentence,
                        }
                    )
                    vid_list.append((max(timestamp[0]/video_anno['fps'],0),min(timestamp[1]/video_anno['fps'],duration)))
            vid_dict[vid] = vid_list
        self.annotations = anno_pairs
        self.vid_dict = vid_dict

        # self.num_clips = 128
        # self.expand_ratio = 0.25
        # self.num_sample = 32
        # self.num_sample_perbin = 3
        # self.get_sampling_mask_weight()

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        sentence = self.annotations[index]['description']
        duration = self.annotations[index]['duration']

        # GloVe
        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)
        # BERT
        # word_vectors = self.get_bert_sentence_tokens(sentence)

        txt_mask = torch.ones(word_vectors.shape[0], 1)

        visual_input, visual_mask = self.get_video_features(video_id)

        # visual_input = sample_to_fixed_length(visual_input, random_sampling=config.DATASET.RANDOM_SAMPLING)
        visual_input = average_to_fixed_length(visual_input)
        num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
        s_times = torch.arange(0,num_clips).float()*duration/num_clips
        e_times = torch.arange(1,num_clips+1).float()*duration/num_clips
        # video的每一个clip（128*128）与ground truth的iou
        overlaps = iou(torch.stack([s_times[:,None].expand(-1,num_clips),
                                    e_times[None,:].expand(num_clips,-1)],dim=2).view(-1,2).tolist(),
                       torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips,num_clips)

        match_score_action, match_score_start, match_score_end = self.get_groundtruth(video_id, duration)

        item = {
            'visual_input': visual_input,
            'vis_mask': visual_mask,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'duration': duration,
            'txt_mask': txt_mask,
            'map_gt': torch.from_numpy(overlaps),
            'gt_action': match_score_action,
            'gt_start': match_score_start,
            'gt_end': match_score_end,
            # 'sample_mask_weight': self.sample_mask_weight,
            # 'description': sentence,
            # 'gt_s_e': [gt_s_time, gt_e_time],
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        assert config.DATASET.VIS_INPUT_TYPE == 'c3d'
        with h5py.File(os.path.join(self.data_dir, 'tall_c3d_features.hdf5'), 'r') as f:
            features = torch.from_numpy(f[vid][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

    def get_bert_sentence_tokens(self, sentence):
        sentence = "[CLS] " + sentence + " [SEP]"
        tokenized_text = self.bert_tokenizer.tokenize(sentence)
        indexed_token = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        sentence_embeddings = []
        
        tokens_tensor = torch.tensor([indexed_token])
        segments_tensor = torch.tensor([segments_ids])

        # Predict hidden states features for each layer
        with torch.no_grad():
            output = self.bert_model(tokens_tensor, segments_tensor)

        token_embeddings = []

        for token_i in range(len(tokenized_text)):
            hidden_layers = []
            vec = output[0][0][token_i]
            token_embeddings.append(vec)

        token_embeddings = torch.tensor(np.array([x.numpy() for x in token_embeddings]))

        return token_embeddings

    def ioa_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores

    def get_groundtruth(self, video_id, duration):
        gt_bbox = []
        for idx in range(len(self.vid_dict[video_id])):
            tmp_start = max(min(1, self.vid_dict[video_id][idx][0] / duration), 0)
            tmp_end = max(min(1, self.vid_dict[video_id][idx][1] / duration), 0)
            gt_bbox.append([tmp_start, tmp_end])
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]

        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        match_score_action = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_action.append(np.max(
                self.ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_xmins, gt_xmaxs)))  # maximum matching overlap
        match_score_start = []
        # [0.01, 0.02,....,1.0]
        for jdx in range(len(self.anchor_xmin)):
            match_score_start.append(np.max(
                self.ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_end.append(np.max(
                self.ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        #match_score_action = torch.Tensor(match_score_action)
        #match_score_start = torch.Tensor(match_score_start)
        #match_score_end = torch.Tensor(match_score_end)
        return match_score_action, match_score_start, match_score_end
