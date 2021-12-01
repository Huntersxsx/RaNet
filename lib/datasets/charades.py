""" Dataset loader for the Charades-STA dataset """
import os
import csv

import h5py
import numpy as np
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

class Charades(data.Dataset):

    
    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)
    
    def __init__(self, split):
        super(Charades, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        self.durations = {}
        with open(os.path.join(self.data_dir, 'Charades_v1_{}.csv'.format(split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length'])

        anno_file = open(os.path.join(self.data_dir, "charades_sta_{}.txt".format(self.split)),'r')
        annotations = []
        for line in anno_file:
            anno, sent = line.split("##")
            sent = sent.split('.\n')[0]
            vid, s_time, e_time = anno.split(" ")
            s_time = float(s_time)
            e_time = min(float(e_time), self.durations[vid])
            if s_time < e_time:
                annotations.append({'video':vid, 'times':[s_time, e_time], 'description': sent, 'duration': self.durations[vid]})
        anno_file.close()
        self.annotations = annotations

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        # sentence = self.annotations[index]['description']
        description = self.annotations[index]['description']
        duration = self.durations[video_id]

        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in description.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        # word_vectors = self.get_bert_sentence_tokens(description)
        txt_mask = torch.ones(word_vectors.shape[0], 1)

        if self.vis_input_type == 'I3D/finetune' or self.vis_input_type == 'I3D/raw':
            visual_input, visual_mask = self.get_i3d_video_features(video_id)
        elif self.vis_input_type =='C3D/C3D-features':
            visual_input, visual_mask = self.get_c3d_video_features(video_id)
        else:
            visual_input, visual_mask = self.get_vgg_video_features(video_id)


        # Time scaled to fixed size
        # visual_input = sample_to_fixed_length(visual_input, random_sampling=True)
        # visual_input = interpolate_to_fixed_length(visual_input)
        visual_input = average_to_fixed_length(visual_input)
        num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
        s_times = torch.arange(0,num_clips).float()*duration/num_clips
        e_times = torch.arange(1,num_clips+1).float()*duration/num_clips
        overlaps = iou(torch.stack([s_times[:,None].expand(-1,num_clips),
                                    e_times[None,:].expand(num_clips,-1)],dim=2).view(-1,2).tolist(),
                       torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips,num_clips)

        gt_s_idx = np.argmax(overlaps)//num_clips
        gt_e_idx = np.argmax(overlaps)%num_clips

        item = {
            'visual_input': visual_input,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'txt_mask': txt_mask,
            'map_gt': torch.from_numpy(overlaps),
            'reg_gt': torch.tensor([gt_s_idx, gt_e_idx]),
            'duration': duration
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_vgg_video_features(self, vid):
        hdf5_file = h5py.File(os.path.join(self.data_dir, '{}.hdf5'.format(self.vis_input_type)), 'r')
        features = torch.from_numpy(hdf5_file[vid][:]).float()
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

    def get_i3d_video_features(self, vid):
        npy_dir = os.path.join(self.data_dir, '{}'.format(self.vis_input_type))
        npy_file = npy_dir + '/' + vid +'.npy'
        features = torch.from_numpy(np.load(npy_file)).float()
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

    def get_c3d_video_features(self, vid):
        pt_dir = os.path.join(self.data_dir, '{}'.format(self.vis_input_type))
        pt_file = pt_dir + '/' + vid +'.pt'
        features = torch.load(pt_file).float()
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

    def get_target_weights(self):
        num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        pos_count = [0 for _ in range(num_clips)]
        total_count = [0 for _ in range(num_clips)]
        pos_weight = torch.zeros(num_clips, num_clips)
        for anno in self.annotations:
            video_id = anno['video']
            gt_s_time, gt_e_time = anno['times']
            duration = self.durations[video_id]
            s_times = torch.arange(0, num_clips).float() * duration / num_clips
            e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
            overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                        e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                           torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)
            overlaps[overlaps >= 0.5] = 1
            overlaps[overlaps < 0.5] = 0
            for i in range(num_clips):
                s_idxs = list(range(0, num_clips - i))
                e_idxs = [s_idx + i for s_idx in s_idxs]
                pos_count[i] += sum(overlaps[s_idxs, e_idxs])
                total_count[i] += len(s_idxs)

        for i in range(num_clips):
            s_idxs = list(range(0, num_clips - i))
            e_idxs = [s_idx + i for s_idx in s_idxs]
            # anchor weights
            # pos_weight[s_idxs,e_idxs] = pos_count[i]/total_count[i]
            # global weights
            pos_weight[s_idxs, e_idxs] = sum(pos_count) / sum(total_count)


        return pos_weight


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
