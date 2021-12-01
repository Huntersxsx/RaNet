from torch import nn
import torch
import math
import numpy as np

from core.config import config
import models.feature_encoder as feature_encoder
import models.choice_generator as choice_generator
import models.modality_interactor as modality_interactor
import models.relation_constructor as relation_constructor


class RaNet(nn.Module):
    def __init__(self):
        super(RaNet, self).__init__()

        self.encoder_layer = getattr(feature_encoder, config.RANET.ENCODER_LAYER.NAME)(config.RANET.ENCODER_LAYER.PARAMS)
        self.generator_layer = getattr(choice_generator, config.RANET.GNERATOR_LAYER.NAME)(config.RANET.GNERATOR_LAYER.PARAMS)
        self.interactor_layer = getattr(modality_interactor, config.RANET.INTERACTOR_LAYER.NAME)(config.RANET.INTERACTOR_LAYER.PARAMS)
        self.relation_layer = getattr(relation_constructor, config.RANET.RELATION_LAYER.NAME)(config.RANET.RELATION_LAYER.PARAMS)
        self.pred_layer = nn.Conv2d(config.RANET.PRED_INPUT_SIZE, 1, 1, 1)

    def forward(self, textual_input, textual_mask, visual_input):

        vis_h, txt_h = self.encoder_layer(visual_input, textual_input, textual_mask) 
        choice_map, map_mask = self.generator_layer(vis_h) 
        fused_map = self.interactor_layer(choice_map, txt_h)  
        relation_map = self.relation_layer(fused_map, map_mask)  
        score_map = self.pred_layer(relation_map) * map_mask.float()  

        return score_map, map_mask


