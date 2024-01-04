import torch
import torch.nn as nn


def compute_attn_weighted_similarity_loss(image_embeddings_dict, evimg_embeddings_dict, block_feature_indexes, block_loss_weight, token_weights_dict):
    # image_embeddings_dict["block_i"]: [N,768,32,32]
    # attn_weights_dict: [N,1,32,32]
    similarity_loss = nn.L1Loss()
    total_similarity_loss = 0.
    source_similarity_loss_list = []
    similarity_loss_list = []
    for i in range(len(token_weights_dict)):
        block_index = block_feature_indexes[i]
        block_weight = block_loss_weight[i]
        tokens_weight = token_weights_dict["block_{}".format(block_index)]
        weighted_image_emdeddings = image_embeddings_dict["block_{}".format(block_index)] * tokens_weight
        weighted_evimg_emdeddings = evimg_embeddings_dict["block_{}".format(block_index)] * tokens_weight
        source_block_loss = similarity_loss(image_embeddings_dict["block_{}".format(block_index)], evimg_embeddings_dict["block_{}".format(block_index)])
        block_loss = similarity_loss(weighted_image_emdeddings,weighted_evimg_emdeddings)

        total_similarity_loss += block_weight * block_loss
        source_similarity_loss_list.append(source_block_loss)
        similarity_loss_list.append(block_loss)

    return total_similarity_loss,source_similarity_loss_list,similarity_loss_list


class TotalLoss(torch.nn.Module):
    def __init__(self, block_feature_indexes,block_loss_weight):
        super(TotalLoss, self).__init__()
        self.block_feature_indexes = block_feature_indexes
        self.block_loss_weight = block_loss_weight

    def forward(self, image_embeddings_dict, evimg_embeddings_dict, token_weights_dict):
        similarity_loss,source_similarity_loss_list,similarity_loss_list = compute_attn_weighted_similarity_loss(image_embeddings_dict, evimg_embeddings_dict, self.block_feature_indexes, self.block_loss_weight, token_weights_dict)
        return similarity_loss,source_similarity_loss_list,similarity_loss_list