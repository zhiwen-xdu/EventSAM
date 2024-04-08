# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from functools import partial

from segment_anything.modeling import Source_ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer



def build_evimg_sam_vit_b(input_signal,encoder_checkpoint,decoder_checkpoint):
    return _build_evimg_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        input_signal=input_signal,
        encoder_checkpoint=encoder_checkpoint,
        decoder_checkpoint=decoder_checkpoint,
    )

def build_evimg_sam_vit_h(input_signal,encoder_checkpoint,decoder_checkpoint):
    return _build_evimg_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        input_signal=input_signal,
        encoder_checkpoint=encoder_checkpoint,
        decoder_checkpoint=decoder_checkpoint,
    )


sam_evimg_model_registry = {
    "vit_b": build_evimg_sam_vit_b,
    "vit_h": build_evimg_sam_vit_h,
}


def _build_evimg_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    input_signal="img",
    encoder_checkpoint=None,
    decoder_checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 512
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=Source_ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    sam_dict = sam.state_dict()


    # For Image/Evimg Encoder Weight Load
    encoder_dict = torch.load(encoder_checkpoint, map_location="cpu")
    decoder_dict = torch.load(decoder_checkpoint, map_location="cpu")

    if input_signal == "img":
        for key,value in encoder_dict.items():
            if "image_encoder" in key:
                new_key = key[7:]
                sam_dict[new_key] = value
                if "pos_embed" in new_key:
                    sam_dict[new_key] = value[:,:32,:32,:]

    elif input_signal == "evimg":
        for key,value in encoder_dict.items():
            if "evimg_encoder" in key:
                new_key = key[7:]
                new_key = new_key.replace("evimg_encoder","image_encoder")
                sam_dict[new_key] = value
                if "pos_embed" in new_key:
                    sam_dict[new_key] = value[:,:32,:32,:]

    # For Source Decoder Weight Load
    for key, value in decoder_dict.items():
        if "image_encoder" not in key :
            # key: prompt_encoder.pe_layer ...
            sam_dict[key] = value

    sam.load_state_dict(sam_dict)

    return sam
