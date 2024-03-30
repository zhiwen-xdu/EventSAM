import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from segment_anything.modeling import Mix_ImageEncoderViT,Mix_EvimgEncoderViT


class Mix_RGBE_Encoder(nn.Module):
    def __init__(
        self,
        image_encoder: Mix_ImageEncoderViT,
        evimg_encoder: Mix_EvimgEncoderViT,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.
        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.evimg_encoder = evimg_encoder


    @property
    def device(self):
        return self.image_pixel_mean.device

    # @torch.no_grad()
    def forward(self,images,evimgs):
        """
        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
        """
        input_images = self.image_preprocess(images)
        input_evimgs = self.evimg_preprocess(evimgs)
        image_embeddings_dict, token_weights_dict = self.image_encoder(input_images)
        evimg_embeddings_dict = self.evimg_encoder(input_evimgs,image_tokens)

        return image_embeddings_dict,evimg_embeddings_dict,token_weights_dict


    def image_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def evimg_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = self.evimg_encoder.img_size - h
        padw = self.evimg_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


def _build_mix_rgbe_encoder_b():
        rgbe_encoder = Mix_RGBE_Encoder(image_encoder=Mix_ImageEncoderViT(
                med_feature_indexes=[2, 5, 8, 11],
                depth=12,
                embed_dim=768,
                img_size=512,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12,
                patch_size=16,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=[2, 5, 8, 11],
                window_size=14,
                out_chans=256),
                evimg_encoder=Mix_EvimgEncoderViT(
                med_feature_indexes=[2, 5, 8, 11],
                depth=12,
                embed_dim=768,
                img_size=512,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12,
                patch_size=16,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=[2, 5, 8, 11],
                window_size=14,
                out_chans=256),)

        rgbe_encoder_dict = rgbe_encoder.state_dict()

        checkpoint_path =".../EventSAM/pretrained/sam_vit_b.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        for key,value in checkpoint.items():
            if "image_encoder" in key:
                new_key = key[14:]
                rgbe_encoder_dict["image_encoder." + new_key] = value
                rgbe_encoder_dict["evimg_encoder." + new_key] = value
                if "pos_embed" in new_key:
                    rgbe_encoder_dict["image_encoder." + new_key] = value[:,:32,:32,:]
                    rgbe_encoder_dict["evimg_encoder." + new_key] = value[:,:32,:32,:]
                if "rel_pos_h" or "rel_pos_w" in new_key:
                    if value.shape[0] == 127:
                        rgbe_encoder_dict["image_encoder." + new_key] = value[32:-32, :]
                        rgbe_encoder_dict["evimg_encoder." + new_key] = value[32:-32, :]

        rgbe_encoder.load_state_dict(rgbe_encoder_dict)

        return rgbe_encoder





