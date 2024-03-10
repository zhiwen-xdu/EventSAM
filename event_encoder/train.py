import os
import torch
import tqdm
# loss function related
from event_encoder.loss.loss_function import TotalLoss
# some more advanced functions
from event_encoder.data_utlis.data_loader import RGBEData
# network related
from models.build_mix_rgbe_encoder import _build_mix_rgbe_encoder_b


def run():
    print('=====Training script for event_vit_encoder')
    # Build dataloaders
    RGBEDataset = RGBEData('.../RGBE-SEG/')
    RGBELoader = torch.utils.data.DataLoader(dataset=RGBEDataset, batch_size=16, shuffle=True)

    # Create the teacher and student encoders
    RGBE_Encoder = _build_mix_rgbe_encoder_b()

    # MultiGPU Train
    RGBE_Encoder = torch.nn.DataParallel(RGBE_Encoder).cuda()

    for name, param in RGBE_Encoder.named_parameters():
        param.requires_grad = False

    train_block_list = ["evimg_encoder.patch_embed"] + ["evimg_encoder.blocks." + str(i) + ".mlp" for i in [2, 5, 8, 11]]
    for name, param in RGBE_Encoder.named_parameters():
        for block_name in train_block_list:
            if block_name in name:
                param.requires_grad = True

    train_para_name_list = []
    for name, param in RGBE_Encoder.named_parameters():
        if param.requires_grad:
            train_para_name_list.append(name)


    print("=====Train Params:",train_para_name_list)
    # # optimizer
    optimizer = torch.optim.Adam(params=[{'params': [p for name, p in RGBE_Encoder.named_parameters() if name in train_para_name_list]}], lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    # Loss functions
    loss_fun = TotalLoss(block_feature_indexes=[2, 5, 8, 11], block_loss_weight=[0.1, 0.4, 0.7, 1.0])

    iteration = 0
    size = 0
    RGBE_Encoder.train()
    for epoch in range(5):
        loss_sum = 0.0
        source_loss_sum_list = [0.0, 0.0, 0.0, 0.0,0.0]
        loss_sum_list = [0.0, 0.0, 0.0, 0.0,0.0]
        print('=====epoch ' + str(epoch))
        for images, evimgs in RGBELoader:
            images = images.cuda()
            evimgs = evimgs.cuda()

            optimizer.zero_grad()

            image_embeddings_dict, evimg_embeddings_dict,token_weights_dict = RGBE_Encoder(images, evimgs)
            loss,source_loss_list,loss_list = loss_fun(image_embeddings_dict, evimg_embeddings_dict, token_weights_dict)

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            source_loss_sum_list = [x + y.item() for x, y in zip(source_loss_sum_list, source_loss_list)]
            loss_sum_list = [x + y.item() for x, y in zip(loss_sum_list, loss_list)]

            iteration += 1
            size += 1

            if iteration % 100 == 0:
                print('iteration:', iteration, 'loss:', loss_sum / 100, "source_loss_list:",[x/100 for x in source_loss_sum_list], 'loss_list:', [x / 100 for x in loss_sum_list])
                loss_sum = 0.0
                source_loss_sum_list = [0.0, 0.0, 0.0, 0.0,0.0]
                loss_sum_list = [0.0, 0.0, 0.0, 0.0,0.0]
            if iteration % 2000 == 0:
                torch.save(RGBE_Encoder.state_dict(), '../checkpoints/rgbe_encoder_0%d_iter.pth' % (iteration // 2000))

        if epoch % 3 == 0:
            scheduler.step()
        print('iteration:', iteration)
        print('loss:', loss_sum/size)
        size = 0


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    run()

