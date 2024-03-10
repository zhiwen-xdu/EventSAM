import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import os
import numpy as np
import cv2


class RGBEData(Dataset):
    def __init__(self, root):
        self.root = root
        self.data_paths = [line.rstrip() for line in open(os.path.join(self.root, 'eventsam_list.txt'))]
        print('The size of data is %d' % (len(self.data_paths)))
        self.image_pixel_mean =  torch.Tensor([0.485,0.456,0.406]).view(-1, 1, 1)
        self.image_pixel_std = torch.Tensor([0.229,0.224,0.225]).view(-1, 1, 1)
        self.evimg_pixel_mean = torch.Tensor([0.485,0.456,0.406]).view(-1, 1, 1)
        self.evimg_pixel_std = torch.Tensor([0.229,0.224,0.225]).view(-1, 1, 1)

    def __len__(self):
        return len(self.data_paths)

    def read_file_paths(self,index):
        all_paths = self.data_paths[index]
        image_path, evimg_path = all_paths.split("  ")
        return image_path, evimg_path

    def __getitem__(self, index):
        image_path, evimg_path, mask_path = self.read_file_paths(index)
        image = cv2.imread(image_path)
        evimg = cv2.imread(evimg_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        evimg = cv2.cvtColor(evimg, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(image)
        evimg = F.to_tensor(evimg)
        image = (image - self.image_pixel_mean) / self.image_pixel_std
        evimg = (evimg - self.evimg_pixel_mean) / self.evimg_pixel_std

        return image,evimg,mask                          # [3,260,346],[3,260,346]


if __name__ == "__main__":
    dataset = RGBEData('.../RGBE-SEG/')
    EventDataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    for image,evimg,_ in EventDataLoader:
        print(image.shape,evimg.shape)
        a = image[0].numpy()
        b = evimg[0].numpy()
        cv2.namedWindow('image')
        cv2.namedWindow('evimg')
        a = np.transpose(a, (1, 2, 0))
        b = np.transpose(b, (1, 2, 0))
        cv2.imshow('image', a)
        cv2.imshow('evimg', b)
        cv2.waitKey(1000)
