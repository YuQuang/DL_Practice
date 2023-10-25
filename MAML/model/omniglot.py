import os
import glob
import torch
import random
import itertools
import cv2 as cv2
import numpy as np
from torch.utils.data.dataset import Dataset

class Omniglot(Dataset):
    def __init__(self, path, n_way=5, k_shot=5, k_query=15, background=True):
        #
        # Parameters
        #
        self.n_way    = n_way
        self.k_shot   = k_shot
        self.k_query  = k_query
        self.img_size = 224
        self.background = background
        
        #
        # Total class combination
        #
        image_path = "images_background" if self.background else "images_evaluation" 
        all_dir = [ f for f in glob.glob(os.path.join(path, image_path, "*", "*")) if os.path.isdir(f) ]
        random.shuffle(all_dir)
        self.all_dir_combination = [ comb for comb in itertools.combinations(all_dir[:100], n_way)]
        random.shuffle(self.all_dir_combination)
        
    def __getitem__(self, index):
        train_dir_list = self.all_dir_combination[index]

        support_label_list, support_set_list = [], []
        query_label_list, query_set_list = [], []
        for index, train_dir in enumerate(train_dir_list):
            image_list = random.choices(
                glob.glob(os.path.join(train_dir, "*.png")),
                k=self.k_shot
            )
            #
            # [
            #   [ [ 1 class 1 image ], [ 2 class 1 image ], [ 3 class 1 image ]...],
            #   [ [ n class k_shot images ]...],
            #   ... batch_size lines
            # ]
            #
            for img_path in image_list:
                img = cv2.imread(img_path)
                img = (img - img.min()) / (img.max() - img.min())
                img = cv2.resize(img, (self.img_size, self.img_size))
                support_set_list.append(
                    img.reshape((3, self.img_size, self.img_size))
                )
            #
            # [
            #   [1, 1, 1, 1, 1, 2, 2, 2, 2, 2 ...],
            #   [1, 1, 1, 1, 1, 2, 2, 2, 2, 2 ...],
            #   ... batch_size lines
            # ]
            #
            for _ in range(self.k_shot):
                support_label_list.append(index)

            #
            # These part is for query set
            #
            image_list = random.choices(
                glob.glob(os.path.join(train_dir, "*.png")),
                k=self.k_query
            )

            for img_path in image_list:
                img = cv2.imread(img_path)
                img = (img - img.min()) / (img.max() - img.min())
                img = cv2.resize(img, (self.img_size, self.img_size))
                query_set_list.append(
                    img.reshape((3, self.img_size, self.img_size))
                )

            for _ in range(self.k_query):
                query_label_list.append(index)
        
        #
        # Return support_set, support_label
        #        query_set, query_label
        #
        return torch.as_tensor(np.array(support_set_list)).float(), \
               torch.as_tensor(np.array(support_label_list)).long(), \
               torch.as_tensor(np.array(query_set_list)).float(), \
               torch.as_tensor(np.array(query_label_list)).long()
        
    def __len__(self):
        return self.all_dir_combination.__len__()
    
if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    dataset     = Omniglot("./omniglot/omniglot-py/", background=False)
    data_loader = DataLoader(dataset, batch_size=1)

    for spt_data, spt_label, qry_data, qry_label in data_loader:
        print(spt_data[0][0].numpy().shape)
        break
    