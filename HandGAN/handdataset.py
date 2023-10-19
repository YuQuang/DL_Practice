import os, re
import cv2 as cv2
from torch.utils.data.dataset import Dataset

class TrainDatasets(Dataset):
    def __init__(self, img_path, file_regex) -> None:
        super().__init__()
        self.img_path = img_path
        self.file_regex = file_regex
        self.all_file_list = []

        for _, _, files in os.walk(self.img_path):
            for file_name in files:
                if re.match(self.file_regex, file_name):
                    self.all_file_list.append(file_name)

    def __getitem__(self, index):
        img = cv2.imread(
            os.path.join(self.img_path, self.all_file_list[index])
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256))
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return img.reshape(1, 256, 256), 1
    
    def __len__(self) -> int:
        return self.all_file_list.__len__()
