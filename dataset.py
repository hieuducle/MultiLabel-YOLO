import os.path

from torch.utils.data import Dataset
import pandas as pd
import cv2
from torchvision.transforms import Compose,ToTensor,Resize,Normalize
class PriceVision(Dataset):
    def __init__(self,label_path,train=True,transform=None):
        self.df = pd.read_csv(label_path)
        self.item_categories = sorted(self.df["item"].unique())
        self.color_categories = sorted(self.df["color"].unique())
        self.transform = transform
        self.train = train

        self.item_label = {}
        self.color_label = {}
        for idx, item in enumerate(self.item_categories):
            self.item_label[item] = idx
        for idx, color in enumerate(self.color_categories):
            self.color_label[color] = idx
        print(self.item_label,self.color_label)

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):

        img_path = self.df.iloc[idx]["filename"]
        if self.train:
            img_path = os.path.join("data_new/train",img_path)
        else:
            img_path = os.path.join("data_new/test", img_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        item = self.item_label[self.df.iloc[idx]["item"]]
        color = self.color_label[self.df.iloc[idx]["color"]]
        return img,item,color
if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = PriceVision("label_test_4_item.csv",train=False,transform=transform)
    img,item,color = dataset.__getitem__(0)


    # image = cv2.imread(img)
    # cv2.imshow("a",image)
    # # image = cv2.imread(path)
    # # print(path, item, color)
    # # cv2.imshow("a",image)
    # cv2.waitKey(0)
