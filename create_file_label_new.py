import os
import pandas as pd
data = {}

def create_file(train=True):
    images_name = []
    items = []
    colors = []
    file_label = "train" if train else "test"

    for img_name in os.listdir("data_new/{}".format(file_label)):
        images_name.append(img_name)
        item,color = img_name.split(".")[0].split("_")[:2]
        items.append(item)
        colors.append(color)

    data["filename"] = images_name
    data["item"] = items
    data["color"] = colors
    df = pd.DataFrame(data)
    df.to_csv("label_{}_final.csv".format(file_label),index=False)

if __name__ == '__main__':
    create_file(train=False)
# a,b = name.split(".")[0].split("_")[:2]
# print(a,b)