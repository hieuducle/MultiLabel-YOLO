import cv2
import os
from sklearn.model_selection import train_test_split
import shutil

if __name__ == '__main__':

    root = "video"
    os.makedirs("data_new/train",exist_ok=True)
    os.makedirs("data_new/test", exist_ok=True)
    for vid in os.listdir(root):
        root_current = os.path.join(root,vid)
        cap = cv2.VideoCapture(root_current)
        os.makedirs("data_new/all_images",exist_ok=True)
        namevid_without_extension = os.path.splitext(vid)[0]
        x = 0
        all_images = []
        while cap.isOpened():
            flag,frame = cap.read()
            if not flag:
                break
            cv2.imwrite("data_new/all_images/{}_{}.jpg".format(namevid_without_extension,x),frame)
            x += 1
        for image_name in os.listdir("data_new/all_images"):
            all_images.append(image_name)
        train_images,test_images = train_test_split(all_images,test_size=0.2,random_state=42)

        for file_img in train_images:
            shutil.move("data_new/all_images/{}".format(file_img),"data_new/train/{}".format(file_img))
        for file_img in test_images:
            shutil.move("data_new/all_images/{}".format(file_img), "data_new/test/{}".format(file_img))
