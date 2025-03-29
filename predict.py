import cv2
from ultralytics import YOLO
from model import MyResnet50
from model_custom import SimpleCNN
import torch
import argparse
import numpy as np
import torch.nn as nn
import os

def get_args():
    parser = argparse.ArgumentParser(description="CNN inference")
    # parser.add_argument("--image-path", "-p", type=str, default="/home/amin/PycharmProjects/PythonProject/PriceVisonMultiColor/data/test/closeup_blue/closeup_blue_17.jpg")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--checkpoint", "-c", type=str, default="trained_models/best_cnn.pt")
    args = parser.parse_args()
    return args

def resize_with_padding(img, target_size=(224, 224), pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    top = (target_size[1] - new_h) // 2
    bottom = target_size[1] - new_h - top
    left = (target_size[0] - new_w) // 2
    right = target_size[0] - new_w - left

    padded_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded_img

def cls_obj(image,model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("check2")
    categories_item = ['laundryDetergent', 'snack']
    categories_color = ["blue", "green", "orrange", "red"]
    model.eval()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # image = resize_with_padding(image)

    image = np.transpose(image, (2, 0, 1)) / 255.0
    image = image[None, :, :, :]  # 1 x 3 x 224 x 224
    image = torch.from_numpy(image).to(device).float()
    softmax = nn.Softmax()
    with torch.no_grad():
        item, color = model(image)
        probs_item = softmax(item)
        probs_color = softmax(color)
    max_idx_item = torch.argmax(probs_item)
    max_idx_color = torch.argmax(probs_color)
    predicted_class_item = categories_item[max_idx_item]
    predicted_class_color = categories_color[max_idx_color]
    return predicted_class_item,predicted_class_color

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = MyResnet50(2, 4)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        classifier.load_state_dict(checkpoint["model"])
    else:
        print("No checkpoint found!")
        exit(0)

    detecter = YOLO("/home/amin/PycharmProjects/PythonProject/PriceVisonMultiColor/runs/detect/train/weights/best.pt")
    cap = cv2.VideoCapture("/home/amin/PycharmProjects/PythonProject/PriceVisonMultiColor/video/Ko.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    delay = int(1000 / fps)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    os.makedirs("output", exist_ok=True)
    out = cv2.VideoWriter("output/output.mp4", fourcc, fps, (720, 720))
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        frame = cv2.resize(frame, (720, 720))
        results = detecter.predict(frame,conf=0.2)

        for cls, box in zip(results[0].boxes.cls, results[0].boxes.xyxy):
            class_id = int(cls)
            xmin,ymin,xmax,ymax = map(int,box.tolist())
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            img = frame[ymin:ymax,xmin:xmax]
            item, color = cls_obj(img, classifier)
            text = "{}_{}".format(item,color)
            cv2.putText(frame,text,(xmin,ymin - 10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        out.write(frame)
        # cv2.imshow("a",frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    # cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done!")
