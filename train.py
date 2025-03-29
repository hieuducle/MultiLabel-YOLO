from torch.utils.data import DataLoader
from dataset import PriceVision
from model import MyResnet50
from model_custom import SimpleCNN
import torch
import argparse
import torch.nn as nn
import torch.optim
from torchvision.transforms import Compose, ToTensor,Normalize,Resize
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

def get_args():
    parser = argparse.ArgumentParser(description="item classifier")
    parser.add_argument("--batch-size","-b",type=int,default=16)
    parser.add_argument("--epochs","-e",type=int,default=100)
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args


def train(args):
    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = PriceVision(label_path="label_train_final.csv",train=True,transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    test_dataset = PriceVision(label_path="label_test_final.csv",train=False,transform=transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.trained_models,exist_ok=True)
    model = MyResnet50(2,4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_acc = 0
    num_iters = len(train_dataloader)
    for epoch in range(start_epoch,args.epochs):
        model.train()
        progress_bar_train = tqdm(train_dataloader,colour="green")
        for iter, (images, item_labels,color_labels) in enumerate(progress_bar_train):
            images = images.to(device)
            item_labels = item_labels.to(device)
            color_labels = color_labels.to(device)

    #         forward
            item_preds,color_preds = model(images)

            item_loss = criterion(item_preds,item_labels)
            color_loss = criterion(color_preds, color_labels)
            losses = item_loss + color_loss
            # loss_value = criterion(outputs,labels)
            progress_bar_train.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, args.epochs, iter+1, num_iters, losses))

    #         backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        model.eval()
        all_item_predictions = []
        all_color_predictions = []
        all_item_labels = []
        all_color_labels = []
        progress_bar_test = tqdm(test_dataloader, colour="green")
        for iter, (images,item_labels,color_labels) in enumerate(progress_bar_test):
            all_item_labels.extend(item_labels)
            all_color_labels.extend(color_labels)

            images = images.to(device)
            item_labels = item_labels.to(device)
            color_labels = color_labels.to(device)

            with torch.no_grad():
                item_preds,color_preds = model(images)
                indices_item = torch.argmax(item_preds.cpu(),dim=1)
                indices_color = torch.argmax(color_preds.cpu(), dim=1)

                all_item_predictions.extend(indices_item)
                all_color_predictions.extend(indices_color)
                # item_loss = criterion(item_preds,item_labels)
                # color_loss = criterion(color_preds,color_labels)

        all_item_labels = [item_labels.item() for item_labels in all_item_labels]
        all_item_predictions = [item_preds.item() for item_preds in all_item_predictions]

        all_color_labels = [color_labels.item() for color_labels in all_color_labels]
        all_color_predictions = [color_preds.item() for color_preds in all_color_predictions]
        accuracy_item = accuracy_score(all_item_labels, all_item_predictions)
        accuracy_color = accuracy_score(all_color_labels, all_color_predictions)

        # progress_bar_test.set_description("Epoch {}: Accuracy item: {}".format(epoch + 1, accuracy_item))
        # progress_bar_test.set_description("Epoch {}: Accuracy color: {}".format(epoch + 1, accuracy_color))
        print("Epoch {}: Accuracy item: {}".format(epoch + 1, accuracy_item))
        print("Epoch {}: Accuracy color: {}".format(epoch + 1, accuracy_color))


        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if accuracy_item > best_acc:
            best_acc = accuracy_item
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))



if __name__ == '__main__':
    args = get_args()
    train(args)


