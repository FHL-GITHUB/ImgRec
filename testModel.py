import torch
from efficientnet_pytorch import EfficientNet
import cv2
from torch.utils.data import DataLoader
import os
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from torchvision.datasets import ImageFolder
from extract import extractBBox
import numpy as np
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y
def test():

    img = cv2.imread("./model_test/up/1.jpg")
    img1 = img
    boxes, bb, images = extractBBox(img)
    font = cv2.FONT_HERSHEY_SIMPLEX

    N = len(boxes)
    for i in range(N):
        cv2.rectangle(img1, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), (0, 255, 0),2)

    classes = ['0', '6', '7', '8', '9', "Bull", 'Down', 'Left', 'Right', 'Stop', 'Up', 'V', 'W', 'X', 'Y', 'Z']
    net = torch.load("./model_dic.pt")
    net.eval()

    VALIDATE_PATH = os.path.join(os.getcwd(), './model_test')


    transform = Compose([ToTensor(), Resize((224, 224)), Grayscale(3)])

    validate_data = ImageFolder(root=VALIDATE_PATH, transform=transform)
    validate_ds = DataLoader(validate_data, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
    for img, lbl in validate_ds:
        scores = net(img)
        result_label = softmax(scores.detach().numpy()).argmax()
        result = softmax(scores.detach().numpy())[0][result_label]
        label = scores.argmax(dim=1).numpy()

    cv2.putText(img1, str(classes[label[0]]), (50, 50), font, 1.2, (0, 255, 0), 2)
    cv2.putText(img1, str(result), (50, 100), font, 1.2, (0, 255, 0), 2)
    path="./model_test/up/result/"+str(classes[label[0]])+".jpg"
    cv2.imwrite(path, img1)

    return str(classes[label[0]]), result_label


if __name__ == '__main__':
    target,label=test()
    print(target,label)
