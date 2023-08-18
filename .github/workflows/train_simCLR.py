import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from torchvision import datasets

from models import utils
from models.SimCLR import Model
import torch.nn.functional as F
from sklearn import model_selection

device = "cuda:7" if torch.cuda.is_available() else "cpu"
print("SimCLR Using {} device".format(device))


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for (pos_1, pos_2), target in train_bar:
        pos_1, pos_2 = pos_1.to(device), pos_2.to(device)
        loss = net(pos_1, pos_2)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    c = len(memory_data_loader.dataset.classes)
    print(c)
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net.f(data.to(device))
            feature = torch.flatten(feature, start_dim=1)
            feature = F.normalize(feature, dim=-1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature = net.f(data)
            feature = torch.flatten(feature, start_dim=1)
            feature = F.normalize(feature, dim=-1)
            total_num += data.size(0)

            # KNN
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            # KNN end

            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')  # 潜在矢量的特征尺寸
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Number of images in each mini-batch')  # 1024  512  256
    parser.add_argument('--epochs', default=500, type=int,
                        help='Number of sweeps over the dataset to train')  # 1000  500  250
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='Name of the pretrained encoder: CIFAR10 / IMAGENET / GTSRB')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    # data prepare
    train_loader, test_loader, memory_loader = utils.get_dataset('SimCLR', args.dataset, args.batch_size)

    # model setup and optimizer config
    model = Model(feature_dim, args.batch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # training loop
    save_name_pre = '{}_{}'.format(batch_size, epochs)  # (feature_dim, temperature, k, batch_size, epochs)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)

        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'Checkpoints/simCLR_{}_{}.pth'.format(args.dataset, save_name_pre))
