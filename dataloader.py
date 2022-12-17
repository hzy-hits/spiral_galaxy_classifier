import numpy as np
import sys
import torch
from tqdm import tqdm
import torch.nn



def train_one_epoch(model, optimizer, data_loader, device, epoch,):

    model.train()
    #pos_weight=torch.tensor([5]).to(device)
    weight = torch.from_numpy(np.array([2, 1,])).float()
    loss_function = torch.nn.CrossEntropyLoss(weight.to(device))
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()
    #sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images,labels=data
        pred = model(images.to(device))


        #pred_classes = torch.max(pred, dim=1)[1]
        loss = loss_function(pred.reshape(-1,2), labels.to(device))
        accu_num += torch.eq(torch.argmax(pred.reshape(-1,2),dim=1), torch.argmax(labels.to(device))).sum()
        loss.backward()#反向传播
        accu_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            #if (epoch + 1) % batch_size == 0:
        optimizer.step()#优化器优化
        optimizer.zero_grad()#梯度清零
        data_loader.desc = "[train epoch {}] loss: {:.6f} acc: {:.2f}%".format(
        epoch, loss.item(),(accu_num.item()/(512*step+512)*100))
            #del images, labels
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
    return accu_loss.item() / (step + 1),(accu_num.item()/(512*step+512)*100)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    weight = torch.from_numpy(np.array([2, 1, ])).float()
    loss_function = torch.nn.CrossEntropyLoss(weight.to(device))
    #result = []
    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    #sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images,labels=data
        pred = model(images.to(device))
        #pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(torch.argmax(pred.reshape(-1,2),dim=1), torch.argmax(labels.to(device))).sum()
        loss = loss_function(pred.squeeze(-1), labels.to(device))
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.6f} acc: {:.2f}%".format(
        epoch,
        loss.item(), (accu_num.item()/(512*step+512)*100))

    return accu_loss.item() / (step + 1),(accu_num.item()/(512*step+512)*100)
@torch.no_grad()
def test(model, data_loader, device, ):
    #weight = torch.from_numpy(np.array([0.9, 0.1, ])).float()
    #loss_function = torch.nn.CrossEntropyLoss(weight.to(device))
    result = []
    model.eval()

    #accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    #accu_loss = torch.zeros(1).to(device)  # 累计损失

    #sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images,names=data
        pred = model(images.to(device))
        #pred_classes = torch.max(pred, dim=1)[1]
        #accu_num += torch.eq(torch.argmax(pred.reshape(-1,2),dim=1), torch.argmax(labels.to(device))).sum()
        if torch.argmax(pred.reshape(-1,2),dim=-1)[0] ==torch.zeros(1).to(device):
            result.append(names)
        #loss = loss_function(pred.squeeze(-1), labels.to(device))
        #accu_loss += loss
        #data_loader.desc = "[valid epoch {}] acc: {:.2f}%".format(
        #epoch, (accu_num.item()/(step+1)*100))

    return result