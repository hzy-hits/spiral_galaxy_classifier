import numpy as np
import os
import glob
import re
from data_loader import myDataset
from dataloader import evaluate
from dataloader import train_one_epoch
import torch
import torch.optim as optim
import torch.utils.data
import time
from model import ResNet18
import argparse

def process_files(input_list, foldername):
    sample_data = []
    fits_files = glob.glob(str(foldername) + '\\*.eps')
    # for x in fits_files:
    # regex = re.compile(r'\d+')
    # print(regex.findall(x)[2])

    # print(newfile)
    fits_files = [int(re.findall(r'\d+', string)[2]) for string in fits_files]
    # (str(foldername))
    print(fits_files)
    for file in fits_files:
        # print(int(file[:-4]))
        # file_name = int(file[:-4])

        if file not in input_list:
            sample_data.append(file)
            # print(file_name)
        # sample_data = sample_data
    #np.savetxt(foldername + 'no_sprial.txt', sample_data, fmt='%d')
    # print(sample_data)


def load_txt_files(folder_path):
    # 创建一个空列表，用来存储所有的数组
    arrays = []
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 检查文件的后缀是否为 '.txt'
        if file_name.endswith('.txt'):
            # 使用 numpy.loadtxt() 读取文件内容并存储到一个数组中
            file_path = os.path.join(folder_path, file_name)

            data = np.loadtxt(file_path)
            data = np.array(data, dtype=int)
            # print(data)
            process_files(data, folder_path + '\\' + file_name[:-4])

    # 返回存储所有数组的列表
    return arrays


"""
def resize_fits_file(filename, size):
    # 读取 FITS 文件
    fits_file = get_pkg_data_filename(filename)
    fits_data = open(fits_file)
    # 调整图像大小
    resized_data = cv.resize(fits_data[0].data, (size, size))
    return resized_data
"""


def main(args):
    torch.cuda.empty_cache()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # tb_writer = SummaryWriter()

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
    # args.data_path)

    # sadata = pd.read_csv("D:\\newidea\\newgalaxy\\data\\label\\train_labels2.csv")
    # sadata = pd.read_csv("D:\\newidea\\test_data.csv")

    # sadata = pd.read_csv("D:\\newidea\\newgalaxy\\date\\train0\\labels.csv")
    # path = "D:\\newidea\\newgalaxy\\data\\train1\\"
    csv_path = "D:\\study\\senior_fall\\thesis\\pre_project\\galaxy_figure\\train_data.csv"
    test_path = "D:\\study\\senior_fall\\thesis\\pre_project\\galaxy_figure\\test_data.csv"
    # path_list = os.listdir("D:\\newidea\\newgalaxy\\data\\train1\\")
    train_dataset = myDataset(csv_path)
    test_dataset = myDataset(test_path)
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0,
              8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        prefetch_factor=2

    )
    time.sleep(1)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        prefetch_factor=2,
    )

    # val_loader = torch.utils.data.DataLoader(val_dataset,
    # batch_size=batch_size,
    # shuffle=False,
    # pin_memory=True,
    # num_workers=nw,
    # collate_fn=val_dataset.collate_fn)

    model = ResNet18(classes_num=args.num_classes).to(device)
    if args.weights != "":
        assert os.path.exists(
            args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除有关分类类别的权重
        # for k in list(weights_dict.keys()):
        # if "head" in k:
        # del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-2)  # 定义优化器

    for epoch in range(args.epochs):
        # train
        total_result=[]
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch,)



        result = evaluate(model=model,
                          data_loader=test_loader,
                          device=device,
                          epoch=epoch)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))  # 保存模型
        total_result.append(result)
    total_result=np.array(total_result)
    np.savetxt(".\\test\\predict" + str(epoch) + ".txt",
                   total_result.reshape(-1,2),
                   fmt='%.8f')
        # 保存文件的方式
        # tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

    # print(result)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5E-4)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument(
        '--weights',
        type=str,
        default='')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
