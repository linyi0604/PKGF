import os
from torchvision import transforms
from dataset import dataset_processing
from torch.utils.data import DataLoader
from model.network import VGG16_PK
import torch
from torch import nn
from utils.utils import Logger, AverageMeter, time_to_str
from timeit import default_timer as timer
import numpy as np
from utils.report import report_precision_se_sp_yi
from utils.utils import data_augmentadion

class Config(object):
    cross_validation_index = ["1"]
    def __init__(self, cross_val_index=cross_validation_index[0]):
        self.gpu_id = "0"
        # self.cross_validation_index = ["0", "1", "2", "3", "4"]
        self.image_path = "../dataset/ACNE04/JPEGImages/"
        self.mask_path = "./results/mask_predict/%s/" % cross_val_index
        self.train_mapping_path = '../dataset/ACNE04/NNEW_trainval_' + cross_val_index + '.txt'
        self.test_mapping_path = '../dataset/ACNE04/NNEW_test_' + cross_val_index + '.txt'
        self.network_input_size = (224, 224)
        self.batch_size = 16
        self.class_num = 4
        self.learning_rate = 0.001
        self.epoch = 120
        self.model_save_path = "./results/model_state_dict/vgg16_pk/model_%s.pkl" % cross_val_index
        self.log_file = "./results/logs/vgg16_pk/%s.log"%cross_val_index
        self.input_channel = 4
        self.num_worker = self.batch_size




def train_test(cross_val_index):

    config = Config(cross_val_index)
    log = Logger()
    log.open(config.log_file, mode="a")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id


    dset_train = dataset_processing.DatasetProcessing_image_PK_Opencv(
    config.image_path, config.mask_path, config.train_mapping_path,
    transform=data_augmentadion())

    dset_test = dataset_processing.DatasetProcessing_image_PK_Opencv(
        config.image_path, config.mask_path, config.test_mapping_path,
        transform=data_augmentadion(train=False))


    train_loader = DataLoader(dset_train,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_worker)

    test_loader = DataLoader(dset_test,
                             batch_size=config.batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_worker)

    model = VGG16_PK(input_channel=config.input_channel, class_num=config.class_num, pretrained=True).cuda()
    # model.load_state_dict(torch.load("./results/model_state_dict/vgg16_pk/model_1.pkl"))


    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.learning_rate, weight_decay=5e-4, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    start = timer()
    max_acc = 0
    best_report = ""

    for epoch in range(config.epoch):
        losses = AverageMeter()
        for step, (img, label, lesion_numer) in enumerate(train_loader):

            img = img.cuda()
            label = label.cuda()
            # train
            model.train()
            pre = model(img)
            loss = loss_func(pre, label)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            losses.update(loss.item(), img.size(0))
        message = '%s  | %0.3f | %0.3f | %s\n' % (
            "train", epoch,
            losses.avg,
            time_to_str((timer() - start), 'min'))

        log.write(message)

        # test process
        with torch.no_grad():
            test_loss = 0
            test_corrects = 0
            y_true = np.array([])
            y_pred = np.array([])

            model.eval()
            for step, (test_img, test_label, test_lesions_number) in enumerate(test_loader):

                test_img = test_img.cuda()
                test_label = test_label.cuda()
                y_true = np.hstack((y_true, test_label.data.cpu().numpy()))

                b_pre = model(test_img)

                loss = loss_func(b_pre, test_label)
                test_loss += loss.data

                _, preds = torch.max(b_pre, 1)
                y_pred = np.hstack((y_pred, preds.data.cpu().numpy()))

                batch_corrects = torch.sum((preds == test_label)).data.cpu().numpy()
                test_corrects += batch_corrects

            # test_loss = test_loss.float() / len(test_loader)
            test_acc = test_corrects / len(test_loader.dataset)  # 3292  #len(test_loader)

            _, _, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true)

            if test_acc > max_acc:
                max_acc = test_acc
                best_report = str(pre_se_sp_yi_report) + "\n"
                torch.save(model.state_dict(), config.model_save_path)
            if True:
                log.write(str(pre_se_sp_yi_report) + '\n')
                log.write("best result until now: \n")
                log.write(str(best_report) + '\n')
        log.write("best result: \n")
        log.write(str(best_report) + '\n')
        print(epoch)
        print(str(pre_se_sp_yi_report))
        print(str(best_report))



if __name__ == '__main__':
    for cross_val_index in Config.cross_validation_index:
        train_test(cross_val_index)
