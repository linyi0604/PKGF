from numpy import meshgrid
from regex import P
from dataset.dataset_processing import DatasetProcessing_image_mask_OpenCV
from model.network import Unetpp
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import os
import torchvision.utils as vutils
import segmentation_models_pytorch as smp
from utils.metrics import diceCoeff

from utils.utils import Logger, AverageMeter, data_augmentadion



class Config(object):
    model_name = "UNet++"
    cross_validation_index = ["1"]
    # cross_validation_index = ["0", "1", "2", "3", "4"]
    def __init__(self, cross_val_index=cross_validation_index[0]):
        self.gpu_id = "0"
        self.image_path = "../dataset/ACNE04/JPEGImages/"
        self.mask_path = "../dataset/ACNE04/mask/"
        self.train_mapping_path = '../dataset/ACNE04/NNEW_trainval_%s.txt' % cross_val_index
        self.test_mapping_path = '../dataset/ACNE04/NNEW_test_%s.txt' % cross_val_index
        self.model_save_path = "results/model_state_dict/segmentation/%s_%s.pkl" % (self.model_name, cross_val_index)
        self.log_path = "results/logs/segmentation/%s_%s.log" % (self.model_name, cross_val_index)

        self.network_input_size = (224, 224)
        self.batch_size = 16
        self.ecpoch = 200
        self.learning_rate = 1e-4

        self.num_workers = self.batch_size



config = Config()

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id


dset_train = DatasetProcessing_image_mask_OpenCV(
        config.image_path, config.mask_path, config.train_mapping_path,
        transform=data_augmentadion())

dset_test = DatasetProcessing_image_mask_OpenCV(
    config.image_path, config.mask_path, config.test_mapping_path,
    transform=data_augmentadion(train=False))



train_loader = DataLoader(dset_train,
                          batch_size=config.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=config.num_workers,
                          drop_last=False)

test_loader = DataLoader(dset_test,
                         batch_size=config.batch_size,
                         shuffle=False,
                         pin_memory=True,
                         num_workers=config.num_workers,
                         drop_last=True)



model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
model = model.cuda()


criterion = torch.nn.BCEWithLogitsLoss()
# criterion = torch.nn.MSELoss()
# criterion = smp.losses.DiceLoss('binary')
metric = smp.utils.metrics.IoU(threshold=0.5)


optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

max_test_iou = -10

log = Logger()
log.open(config.log_path, mode="a")

best_msg = "\nBest until now:"
for epoch in range(config.ecpoch):
    msg = ""

    model.train()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    for step, (img_name, img, mask, severity, lesion_numer) in enumerate(train_loader):
        img = img.cuda()
        mask = mask.cuda()
        # train
        pre_mask = model(img)

        loss1 = criterion(pre_mask, mask)
        # loss2 = criterion2(pre_mask, mask*10)
        # loss = loss1 + loss2
        loss = loss1

        losses.update(loss.item(), img.size(0))
        iou = metric(pre_mask, mask)
        dice = diceCoeff(pre_mask, mask)


        ious.update(iou.item(), img.size(0))
        dices.update(dice.item(), img.size(0))
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    msg += "\ntrain:%s\n"%epoch
    msg += "    loss:%.6f"%losses.avg
    msg += "    iou:%.6f"%ious.avg
    msg += "    dice:%.6f"%dices.avg

    visualize = []
    model.eval()
    test_losses = AverageMeter()
    test_ious = AverageMeter()
    test_dices = AverageMeter()
    with torch.no_grad():
        for step, (img_name, img, mask, severity, lesion_numer) in enumerate(test_loader):
            img = img.cuda()
            mask = mask.cuda()
            pre_mask = model(img)

            loss = criterion(pre_mask, mask)
            iou = metric(pre_mask, mask)
            dice = diceCoeff(pre_mask, mask)
            test_losses.update(loss.item(), img.size(0))
            test_ious.update(iou.item(), img.size(0))
            test_dices.update(dice.item(), img.size(0))
            visualize.append(mask)
            visualize.append(pre_mask)
    
    msg += "\ntest:\n"
    msg += "    loss:%6f"%test_losses.avg
    msg += "     iou:%6f"%test_ious.avg
    msg += "     dice:%6f"%test_dices.avg
    
    

    if max_test_iou < test_ious.avg:
        best_msg = "\nBest until now:" + msg
        max_test_iou = test_ious.avg
        torch.save(model.state_dict(), config.model_save_path)

        visualize = torch.cat(visualize, dim=3)
        vutils.save_image(visualize,  "./results/prediction/test_mask.png")
        
    log.write(msg)
    log.write(best_msg)