from numpy import meshgrid
from regex import P
from dataset.dataset_processing import DatasetProcessing_image_mask_OpenCV
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import os
import torchvision.utils as vutils
import segmentation_models_pytorch as smp

from albumentations import Compose, RandomRotate90, Flip, Transpose, \
    OneOf, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, \
        Blur, ShiftScaleRotate, OpticalDistortion, GridDistortion, IAAPiecewiseAffine, \
            CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, \
                HueSaturationValue, Resize

from utils.utils import Logger, AverageMeter
import cv2


class Config(object):
    cross_validation_index = ["4"]
    # cross_validation_index = ["0", "1", "2", "3", "4"]
    def __init__(self, cross_val_index=cross_validation_index[0]):
        self.gpu_id = "0"
        self.image_path = "../dataset/ACNE04/JPEGImages/"
        self.mask_path = "../dataset/ACNE04/mask/"
        self.train_mapping_path = '../dataset/ACNE04/NNEW_trainval_%s.txt' % cross_val_index
        self.test_mapping_path = '../dataset/ACNE04/NNEW_test_%s.txt' % cross_val_index
        self.model_save_path = "results/model_state_dict/segmentation/model_%s.pkl" % cross_val_index
        self.precit_save_path = "./results/mask_predict/%s/"%cross_val_index

        self.network_input_size = (224, 224)
        self.batch_size = 1
        self.ecpoch = 5000
        self.learning_rate = 1e-4

        self.num_workers = self.batch_size


def data_augmentadion(p=0.5, train=True):
    if train:
        T =  Compose([
            Resize(height=224, width=224, p=1),
            RandomRotate90(p=p),
            Flip(p=p),
            Transpose(p=p),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2*p),
            OneOf([
                MotionBlur(p=0.2*p),
                MedianBlur(blur_limit=3, p=0.1*P),
                Blur(blur_limit=3, p=0.1*p),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2*p),
            OneOf([
                OpticalDistortion(p=0.3*p),
                GridDistortion(p=0.1*p),
                IAAPiecewiseAffine(p=0.3*p),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3*p),
            HueSaturationValue(p=0.3*p),
        ], p=1) 
    else:
        T = Compose([
                Resize(height=224, width=224, p=1),
        ])
    return T



config = Config()

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

dset_train = DatasetProcessing_image_mask_OpenCV(
    config.image_path, config.mask_path, config.train_mapping_path,
    transform=data_augmentadion(train=False))



train_loader = DataLoader(dset_train,
                         batch_size=config.batch_size,
                         shuffle=False,
                         pin_memory=True,
                         num_workers=config.num_workers,
                         drop_last=False)

dset_test = DatasetProcessing_image_mask_OpenCV(
    config.image_path, config.mask_path, config.test_mapping_path,
    transform=data_augmentadion(train=False))



test_loader = DataLoader(dset_test,
                         batch_size=config.batch_size,
                         shuffle=False,
                         pin_memory=True,
                         num_workers=config.num_workers,
                         drop_last=False)


model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
model = model.cuda()
model.load_state_dict(torch.load(config.model_save_path))




visualize = []
model.eval()


with torch.no_grad():
    for step, (img_name, img, mask, severity, lesion_numer) in enumerate(train_loader):
        img = img.cuda()
        mask = mask.cuda()
        pre_mask = model(img)
        
        print(img_name)
        vutils.save_image(pre_mask,  config.precit_save_path +"%s"%img_name[0])



with torch.no_grad():
    for step, (img_name, img, mask, severity, lesion_numer) in enumerate(test_loader):
        img = img.cuda()
        mask = mask.cuda()
        pre_mask = model(img)
        
        print(img_name)
        vutils.save_image(pre_mask,  config.precit_save_path +"%s"%img_name[0])


