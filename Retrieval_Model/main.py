import glob
import os
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from Retrieval_Model.model.gMLP import VisiongMLP
from Retrieval_Model.test import test_stage
from Retrieval_Model.train import train_stage
from Retrieval_Model.utils import WarmupMultiStepLR, Zhang_Dataset
from Retrieval_Model.utils import RandomIdentitySampler
from torchvision import transforms

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load cipher-image
train_dir = '../data/cipherimages/corel10k/*/'
train_list = glob.glob(os.path.join(train_dir, '*.jpg'))

# load labels
srcFiles = glob.glob('../data/cipherimages/corel10k/*')
labels_dict = {}
count = 0
class_names = []
for class_name in srcFiles:
    labels_dict[class_name.split('/')[-1].split('\\')[-1]] = count
    class_names.append(class_name.split('/')[-1].split('\\')[-1])
    count = count + 1

# 分层采样
# labels = [labels_dict[path.split('\\')[-2]] for path in train_list]
#
# # split train and test set
# seed = 2020
# train_list, valid_list = train_test_split(train_list,
#                                           test_size=0.3,
#                                           stratify=labels,
#                                           random_state=seed)

# 前70类训练
train_list_first70 = []
valid_list = []
for path in train_list:
    if path.split('\\')[-2] in class_names[:70]:
        train_list_first70.append(path)
    else:
        valid_list.append(path)
train_list = train_list_first70

print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")

train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

train_data = Zhang_Dataset(train_list, transform=train_transforms, labels_dict=labels_dict)
valid_data = Zhang_Dataset(valid_list, transform=val_transforms, labels_dict=labels_dict)

batch_size = 50
# train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
#                           sampler=RandomIdentitySampler(train_data, batch_size, 5), num_workers=0)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=0, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, )

print(len(train_data), len(train_loader))
del train_data
del valid_data

# load model
model = VisiongMLP(image_size=(192, 128), n_channels=3, patch_size=8, d_model=512,
                   d_ffn=1024, n_blocks=10, n_classes=102, )

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = WarmupMultiStepLR(optimizer=optimizer, milestones=[20, 40])

# train model
# train_stage(model, train_loader, optimizer, scheduler, epochs=20)

# valid model
test_stage(model, valid_loader)
