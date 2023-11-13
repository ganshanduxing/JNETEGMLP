from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def test_stage(load_model, valid_loader):
    model_path = '../data/model/gMLP.pth'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    new_checkpoint = OrderedDict()
    for key, values in checkpoint.items():
        name = key[7:]  # remove "module."
        new_checkpoint[name] = values
    load_model.load_state_dict(new_checkpoint)
    load_model.eval().cuda()
    load_model.eval()
    feas = torch.zeros([30000, 128])
    labels = torch.zeros([3000])
    count = 0
    with torch.no_grad():
        for data, label in tqdm(valid_loader):
            data = data.cuda()
            label = label.cuda()
            fea, _ = load_model(data)
            # feas.append(fea)
            # labels.append(label)
            feas[count:count+data.shape[0]] = fea
            labels[count:count+data.shape[0]] = label
            count = count + data.shape[0]
        # feas = torch.stack(feas).reshape((-1, 128))
        # labels = torch.stack(labels).reshape(-1, 1)
        average_precision_li = []
        for idx in tqdm(range(len(labels))):
            query = feas[idx].expand(feas.shape)
            label = labels[idx]
            sim = F.cosine_similarity(feas, query)
            _, indices = torch.topk(sim, 100)
            match_list = labels[indices] == label
            pos_num = 0
            total_num = 0
            precision_li = []
            for item in match_list[1:]:
                if item == 1:
                    pos_num += 1
                    total_num += 1
                    precision_li.append(pos_num / float(total_num))
                else:
                    total_num += 1
            if not precision_li:
                average_precision_li.append(0)
            else:
                average_precision = np.mean(precision_li)
                average_precision_li.append(average_precision)
        mAP = np.mean(average_precision_li)
    print(f'test mAP: {mAP}')
