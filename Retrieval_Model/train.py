import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from Retrieval_Model.loss.Tripet_Loss import TripletLoss


def train_stage(model, train_loader, optimizer, scheduler, epochs=20):
    model = torch.nn.DataParallel(model).cuda()
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.enabled = False

    criterion = nn.CrossEntropyLoss()
    tri_loss = TripletLoss(margin=0.3)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        #     scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch + 1, optimizer.param_groups[0]['lr']))

        for data, label in tqdm(train_loader):
            model.train()
            data = data.cuda()
            label = label.cuda()
            fea, output = model(data)
            loss = criterion(output, label) + tri_loss(fea, label)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        scheduler.step()
        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n")

    # save model
    torch.save(model.state_dict(), '../data/model/gMLP.pth')
