from torch.utils.data import DataLoader
from torch import nn, cuda, optim, max, sum, no_grad
from torch.autograd import Variable

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models

import time
import os

simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train = ImageFolder(os.path.join("dogs_vs_cats", "train"), simple_transform)
valid = ImageFolder(os.path.join("dogs_vs_cats", "valid"), simple_transform)

train_data_gen = DataLoader(train, batch_size=128, shuffle=True, num_workers=3)
valid_data_gen = DataLoader(valid, batch_size=128, num_workers=3)

dataset_sizes = {"train": len(train_data_gen.dataset), "valid": len(valid_data_gen.dataset)}
dataloaders = {"train": train_data_gen, "valid": valid_data_gen}

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

if cuda.is_available():
    model_ft = model_ft.cuda()

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(1, num_epochs+1):
        print("epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data
                if cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += sum(preds == labels.data)
            scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.cpu().numpy() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
