import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
import os
import torch.backends.cudnn as cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
start_time = time.time()
batch_size = 128
learning_rate = 0.1
resume_file='checkpoint_linear.pth.tar'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
])


train_dataset = datasets.CIFAR10(root='data/cifar10/',
                                 train=True,
                                 transform=transform_train,
                                 download=False)

test_dataset = datasets.CIFAR10(root='data/cifar10/',
                                train=False,
                                transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=4)


class Vgg(nn.Module):
    def __init__(self, num_classes=10):
        super(Vgg, self).__init__()
        self.features = nn.Sequential(

                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(128, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    # nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.MaxPool2d(kernel_size=2, stride=2),


                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                )
        self.classifier = nn.Sequential(
            nn.Linear(2048,1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, num_classes),)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x.size()=[batch_size, channel, width, height]
        #          [128, 512, 2, 2]
        # flatten 결과 => [128, 512x2x2]
        x = self.classifier(x)
        return x


if __name__ == '__main__':

    print('done')

    model = Vgg()
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().cuda()

    if torch.cuda.device_count() > 0:
        print("USE", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        cudnn.benchmark = True
    else:
        print("USE ONLY CPU!")

    if torch.cuda.is_available():
        model.cuda()
        print(torch.__version__)
        print(torch.cuda.get_device_name(0))


    def train(epoch):
        model.train()
        train_loss = 0
        total = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = Variable(data.cuda()), Variable(target.cuda())
            else:
                data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            # torch.max() : (maximum value, index of maximum value) return.
            # 1 :  row마다 max계산 (즉, row는 10개의 class를 의미)
            # 0 : column마다 max 계산
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            if batch_idx % 30 == 0:
                print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                      .format(epoch, batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test():
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                data, target = Variable(data.cuda()), Variable(target.cuda())
            else:
                data, target = Variable(data), Variable(target)

            outputs = model(data)
            loss = criterion(outputs, target)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
        print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
          .format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def run_training(learning_rate, start_epoch=0):
        for epoch in range(start_epoch, 300):
            if epoch < 80:
                learning_rate2 = learning_rate
            elif epoch < 115:
                learning_rate2 = learning_rate * 0.1
            elif epoch < 130:
                learning_rate2 = learning_rate * 0.01
            else:
                learning_rate2 = learning_rate * 0.001
            for param_group in optimizer.param_groups:
                param_group['learning_rate'] = learning_rate2

            print('learning_rate: {}'.format(learning_rate2))
            train(epoch)
            test()

            if (epoch % 10 == 0):
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'learning_rate': learning_rate,
                    'optimizer': optimizer.state_dict(),
                }, resume_file)
                print('saved checkpoint')

        now = time.gmtime(time.time() - start_time)
        print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


    resume=False
    if resume:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            learning_rate = checkpoint['learning_rate']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))

            run_training(0.1, start_epoch)

        else:
            print("=> no checkpoint found at '{}'".format(resume_file))

    else:
        run_training(learning_rate)

    torch.save(model.state_dict(), 'main_test1.pt')
