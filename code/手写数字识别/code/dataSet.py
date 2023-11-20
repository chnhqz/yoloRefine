import torch
import torchvision
from tqdm import tqdm
import matplotlib
import model

device = "cuda:0" if torch.cuda.is_available() else "cpu"

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

path = './data/'  # 数据集下载后保存的目录

# 下载训练集和测试集
trainData = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
testData = torchvision.datasets.MNIST(path, train=False, transform=transform)

# 设定每一个Batch的大小
BATCH_SIZE = 256

# 构建数据集和测试集的DataLoader
trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)

net = model.Net()

print(net.to(device))

lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

EPOCHS = 10
# 存储训练过程
history = {'Test Loss': [], 'Test Accuracy': []}
for epoch in range(1, EPOCHS + 1):
    processBar = tqdm(trainDataLoader, unit='step')
    net.train(True)
    for step, (trainImgs, labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)

        net.zero_grad()
        outputs = net(trainImgs)
        loss = lossF(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predictions == labels) / labels.shape[0]
        loss.backward()

        optimizer.step()
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                   (epoch, EPOCHS, loss.item(), accuracy.item()))

        if step == len(processBar) - 1:
            correct, totalLoss = 0, 0
            net.train(False)
            for testImgs, labels in testDataLoader:
                testImgs = testImgs.to(device)
                labels = labels.to(device)
                outputs = net(testImgs)
                loss = lossF(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)

                totalLoss += loss
                correct += torch.sum(predictions == labels)
            testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
            testLoss = totalLoss / len(testDataLoader)
            history['Test Loss'].append(testLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                       (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(),
                                        testAccuracy.item()))
    processBar.close()

# 对测试Loss进行可视化
matplotlib.pyplot.plot(history['Test Loss'], label='Test Loss')
matplotlib.pyplot.legend(loc='best')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.xlabel('Epoch')
matplotlib.pyplot.ylabel('Loss')
matplotlib.pyplot.show()

# 对测试准确率进行可视化
matplotlib.pyplot.plot(history['Test Accuracy'], color='red', label='Test Accuracy')
matplotlib.pyplot.legend(loc='best')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.xlabel('Epoch')
matplotlib.pyplot.ylabel('Accuracy')
matplotlib.pyplot.show()

torch.save(net, './model.pth')
