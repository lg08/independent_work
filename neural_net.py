import torch
import torchvision
import torchvision.transforms as transforms
import sys
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np

from database import CustomImageDataset, ToTensor, make_label_csv_files
from database import check_size

# using this example
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

#------------------------------------------------------------------------------#
#                         Set Up Neural Net We Will Use                        #
#------------------------------------------------------------------------------#
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc1 = nn.Linear(16 * 4, 120)
        self.fc1 = nn.Linear(59536, 120)
        self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --------------------------------------------------------------



#==============================================================================#
#--------------     Now creating our own custom PV_Net class     --------------#
#==============================================================================#
class PV_Net():

    def __init__(self):
        self.classes=["FALSE", "TRUE"]

    # completely initialize the datasets we will use
    def init_datasets(self, data_path, split, filename="image_array"):
        print("Initializing dataset...")
        # first check to make sure they're all the same size
        if not check_size(data_path):
            print("ERROR, dataset not all the same size")
            os.sys.exit()
        else:
            print("Dataset all the same size, proceeding...")
        # then make the labels csv file
        train_label_name, test_label_name = make_label_csv_files(data_path,
                                                                 split, filename)
        # make the trainingset
        self.trainset = CustomImageDataset(
            csv_file=train_label_name,
            root_dir=data_path,
            transform=ToTensor()
        )
        # make the trainloader
        self.trainloader = DataLoader(self.trainset, batch_size=4,
                                shuffle=True, num_workers=0)
        # make the testing set
        self.testset = CustomImageDataset(
            csv_file=test_label_name,
            root_dir=data_path,
            transform=ToTensor()
        )
        # make the testing loader
        self.testloader = DataLoader(self.testset, batch_size=4,
                                shuffle=True, num_workers=0)
        print("Finished Initializing Dataset!")


    def train(self):
        print("Training Model...")
        net = Net()
        net = net.float()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                # if i % 2000 == 1999:    # print every 2000 mini-batches
                if i % 20 == 0:    # print every 2 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        self.net = net
        print("Finished Training Model!")

    def save_net(self, path='./extras/solar_panel_nueral_net.pth'):
        print("Saving Neural Net...")
        try:
            torch.save(self.net.state_dict(), path)
            print("Neural Net Saved!")
        except:
            print("Currently no neural network configured, please train one\
                  first!")

    def load_net(self, path='./extras/solar_panel_nueral_net.pth'):
        print("Loading Neural Net...")
        self.net = Net()
        self.net = self.net.float()
        self.net.load_state_dict(torch.load(path))
        print("Neural Net Loaded!")

    def test(self):
        print("Testing Model...")
        dataiter = iter(self.testloader)
        images, labels = dataiter.next()
        print(images)
        outputs = self.net(images.float())
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % self.classes[predicted[j]]
                                      for j in range(4)))
        # plt.imshow(np.transpose(images[0], (1,2,0)), interpolation='nearest')
        # plt.show()

        fig = plt.figure()
        rows = 2
        columns = 2

        for i in range(4):
            fig.add_subplot(rows, columns, (i+1))
            # showing image
            img = np.transpose(images[i], (1,2,0))
            plt.imshow(img, interpolation='nearest')
            plt.axis('off')
            plt.title(f"pic {i}: {self.classes[predicted[i]]}")
        plt.show()

    def test_whole(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.net(images.float())
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
