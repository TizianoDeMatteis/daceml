import torch
from dace.transformation.dataflow import RedundantSecondArray
from torch import nn
import torch.nn.functional as F
from daceml.pytorch import DaceModule
import numpy as np
##
# This is a mix of
# - https://github.com/maorshutman/lenet5-pytorch/blob/master/lenet5.py
# - https://www.kaggle.com/usingtc/lenet-with-pytorch
from daceml.transformation import ConstantFolding


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(1, 6, (5, 5), padding=2)
        self.s2 = nn.MaxPool2d(2, 2)
        self.c3 = nn.Conv2d(6, 16, (5, 5))
        self.s4 = nn.MaxPool2d(2, 2)
        # c5 can be implemented as fully connected
        # but must be flattened
        # self.c5 = nn.Linear(16 * 5 * 5, 120)
        self.c5 = nn.Conv2d(16, 120, 5, bias=True)

        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)
        self.activation = nn.Tanh()
        #self.drop_layer = nn.Dropout(self.dropout)

    def forward(self, x):

        # C1
        x = self.c1(x)
        x = self.activation(x)

        # S2
        print(x.shape)
        x = self.s2(x)
        # C3
        x = self.c3(x)
        x = self.activation(x)

        # S4
        x = self.s4(x)

        # C5
        x = self.c5(x)
        x = self.activation(x)
        x = x.view(x.shape[0], -1)
        # or alternatively
        # x = torch.flatten(x, 1)


        #alternative to C5
        # x = x.view(-1, self.num_flat_features(x))
        # x = self.c5(x)
        # x = self.activation(x)

        # F6, F7
        x = self.f6(x)
        x = self.activation(x)
        # x = self.drop_layer(x)
        x = self.f7(x)
        x = nn.Softmax(dim=1)(x)
        return x



        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


module = LeNet()
res = module(torch.ones(1,1,28,28))
print(res)
dace_module = DaceModule(module)


res2= dace_module(torch.ones(1,1,28,28))

dace_module.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray], validate_all=True)
dace_module.sdfg.save('/tmp/out.sdfg')

if abs( np.linalg.norm(res.data.numpy()-res2[0])/res2[0].size)< 1e-5:
    print("Result is correct")
else:
    print("Result is WRONG")
