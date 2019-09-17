import torch
import torch.nn as nn
import const


class CNNCore(nn.Module):
    def __init__(self, inputDimFeature, numFilter, outputDim):
        super(CNNCore, self).__init__()

        self.layer1 = torch.randn([numFilter, inputDimFeature, const.CH_NUM_1], requires_grad=True)
        self.bias1 = torch.zeros(1, requires_grad=True)
        self.fn1 = torch.nn.Sigmoid()
        self.attent1 = torch.randn([const.CH_NUM_1, const.FINGER_PRINT_SIZE], requires_grad=True)
        self.sm1 = torch.nn.Softmax(dim=-1)

        self.layer2 = torch.randn([numFilter, const.CH_NUM_1, const.CH_NUM_2], requires_grad=True)
        self.bias2 = torch.zeros(1, requires_grad=True)
        self.fn2 = torch.nn.Sigmoid()
        self.attent2 = torch.randn([const.CH_NUM_2, const.FINGER_PRINT_SIZE], requires_grad=True)
        self.sm2 = torch.nn.Softmax(dim=-1)

        self.layer3 = torch.randn([numFilter, const.CH_NUM_2, const.CH_NUM_3], requires_grad=True)
        self.bias3 = torch.zeros(1, requires_grad=True)
        self.fn3 = torch.nn.Sigmoid()
        self.attent3 = torch.randn([const.CH_NUM_3, const.FINGER_PRINT_SIZE], requires_grad=True)
        self.sm3 = torch.nn.Softmax(dim=-1)

        self.layer4 = torch.randn([numFilter, const.CH_NUM_3, const.CH_NUM_4], requires_grad=True)
        self.bias4 = torch.zeros(1, requires_grad=True)
        self.fn4 = torch.nn.Sigmoid()
        self.attent4 = torch.randn([const.CH_NUM_4, const.FINGER_PRINT_SIZE], requires_grad=True)
        self.sm4 = torch.nn.Softmax(dim=-1)

        self.fc = torch.nn.Linear(const.FINGER_PRINT_SIZE, outputDim, bias=True)
        self.lg = torch.nn.Sigmoid()

    def forward(self, inputs):
        rlayer1 = torch.add(torch.matmul(inputs, self.layer1), self.bias1)
        rlayer1 = self.fn1(rlayer1)
        pool1 = torch.unsqueeze(torch.sum(rlayer1, dim=1), dim=1)
        attent1 = self.sm1(torch.matmul(rlayer1, self.attent1))
        poolAt1 = torch.sum(attent1, dim=1)

        rlayer2 = torch.add(torch.matmul(pool1, self.layer2), self.bias2)
        rlayer2 = self.fn2(rlayer2)
        pool2 = torch.unsqueeze(torch.sum(rlayer2, dim=1), dim=1)
        attent2 = self.sm2(torch.matmul(rlayer2, self.attent2))
        poolAt2 = torch.sum(attent2, dim=1)

        rlayer3 = torch.add(torch.matmul(pool2, self.layer3), self.bias3)
        rlayer3 = self.fn3(rlayer3)
        pool3 = torch.unsqueeze(torch.sum(rlayer3, dim=1), dim=1)
        attent3 = self.sm3(torch.matmul(rlayer3, self.attent3))
        poolAt3 = torch.sum(attent3, dim=1)

        rlayer4 = torch.add(torch.matmul(pool3, self.layer4), self.bias4)
        rlayer4 = self.fn4(rlayer4)
        # pool4 = torch.unsqueeze(torch.sum(rlayer4,dim=1),dim=1)
        attent4 = self.sm3(torch.matmul(rlayer4, self.attent4))
        poolAt4 = torch.sum(attent4, dim=1)

        re = poolAt1 + poolAt2 + poolAt3 + poolAt4
        re = torch.sum(re, dim=1)

        out = self.fc(re)
        out = self.lg(out)
        return out, re

    def __getF2Err(self, err):
        err = torch.mul(err, err)
        err = torch.sum(err)
        # err =  self.mseLoss(err,self.zeroTargets)
        return err

    def getLoss(self, out, target, z, n):

        loss = 0
        err = out - target
        err = self.__getF2Err(err)
        err *= n
        loss += err
        loss += const.CNN_LB_1 * self.__getF2Err(self.fc._parameters['weight']) + const.CNN_LB_2 * n * self.__getF2Err(
            z)
        return loss


if __name__ == "__main__":
    pass
