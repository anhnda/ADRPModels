import numpy as np
import utils
import const
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import time

from sklearn import svm
import os


class Model:
    def __init__(self):
        self.isFitAndPredict = False
        self.name = "General"
        self.repred = ""
        self.model = ""

    def fit(self, intputTrain, outputTrain):
        pass

    def predict(self, input):
        pass

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        pass

    def getInfo(self):
        pass


class RandomModel(Model):
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "Random"

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        outputs = np.random.choice([0, 1], size=(inputTest.shape[0], outputTrain.shape[1]), p=[0.5, 0.5])
        print(outputs.shape)
        self.repred = np.random.choice([0, 1], size=(outputTrain.shape[0], outputTrain.shape[1]), p=[0.5, 0.5])
        return outputs

    def getInfo(self):
        return "Uniform 0.5 0.5"


class MultiSVM(Model):

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "SVM"

    def svmModel(self, i):

        # print (os.getpid(), i)
        def checkOneClass(inp, nSize):
            s = sum(inp)
            if s == 0:
                ar = np.zeros(nSize)
            elif s == len(inp):
                ar = np.ones(nSize)
            else:
                ar = -1
            return ar

        model2 = svm.SVC(C=const.SVM_C, gamma='auto', kernel='rbf', probability=True)
        output = self.sharedOutputTrain.array[:, i]
        nTest = self.sharedInputTest.array.shape[0]
        ar = checkOneClass(output, nTest)
        ar2 = checkOneClass(output, self.sharedInputTrain.array.shape[0])

        if type(ar) == int:

            model2.fit(self.sharedInputTrain.array, output)
            output = model2.predict_proba(self.sharedInputTest.array)[:, 1]
            rep = model2.predict_proba(self.sharedInputTrain.array)[:, 1]
        else:
            output = ar
            rep = ar2

        return output, rep, i

    def __call__(self, i):
        return self.svmModel(i)

    def fitAndPredict(self, inputTrain, outputTrain, inputTest):
        from sklearn import svm

        print(inputTrain.shape, outputTrain.shape, inputTest.shape)

        def checkOneClass(inp, nSize):
            s = sum(inp)
            if s == 0:
                ar = np.zeros(nSize)
            elif s == len(inp):
                ar = np.ones(nSize)
            else:
                ar = -1
            return ar

        nClass = outputTrain.shape[1]
        outputs = []
        reps = []
        nTest = inputTest.shape[0]
        print("SVM for %s classes" % nClass)
        model = svm.SVC(C=const.SVM_C, gamma='auto', kernel='rbf', probability=True)
        self.model = model

        from models.sharedMem import SharedNDArray
        from multiprocessing import Pool

        self.sharedInputTrain = SharedNDArray.copy(inputTrain)
        self.sharedInputTest = SharedNDArray.copy(inputTest)
        self.sharedOutputTrain = SharedNDArray.copy(outputTrain)

        if const.SVM_PARALLEL:
            print ("In parallel mode")
            start = time.time()

            iters = np.arange(0, nClass)
            pool = Pool(const.N_PARALLEL)
            adrOutputs = pool.map_async(self, iters)
            pool.close()
            pool.join()

            outputs = []
            reps = []
            while not adrOutputs.ready():
                print("num left: {}".format(adrOutputs._number_left))
                adrOutputs.wait(1)

            print(adrOutputs)
            dout = dict()
            for output in adrOutputs.get():
                dout[output[2]] = output[0], output[1]

            for ii in range(len(dout)):
                out1, out2 = dout[ii]
                outputs.append(out1)
                reps.append(out2)

            end = time.time()
            print("Elapsed: ", end - start)

        else:
            print ("In sequential mode")
            start = time.time()

            for i in range(nClass):
                if i % 10 == 0:
                    print("\r%s" % i, end="")
                output = outputTrain[:, i]
                ar = checkOneClass(output, nTest)
                ar2 = checkOneClass(output, inputTrain.shape[0])

                # print clf
                if type(ar) == int:
                    model.fit(inputTrain, output)
                    output = model.predict_proba(inputTest)[:, 1]
                    rep = model.predict_proba(inputTrain)[:, 1]
                else:
                    output = ar
                    rep = ar2
                outputs.append(output)
                reps.append(rep)

            end = time.time()
            print("Elapsed: ", end - start)

        outputs = np.vstack(outputs).transpose()
        reps = np.vstack(reps).transpose()
        print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        self.repred = reps

        print(outputs.shape)
        print("\nDone")
        return outputs

    def getInfo(self):
        return self.model


class LogisticModel(Model):

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "LR"

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        from sklearn.linear_model import LogisticRegression

        print(intpuTrain.shape, outputTrain.shape, inputTest.shape)

        def checkOneClass(inp, nSize):
            s = sum(inp)
            if s == 0:
                ar = np.zeros(nSize)
            elif s == len(inp):
                ar = np.ones(nSize)
            else:
                ar = -1
            return ar

        nClass = outputTrain.shape[1]
        outputs = []
        reps = []
        nTest = inputTest.shape[0]
        print("LR for %s classes" % nClass)
        model = LogisticRegression(C=const.SVM_C, solver='lbfgs')
        self.model = model
        for i in range(nClass):
            if i % 10 == 0:
                print("\r%s" % i, end="")
            output = outputTrain[:, i]
            ar = checkOneClass(output, nTest)
            ar2 = checkOneClass(output, intpuTrain.shape[0])

            # print clf
            if type(ar) == int:

                model.fit(intpuTrain, output)
                output = model.predict_proba(inputTest)[:, 1]
                rep = model.predict_proba(intpuTrain)[:, 1]
            else:
                output = ar
                rep = ar2
            outputs.append(output)
            reps.append(rep)

        outputs = np.vstack(outputs).transpose()
        reps = np.vstack(reps).transpose()
        print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        self.repred = reps

        print(outputs.shape)
        print("\nDone")
        return outputs

    def getInfo(self):
        return self.model


class RFModel(Model):

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "RF"

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        from sklearn.ensemble import RandomForestClassifier
        print(intpuTrain.shape, outputTrain.shape, inputTest.shape)

        def checkOneClass(inp, nSize):
            s = sum(inp)
            if s == 0:
                ar = np.zeros(nSize)
            elif s == len(inp):
                ar = np.ones(nSize)
            else:
                ar = -1
            return ar

        nClass = outputTrain.shape[1]
        predicts = []
        nTest = inputTest.shape[0]
        print("RF for %s classes" % nClass)
        cc = 0
        reps = []
        model = RandomForestClassifier(n_estimators=const.RF)
        self.model = model
        for i in range(nClass):
            if i % 10 == 0:
                print("\r%s" % i, end="")
            output = outputTrain[:, i]
            ar = checkOneClass(output, nTest)
            ar2 = checkOneClass(output, intpuTrain.shape[0])

            # print model
            if type(ar) == int:

                model.fit(intpuTrain, output)
                pred = model.predict_proba(inputTest)[:, 1]
                rep = model.predict_proba(intpuTrain)[:, 1]

            else:
                pred = ar
                rep = ar2
                cc += 1
            predicts.append(pred)
            reps.append(rep)

        outputs = np.vstack(predicts).transpose()
        reps = np.vstack(reps).transpose()
        print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        self.repred = reps
        print("\nDone. Null cls: %s" % cc)
        print(outputs.shape)

        return outputs

    def getInfo(self):
        return self.model


class GBModel(Model):

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "GB"

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        from sklearn.ensemble import GradientBoostingClassifier
        print(intpuTrain.shape, outputTrain.shape, inputTest.shape)

        def checkOneClass(inp, nSize):
            s = sum(inp)
            if s == 0:
                ar = np.zeros(nSize)
            elif s == len(inp):
                ar = np.ones(nSize)
            else:
                ar = -1
            return ar

        nClass = outputTrain.shape[1]
        predicts = []
        nTest = inputTest.shape[0]
        print("Gradient Boosting Classifier for %s classes" % nClass)
        cc = 0
        reps = []
        clf = GradientBoostingClassifier(n_estimators=const.RF)
        self.model = clf
        for i in range(nClass):
            if i % 10 == 0:
                print("\r%s" % i, end="")
            output = outputTrain[:, i]
            ar = checkOneClass(output, nTest)
            ar2 = checkOneClass(output, intpuTrain.shape[0])

            # print clf
            if type(ar) == int:

                clf.fit(intpuTrain, output)
                pred = clf.predict_proba(inputTest)[:, 1]
                rep = clf.predict_proba(intpuTrain)[:, 1]

            else:
                pred = ar
                rep = ar2
                cc += 1
            predicts.append(pred)
            reps.append(rep)

        outputs = np.vstack(predicts).transpose()
        reps = np.vstack(reps).transpose()
        print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        self.repred = reps
        print("\nDone. Null cls: %s" % cc)
        print(outputs.shape)

        return outputs

    def getInfo(self):
        return self.model


class KNN(Model):
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "KNN"

    def fitAndPredict(self, inputTrain, outputTrain, inputTest):
        print(inputTrain.shape, outputTrain.shape, inputTest.shape)

        nTrain, nTest = inputTrain.shape[0], inputTest.shape[0]
        outSize = outputTrain.shape[1]
        simMatrix = np.ndarray((nTest, nTrain), dtype=float)
        for i in range(nTest):
            for j in range(nTrain):
                simMatrix[i][j] = utils.getTanimotoScore(inputTest[i], inputTrain[j])

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        args = args[:, :const.KNN]
        # print args

        outputs = []
        for i in range(nTest):
            out = np.zeros(outSize, dtype=float)
            matches = args[i]
            simScores = simMatrix[i, matches]
            ic = -1
            sum = 1e-10
            for j in matches:
                ic += 1
                out += simScores[ic] * outputTrain[j]
                sum += simScores[ic]
            out /= sum
            outputs.append(out)
        outputs = np.vstack(outputs)

        return outputs

    def getInfo(self):
        return "KNN %s" % const.KNN


class KGSIM(Model):
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "KGSIM"

    def fitAndPredict(self, inputTrain, outputTrain, inputTest):
        print(len(inputTrain), outputTrain.shape, len(inputTest))

        nTrain, nTest = len(inputTrain), len(inputTest)
        outSize = outputTrain.shape[1]
        simMatrix = np.ndarray((nTest, nTrain), dtype=float)
        for i in range(nTest):
            for j in range(nTrain):
                simMatrix[i][j] = utils.get3WJaccardOnSets(inputTest[i], inputTrain[j])

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        args = args[:, :const.KGSIM]
        # print args

        outputs = []
        for i in range(nTest):
            out = np.zeros(outSize, dtype=float)
            matches = args[i]
            simScores = simMatrix[i, matches]
            ic = -1
            sum = 1e-10
            for j in matches:
                ic += 1
                out += simScores[ic] * outputTrain[j]
                sum += simScores[ic]
            out /= sum
            outputs.append(out)
        outputs = np.vstack(outputs)

        return outputs

    def getInfo(self):
        return "KGSIM %s" % const.KNN


class MFModel(Model):
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "MF"

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        from sklearn.decomposition import NMF

        self.model = NMF(const.N_FEATURE)
        chemFeatures = self.model.fit_transform(outputTrain)
        adrFeatures = self.model.components_

        nTrain, nTest = intpuTrain.shape[0], inputTest.shape[0]
        outSize = outputTrain.shape[1]
        simMatrix = np.ndarray((nTest, nTrain), dtype=float)
        for i in range(nTest):
            for j in range(nTrain):
                simMatrix[i][j] = utils.getTanimotoScore(inputTest[i], intpuTrain[j])

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        args = args[:, :const.KNN]
        # print args

        testFeatures = []
        for i in range(nTest):
            newF = np.zeros(const.N_FEATURE, dtype=float)
            matches = args[i]
            simScores = simMatrix[i, matches]
            ic = -1
            sum = 1e-10
            for j in matches:
                ic += 1
                newF += simScores[ic] * chemFeatures[j]
                sum += simScores[ic]
            newF /= sum
            testFeatures.append(newF)
        testVecs = np.vstack(testFeatures)
        self.repred = np.matmul(chemFeatures, adrFeatures)
        out = np.matmul(testVecs, adrFeatures)
        return out

    def getInfo(self):
        return "MF %s" % const.N_FEATURE


class CCAModel(Model):
    def __init__(self):
        self.isFitAndPredict = False
        from sklearn.cross_decomposition import CCA
        self.model = CCA(n_components=const.CCA)
        self.name = "CCA"

    def fit(self, inputTrain, outputTrain):
        self.model.fit(inputTrain, outputTrain)
        print(self.model.x_weights_.shape)
        print(self.model.y_weights_.shape)
        print(np.sum(np.multiply(self.model.x_weights_, self.model.x_weights_), axis=0))

        def calCanonicalCoefficent():
            px = np.matmul(inputTrain, self.model.x_weights_)
            py = np.matmul(outputTrain, self.model.y_weights_)
            spxy = np.multiply(px, py)
            spxy = np.sum(spxy, axis=0)
            s1 = np.multiply(px, px)
            s1 = np.sum(s1, axis=0)
            s1 = np.sqrt(s1)
            s2 = np.multiply(py, py)
            s2 = np.sum(s2, axis=0)
            s2 = np.sqrt(s2)
            s = np.multiply(s1, s2)
            corr = np.divide(spxy, s)
            return np.diag(corr)

        self.corrmx = calCanonicalCoefficent()

        def eval():
            px = np.matmul(inputTrain, self.model.x_loadings_)
            py = np.matmul(outputTrain, self.model.y_loadings_)
            x = px - py
            x = np.multiply(x, x)
            print(x.shape)
            s = np.sum(x)
            print(s)

        #eval()
        y = self.predict(inputTrain)
        self.repred = y
        print("In Train: ", roc_auc_score(outputTrain.reshape(-1), y.reshape(-1)))

    def getInfo(self):
        return self.model

    def predict(self, input):
        # y = np.matmul(input,self.model.x_loadings_)
        # b = pinv(self.model.y_loadings_)
        # y = np.matmul(y,b)

        # y = np.matmul(input,self.model.x_weights_)
        # y = np.matmul(y,self.corrmx)
        # y = np.matmul(y,self.model.y_weights_.transpose())
        # print y.shape
        v = self.model.predict(input)
        # print v.shape
        return v


class RSCCAModel(Model):
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "RSCCA"

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, ifold=0):
        u = np.loadtxt("%s/u_%s" % (const.SCCA_RE_DIR, ifold))
        v = np.loadtxt("%s/v_%s" % (const.SCCA_RE_DIR, ifold))
        print(u.shape, v.shape)
        from numpy.linalg import pinv

        v = pinv(v)

        def predict(input):
            y = np.matmul(input, u)
            y = np.matmul(y, v)
            return y

        self.repred = predict(inputTrain)
        out = predict(inputTest)
        print(self.repred.shape)

        print(out.shape)
        return out


class CNNModel(Model):
    def __init__(self):
        self.isFitAndPredict = False
        self.name = "CNN"

    def fitAndPredict(self, iFold):
        from dataProcessor.DataFactory import GenAllData
        from models.cnnCore import CNNCore
        import random

        dataLoader = GenAllData()
        datas = dataLoader.loadFold(iFold)
        self.model = CNNCore(dataLoader.N_FEATURE, 5, dataLoader.N_ADRS)
        # self.loss = MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        N_TRAIN_DRUG = len(datas[const.EC_TRAIN_INP_DATA_INDEX])
        # orderTrain = np.arange(0,N_TRAIN_DRUG)
        self.currentTranIdx = 0

        self.TRAIN_ORDER = np.arange(0, N_TRAIN_DRUG)

        random.seed(1)
        random.shuffle(self.TRAIN_ORDER)

        def getSubList(ls, ids):
            subList = []
            for i in ids:
                subList.append(ls[i])
            return subList

        def getNextMiniBatchTrain():

            batchSize = 1  # Dont change this value.
            start = self.currentTranIdx
            end = self.currentTranIdx + batchSize
            if end <= N_TRAIN_DRUG:
                features = getSubList(datas[const.EC_TRAIN_INP_DATA_INDEX], self.TRAIN_ORDER[start:end])
                features = dataLoader.paddingECEPFeatureToNumpyArray(features)
                self.currentTranIdx = end
                outputs = datas[3][self.TRAIN_ORDER[start:end]]
            else:

                end = N_TRAIN_DRUG
                parts1Features = getSubList(datas[const.EC_TRAIN_INP_DATA_INDEX], self.TRAIN_ORDER[start:end])
                parts1Out = datas[const.EC_TRAIN_OUT_DATA_INDEX][self.TRAIN_ORDER[start:end]]

                end = batchSize - (end - start)
                start = 0

                random.shuffle(self.TRAIN_ORDER)
                parts2Features = getSubList(datas[const.EC_TRAIN_INP_DATA_INDEX], self.TRAIN_ORDER[start:end])
                parts2Out = datas[const.EC_TRAIN_OUT_DATA_INDEX][self.TRAIN_ORDER[start:end]]

                features = []
                for f in parts1Features:
                    features.append(f)
                for f in parts2Features:
                    features.append(f)
                features = dataLoader.paddingECEPFeatureToNumpyArray(features)
                outputs = np.vstack([parts1Out, parts2Out])

            self.currentTranIdx = end

            return features, outputs

        for iter in range(const.CNN_MAX_ITER):
            input, output = getNextMiniBatchTrain()
            inputx = torch.unsqueeze(torch.from_numpy(input).float(), dim=1)
            outputx = torch.from_numpy(output).float()

            optimizer.zero_grad()
            out, z = self.model.forward(inputx)
            # err = self.loss(out, outputx)
            err = self.model.getLoss(out, outputx, z, N_TRAIN_DRUG)

            err.backward()
            optimizer.step()

            if iter % 1000 == 0 and iter > 0:
                print("Train shape", inputx.shape)

                # print(err)
                torch.no_grad()
                # Testing...
                outs = []
                print("Test shape", len(datas[const.EC_TEST_INP_DATA_INDEX]))

                for featureTest in datas[const.EC_TEST_INP_DATA_INDEX]:
                    featureTest = dataLoader.paddingECEPFeatureToNumpyArray([featureTest])
                    featureTest = torch.unsqueeze(torch.from_numpy(featureTest).float(), dim=1)
                    out, _ = self.model.forward(featureTest)
                    out = out.detach().numpy()[0]
                    outs.append(out)
                outs = np.vstack(outs)

                print("Eval: ", iter, roc_auc_score(datas[const.EC_TEST_OUT_DATA_INDEX].reshape(-1), outs.reshape(-1)),
                      average_precision_score(datas[const.EC_TEST_OUT_DATA_INDEX].reshape(-1), outs.reshape(-1)))

                torch.enable_grad()
        # Repredict training data
        outTrains = []
        for featureTrain in datas[const.EC_TRAIN_INP_DATA_INDEX]:
            featureTrain = dataLoader.paddingECEPFeatureToNumpyArray([featureTrain])
            featureTrain = torch.unsqueeze(torch.from_numpy(featureTrain).float(), dim=1)
            out, _ = self.model.forward(featureTrain)
            out = out.detach().numpy()[0]
            outTrains.append(out)
        outTrains = np.vstack(outTrains)
        self.repred = outTrains
        return outs


class NeuNModel(Model):

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "NeuN"

    def fitAndPredict(self, input, output, inputTest, outputTest=None):
        from torch.nn.modules.loss import MSELoss
        nInput, dimInput = input.shape
        nOutput, dimOutput = output.shape
        modules = []
        # modules.append(nn.Linear(dimInput, const.NeuN_H1))
        # modules.append(nn.ReLU())
        # modules.append(nn.Linear(const.NeuN_H1, dimOutput))

        modules.append(nn.Linear(dimInput, const.NeuN_H1))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(const.NeuN_H1, const.NeuN_H2))
        modules.append(nn.Sigmoid())
        modules.append(nn.Linear(const.NeuN_H2, dimOutput))

        modules.append(nn.Sigmoid())

        #modules.append(nn.Softmax(dim=-1))

        self.model = nn.Sequential(*modules)
        self.loss = MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=const.LEARNING_RATE)
        outs = []
        inputx = torch.from_numpy(input).float()
        outputx = torch.from_numpy(output).float()
        inputTestx = torch.from_numpy(inputTest).float()
        for i in range(const.NeuIter):
            optimizer.zero_grad()
            out = self.model.forward(inputx)
            err = self.loss(out, outputx)
            err.backward()
            # print err.data
            optimizer.step()
            if i % 10 == 0:
                out2 = self.model.forward(inputTestx)
                out2 = out2.detach().numpy()
                outs.append(out2)
                out = out.detach().numpy()
                print("In Train: ", roc_auc_score(output.reshape(-1), out.reshape(-1)))
                if outputTest is not None:
                    print("Eval: ", roc_auc_score(outputTest.reshape(-1), out2.reshape(-1)))

        outx = self.model.forward(inputx)
        outx = outx.detach().numpy()
        # print "In Train: ",roc_auc_score(output.reshape(-1),outx.reshape(-1))

        self.repred = outx

        return outs

    def getInfo(self):
        return self.model
