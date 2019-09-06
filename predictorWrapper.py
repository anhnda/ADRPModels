#from DataFactory import DataLoader, DataLoader2
import const
from sklearn.metrics import roc_auc_score, auc, roc_curve, average_precision_score
import numpy as np


class PredictorWrapper():

    def __init__(self, model=None):
        # self.dataLoader = DataLoader()
        # self.loader2 = DataLoader2()
        pass

    def __getMeanSE(self, ar):
        mean = np.mean(ar)
        se = np.std(ar) / np.sqrt(len(ar))
        return mean, se

    def evalAModel(self, model):
        # print model.getInfo()
        from DataFactory import GenECFPData

        from logger.logger2 import MyLogger

        logger = MyLogger("results/logs_%s.dat" % model.name)
        logger.infoAll("K-Fold data folder: %s" % const.CURRENT_KFOLD)
        logger.infoAll("Model: %s" % model.name)
        logger.infoAll("Format: AUC STDERR AUPR STDERR")
        dataLoader = GenECFPData()


        arAuc = []
        arAupr = []
        trainAucs = []
        trainAuprs = []

        for i in range(const.KFOLD):
            datas = dataLoader.loadFold(i)
            trainInp, trainOut, testInp, testOut = datas[1], datas[2], datas[4], datas[5]
            print (trainInp.shape, trainOut.shape, testInp.shape, testOut.shape)
            if model.name == "CNN":
                predictedValues = model.fitAndPredict(i)
            else:
                if model.isFitAndPredict:
                    if model.name == "NeuN":
                        predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut)
                    elif model.name == 'SCCA':
                        predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, i)

                    else:
                        predictedValues = model.fitAndPredict(trainInp, trainOut, testInp)
                else:
                    model.fit(trainInp, trainOut)
                    predictedValues = model.predict(testInp)
                # aucs = auc(testOut, predictedValues)
                # auprs = average_precision_score(testOut, predictedValues)
                if model.name == "NeuN":
                    predictedValues = predictedValues[-1]

            aucs = roc_auc_score(testOut.reshape(-1), predictedValues.reshape(-1))
            auprs = average_precision_score(testOut.reshape(-1), predictedValues.reshape(-1))
            if model.name == "KNN":
                model.repred = model.fitAndPredict(trainInp, trainOut, trainInp)

            trainAUC = roc_auc_score(trainOut.reshape(-1), model.repred.reshape(-1))
            trainAUPR = average_precision_score(trainOut.reshape(-1), model.repred.reshape(-1))
            trainAucs.append(trainAUC)
            trainAuprs.append(trainAUPR)

            print(aucs, auprs)
            arAuc.append(aucs)
            arAupr.append(auprs)
        meanAuc, seAuc = self.__getMeanSE(arAuc)
        meanAupr, seAupr = self.__getMeanSE(arAupr)

        meanTrainAUC, seTranAUC = self.__getMeanSE(trainAucs)
        meanTranAUPR, seTrainAUPR = self.__getMeanSE(trainAuprs)


        # logger.infoAll(model.name)
        logger.infoAll(model.getInfo())
        logger.infoAll((trainInp.shape, testOut.shape))
        logger.infoAll("Test : %s %s %s %s" % (meanAuc, seAuc, meanAupr, seAupr))
        logger.infoAll("Train: %s %s %s %s" % (meanTrainAUC, seTranAUC, meanTranAUPR, seTrainAUPR))

        return meanAuc, seAuc, meanAupr, seAupr
