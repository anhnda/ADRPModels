# from DataFactory import DataLoader, DataLoader2
import const
from sklearn.metrics import roc_auc_score, average_precision_score
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
        from dataProcessor import DataFactory
        from dataProcessor.DataFactory import GenAllData

        from logger.logger2 import MyLogger

        logger = MyLogger("results/logs_%s.dat" % model.name)
        logger.infoAll("K-Fold data folder: %s" % const.CURRENT_KFOLD)
        logger.infoAll("Model: %s" % model.name)
        logger.infoAll("Format: AUC STDERR AUPR STDERR")
        dataLoader = GenAllData()

        arAuc = []
        arAupr = []
        trainAucs = []
        trainAuprs = []

        for i in range(const.KFOLD):
            datas = dataLoader.loadFold(i)
            trainInp, trainKGInp, trainOut, testInp, testKGInp, testOut = datas[1], datas[2], datas[3], datas[5], datas[
                6], datas[7]

            if const.FEATURE_MODE == const.BIO2RDF_FEATURE:
                trainInp = DataFactory.convertBioRDFSet2Array(trainKGInp)
                testInp = DataFactory.convertBioRDFSet2Array(testKGInp)

            print(trainInp.shape, trainOut.shape, testInp.shape, testOut.shape)
            if model.name == "CNN":
                predictedValues = model.fitAndPredict(i)
            elif model.name == "KGSIM":
                if const.FEATURE_MODE != const.BIO2RDF_FEATURE:
                    logger.infoAll("Fatal error: KGSIM only runs with const.FEATURE_MODE == const.BIO2RDF_FEATURE. "
                                   "Current mode is const.CHEM_FEATURE.")
                    exit(-1)
                predictedValues = model.fitAndPredict(trainKGInp, trainOut, testKGInp)
                model.repred = model.fitAndPredict(trainKGInp, trainOut, trainKGInp)
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
