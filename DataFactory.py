from io import open
import numpy as np
import const
import random

import utils


class DataLoader2:
    def loadFold(self, ifold):
        inputTrainP = np.loadtxt("%s/inputTrainP_%s" % (const.DATA_ROOT_2, ifold))
        outputTrainP = np.loadtxt("%s/outputTrainP_%s" % (const.DATA_ROOT_2, ifold))
        inputTesto = np.loadtxt("%s/inputTestP_%s" % (const.DATA_ROOT_2, ifold))
        outputTest = np.loadtxt("%s/outputTestP_%s" % (const.DATA_ROOT_2, ifold))
        return inputTrainP, outputTrainP, inputTesto, outputTest


class DataLoader:
    def __init__(self):
        self.allChems = []
        self.allAdrs = []
        pass

    @staticmethod
    def __convertBinStringToArray(bs):
        sz = len(bs)
        ar = np.zeros(sz, dtype=int)
        for i in range(sz):
            if bs[i] == "1":
                ar[i] = 1
        return ar

    def loadLiuData(self):
        f = open(const.DATA_PATH)
        while True:
            line = f.readline()
            if line == "":
                break
            line = line.strip()
            parts = line.split("|")
            chem = self.__convertBinStringToArray(parts[4])
            adr = self.__convertBinStringToArray(parts[3])
            self.allAdrs.append(adr)
            self.allChems.append(chem)

        f.close()

        ndArrayChem = np.vstack(self.allChems)
        ndArrayADR = np.vstack(self.allAdrs)
        self.mergedData = np.concatenate([ndArrayChem, ndArrayADR], axis=1)
        self.nCHem = ndArrayChem.shape[0]
        self.nDInput = ndArrayChem.shape[1]
        self.nDOutput = ndArrayADR.shape[1]
        if self.nDInput != const.INPUT_SIZE:
            print("Missmatch in dimensions")
            exit(-1)

    def getTrainTestPathByIFold(self, ifold):
        pTrain = "%s/%s_%s" % (const.KFOLD_FOLDER, const.TRAIN_PREFIX, ifold)

        pTest = "%s/%s_%s" % (const.KFOLD_FOLDER, const.TEST_PREFIX, ifold)
        return pTrain, pTest

    def exportKFold(self):
        self.loadLiuData()
        nChem = len(self.allChems)
        ar = np.ndarray(nChem, dtype=int)
        for i in range(nChem):
            ar[i] = i
        random.seed(1)
        random.shuffle(ar)
        self.mergedData = self.mergedData[ar]

        foldSize = nChem / const.KFOLD
        for i in range(const.KFOLD):
            pTrain, pTest = self.getTrainTestPathByIFold(i)
            arTrain = []
            arTest = []

            start = i * foldSize
            end = (i + 1) * foldSize
            if i == const.KFOLD - 1:
                end = nChem
            for jj in range(nChem):
                ar = arTrain
                if jj >= start and jj < end:
                    ar = arTest
                ar.append(self.mergedData[jj])

            arTrain = np.vstack(arTrain)
            arTest = np.vstack(arTest)
            np.savetxt(pTrain, arTrain)
            np.savetxt(pTest, arTest)

    def splitMergeMatrix(self, mx):
        inputs, outputs = mx[:, :const.INPUT_SIZE], mx[:, const.INPUT_SIZE:]
        return inputs, outputs

    def loadFold(self, iFold):
        pTrain, pTest = self.getTrainTestPathByIFold(iFold)
        matTrain = np.loadtxt(pTrain)
        matTest = np.loadtxt(pTest)
        return matTrain, matTest


class GenECFPData():
    @staticmethod
    def __convertBinStringToArray(bs):
        sz = len(bs)
        ar = np.zeros(sz, dtype=int)
        for i in range(sz):
            if bs[i] == "1":
                ar[i] = 1
        return ar

    def paddingECEPFeatureToNumpyArray(self, features):
        numDrug = len(features)
        if numDrug == 1:
            return np.asarray(features)
        ar = np.zeros([numDrug, self.MAX_ATOMS, self.N_FEATURE])
        # for i,data in enumerate(features):
        #     for j,d in enumerate(data):
        #         ar[i][j][d] = 1
        for drugIdx in range(numDrug):
            drugData = features[drugIdx]
            nAtom, nFeature = drugData.shape

            for iAtom in range(nAtom):
                for iFeature in range(nFeature):
                    ar[drugIdx][iAtom][iFeature] = drugData[iAtom][iFeature]
        return ar

    def loadECFPLiuData(self):
        ECFPFeatures = utils.load_obj(const.ECFP_FEATURE_PATH)
        fADR = open(const.ECFP_ADR_PATH)
        Chems = []
        ADRs = []
        while True:
            line = fADR.readline()
            if line == "":
                break
            line = line.strip()
            parts = line.split("|")
            chem = self.__convertBinStringToArray(parts[4])
            adr = self.__convertBinStringToArray(parts[3])

            Chems.append(chem)
            ADRs.append(adr)
        fADR.close()

        if len(ECFPFeatures) != len(Chems):
            print("Fatal error. Missmatched data")
            exit(-1)

        fin = open(const.ECFP_INFO)
        self.N_DRUGS = int(fin.readline().split(":")[-1].strip())
        self.N_FEATURE = int(fin.readline().split(":")[-1].strip())
        self.MAX_ATOMS = int(fin.readline().split(":")[-1].strip())

        self.ECFPFeatures = ECFPFeatures
        self.Chems = Chems
        self.ADRs = ADRs

    def loadECFPAEOLUSData(self):
        ECFPFeatures = utils.load_obj(const.AEOLUS_ECFP_PATH)
        fADR = open(const.AEOLUS_ADR_PATH)
        fChem = open(const.AEOLUS_CHEM_PATH)
        Chems = []
        ADRs = []
        while True:
            lineADR = fADR.readline()
            if lineADR == "":
                break
            lineADR = lineADR.strip()
            lineChem = fChem.readline().strip()

            adrString = lineADR.split("|")[-1].replace(",", "")
            chemString = lineChem.split("|")[-1].replace(",", "")[:881]
            #print(len(chemString), len(adrString))
            chem = self.__convertBinStringToArray(chemString)
            adr = self.__convertBinStringToArray(adrString)

            Chems.append(chem)
            ADRs.append(adr)
        fADR.close()
        fChem.close()
        if len(ECFPFeatures) != len(Chems):
            print("Fatal error. Missmatched data")
            exit(-1)

        fin = open(const.AEOLUS_INFO)
        self.N_DRUGS = int(fin.readline().split(":")[-1].strip())
        self.N_FEATURE = int(fin.readline().split(":")[-1].strip())
        self.MAX_ATOMS = int(fin.readline().split(":")[-1].strip())

        self.ECFPFeatures = ECFPFeatures
        self.Chems = Chems
        self.ADRs = ADRs

    def getTrainTestPathByIFold(self, ifold, root=const.KFOLD_FOLDER_EC_Liu):
        pTrainECFeature = "%s/%s_ec_%s" % (root, const.TRAIN_PREFIX_EC, ifold)
        pTrainChemFeature = "%s/%s_chem_%s" % (root, const.TRAIN_PREFIX_EC, ifold)
        pTrainADRs = "%s/%s_ADR_%s" % (root, const.TRAIN_PREFIX_EC, ifold)

        pTestECFeature = "%s/%s_ec_%s" % (root, const.TEST_PREFIX_EC, ifold)
        pTestChemFeature = "%s/%s_chem_%s" % (root, const.TEST_PREFIX_EC, ifold)
        pTestADRs = "%s/%s_ADR_%s" % (root, const.TEST_PREFIX_EC, ifold)

        return pTrainECFeature, pTrainChemFeature, pTrainADRs, pTestECFeature, pTestChemFeature, pTestADRs

    def exportKFold(self, root=const.KFOLD_FOLDER_EC_Liu):
        foldSize = self.N_DRUGS / const.KFOLD
        order = np.arange(0, self.N_DRUGS)
        random.seed(1)
        random.shuffle(order)

        for i in range(const.KFOLD):
            # pTrainECFeature, pTrainChemFeature, pTrainADRs, pTestECFeature, pTestChemFeature, pTestADRs
            paths = self.getTrainTestPathByIFold(i, root)

            arTrain = []
            arTest = []

            start = i * foldSize
            end = (i + 1) * foldSize
            if i == const.KFOLD - 1:
                end = self.N_DRUGS
            for jj in range(self.N_DRUGS):
                ar = arTrain
                if start <= jj < end:
                    ar = arTest
                ix = order[jj]
                ar.append([self.ECFPFeatures[ix], self.Chems[ix], self.ADRs[ix]])

            ars = [arTrain, arTest]

            for ii in range(2):
                for jj in range(3):
                    path = paths[ii * 3 + jj]
                    tmp = ars[ii]
                    data = []
                    for d in tmp:
                        data.append(d[jj])
                    if jj == 0:
                        utils.save_obj(data, path)
                    else:
                        data = np.vstack(data)
                        np.savetxt(path, data)

    def loadFold(self, iFold, root=const.CURRENT_KFOLD):

        print("IFOLD: %s, FOLDER: %s" % (iFold, root))
        if root == const.KFOLD_FOLDER_EC_Liu:
            fin = open(const.ECFP_INFO)
        else:
            fin = open(const.AEOLUS_INFO)


        self.N_DRUGS = int(fin.readline().split(":")[-1].strip())
        self.N_FEATURE = int(fin.readline().split(":")[-1].strip())
        self.MAX_ATOMS = int(fin.readline().split(":")[-1].strip())


        paths = self.getTrainTestPathByIFold(iFold, root)
        data = []
        for i in range(2):
            for j in range(3):
                path = paths[i * 3 + j]
                if j == 0:
                    data.append(utils.load_obj(path))
                else:
                    data.append(np.loadtxt(path))

        self.N_ADRS = data[2].shape[1]

        print(self.N_DRUGS, self.N_ADRS)

        return data


def genKFoldECFPLiu():
    data = GenECFPData()
    data.loadECFPLiuData()
    data.exportKFold(const.KFOLD_FOLDER_EC_Liu)

def genKFoldECFPAEOLUS():

    data = GenECFPData()
    data.loadECFPAEOLUSData()
    data.exportKFold(const.KFOLD_FOLDER_EC_AEOLUS)


if __name__ == "__main__":

    # Original Liu_Data
    # data = DataLoader()
    # data.exportKFold()

    # Liu_Data With ECFP
    # genKFoldECFPLiu()

    # AeolusData
    # genKFoldECFPAEOLUS()

    pass
