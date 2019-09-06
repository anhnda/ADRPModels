from predictorWrapper import PredictorWrapper
from models.models import MultiSVM, KNN, CCAModel, RFModel, RandomModel, NeuNModel, GBModel, RSCCAModel, MFModel, \
    LogisticModel
from optparse import OptionParser
import const


def runSVM():
    wrapper = PredictorWrapper()
    PLIST = [i for i in range(1, 2)]
    for p in PLIST:
        const.SVM_C = p
        model = MultiSVM()
        print(wrapper.evalAModel(model))


def runRF():
    wrapper = PredictorWrapper()
    PLIST = [10 * i for i in range(1, 2)]

    for p in PLIST:
        const.RF = p
        model = RFModel()
        print(wrapper.evalAModel(model))


def runGB():
    wrapper = PredictorWrapper()
    PLIST = [10 * i for i in range(1, 2)]

    for p in PLIST:
        const.RF = p
        model = GBModel()
        print(wrapper.evalAModel(model))


def runKNN():
    wrapper = PredictorWrapper()
    KLIST = [10 * i for i in range(1, 2)]
    for k in KLIST:
        const.KNN = k
        model = KNN()
        print(wrapper.evalAModel(model))


def runCCA():
    wrapper = PredictorWrapper()
    NCLIST = [10 * i for i in range(1, 2)]
    for c in NCLIST:
        const.CCA = c
        model = CCAModel()
        print(wrapper.evalAModel(model))


def runSCCA():
    wrapper = PredictorWrapper()
    NCLIST = [10 * i for i in range(1, 2)]
    for c in NCLIST:
        const.CCA = c
        model = RSCCAModel()
        print(wrapper.evalAModel(model))


def runRandom():
    wrapper = PredictorWrapper()
    model = RandomModel()
    print(wrapper.evalAModel(model))


def runMF():
    wrapper = PredictorWrapper()
    KLIST = [10 * i for i in range(1, 2)]
    for k in KLIST:
        const.N_FEATURE = k
        model = MFModel()
        print(wrapper.evalAModel(model))


def runNeu():
    wrapper = PredictorWrapper()
    PLIST = [10 * i for i in range(1, 2)]
    for p in PLIST:
        const.NeuN_H1 = p
        model = NeuNModel()
        print(wrapper.evalAModel(model))


def runLR():
    wrapper = PredictorWrapper()
    PLIST = [1 * i for i in range(1, 2)]
    for p in PLIST:
        const.SVM_C = p
        model = LogisticModel()
        print(wrapper.evalAModel(model))


def runCNN():
    from models.models import CNNModel
    wrapper = PredictorWrapper()
    model = CNNModel()
    print(wrapper.evalAModel(model))


if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-m", "--model", dest="modelName", type='string', default="KNN",
                      help="MODELNAME: KNN: k-nearest neighbor,\n"
                            "CCA: canonical correlation analysis,\n"
                            "RF: random forest,\n"
                            "SVM: support vector machines,\n"
                            "RD: random forest,\n"
                            "GB: gradient boosting,\n"
                            "LR: logistic regression,\n"
                            "MF: matrix factorization,\n"
                            "NN: multilayer feedforward neural network,\n"
                            "CNN: neural fingerprint model [default: %default]")
    parser.add_option("-d", "--data", dest="data", type='string', default="Liu", help="data: Liu, Aeolus [default: "
                                                                                      "%default]")
    parser.add_option("-i", "--init", dest="init", action='store_true', default=False)

    (options, args) = parser.parse_args()

    init = options.init

    if init == True:
        import DataFactory
        DataFactory.genKFoldECFPLiu()
        DataFactory.genKFoldECFPAEOLUS()

    if options.data == "Liu":
        const.CURRENT_KFOLD = const.KFOLD_FOLDER_EC_Liu
    elif options.data == "Aeolus":
        const.CURRENT_KFOLD = const.KFOLD_FOLDER_EC_AEOLUS
    else:
        print("Fatal error: Unknown data. Only Liu and AEOLUS datasets are supported.")



    modelName = options.modelName

    if modelName == "KNN":
        runKNN()
    elif modelName == "CCA":
        runCCA()
    elif modelName == "RF":
        runRF()
    elif modelName == "SVM":
        runSVM()
    elif modelName == "RD":
        runRandom()
    elif modelName == "NN":
        runNeu()
    elif modelName == "GB":
        runGB()
    # elif methodName == "SCCA":
    #     runSCCA()
    elif modelName == "MF":
        runMF()
    elif modelName == "LR":
        runLR()
    elif modelName == "CNN":
        runCNN()
    else:
        print("Method named %s is unimplemented." % modelName)
