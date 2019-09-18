from predictorWrapper import PredictorWrapper
from models.models import MultiSVM, KNN, KGSIM, CCAModel, RFModel, RandomModel, NeuNModel, GBModel, RSCCAModel, MFModel, \
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
    PLIST = [60 * i for i in range(1, 2)]

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


def runKGSIM():
    wrapper = PredictorWrapper()
    KLIST = [20 * i for i in range(1, 2)]
    for k in KLIST:
        const.KGSIM = k
        model = KGSIM()
        print(wrapper.evalAModel(model))
def runCCA():
    wrapper = PredictorWrapper()
    NCLIST = [10 * i for i in range(1, 10)]
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


def runDCN():
    from models.models import CNNModel
    wrapper = PredictorWrapper()
    model = CNNModel()
    print(wrapper.evalAModel(model))


if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-m", "--model", dest="modelName", type='string', default="KGSIM",
                      help="MODELNAME: KNN: k-nearest neighbor,\n"
                           "KGSIM: knowledge graph similairity,\n"
                            "CCA: canonical correlation analysis,\n"
                            "RF: random forest,\n"
                            "SVM: support vector machines,\n"
                            "RD: random forest,\n"
                            "GB: gradient boosting,\n"
                            "LR: logistic regression,\n"
                            "MF: matrix factorization,\n"
                            "MLN: multilayer feedforward neural network,\n"
                            "DCN: neural fingerprint model [default: %default]")
    parser.add_option("-d", "--data", dest="data", type='string', default="Liu", help="data: Liu, Aeolus [default: "
                                                                                      "%default]")
    parser.add_option("-i", "--init", dest="init", action='store_true', default=False)
    parser.add_option("-f", "--feature", dest="feature", type='int', default=0, help='feature: 0 PubChem, 1 ChemBio. '
                                                                                     '[default: %default]. '
                                                                                     'CNN is assigned to 2DChem  '
                                                                                     'descriptors. ')


    (options, args) = parser.parse_args()

    init = options.init

    if init == True:
        from dataProcessor import DataFactory

        DataFactory.genKFoldECFPLiu()
        DataFactory.genKFoldECFPAEOLUS()
        print("Generating %s-Fold data completed.\n" % const.KFOLD)
        exit(-1)

    if options.data == "Liu":
        const.CURRENT_KFOLD = const.KFOLD_FOLDER_EC_Liu
    elif options.data == "AEOLUS":
        const.CURRENT_KFOLD = const.KFOLD_FOLDER_EC_AEOLUS
    else:
        print("Fatal error: Unknown data. Only Liu and AEOLUS datasets are supported.")



    modelName = options.modelName
    const.FEATURE_MODE = options.feature

    if modelName == "KNN":
        runKNN()
    elif modelName == "KGSIM":
        runKGSIM()
    elif modelName == "CCA":
        runCCA()
    elif modelName == "RF":
        runRF()
    elif modelName == "SVM":
        runSVM()
    elif modelName == "RD":
        runRandom()
    elif modelName == "MLN":
        runNeu()
    elif modelName == "GB":
        runGB()
    # elif methodName == "SCCA":
    #     runSCCA()
    elif modelName == "MF":
        runMF()
    elif modelName == "LR":
        runLR()
    elif modelName == "DCN":
        runDCN()
    else:
        print("Method named %s is unimplemented." % modelName)
