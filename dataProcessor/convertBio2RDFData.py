import const
import utils

PREDICATE_SKIP_PATTERNS = ["identifier", "title", "description",
                           "drugbank-id", "x-cas", "namespace",
                           ":uri", "22-rdf-syntax-ns#type", "inDataset",
                           "rdf-schema#label", "owl#sameAs", "rdf-schema#seeAlso"]

MIN_FEATURE_COUNT = 3
def int2StringArray(intArr):
    strArr = []
    for i in intArr:
        strArr.append("%s" % i)
    return strArr


def exportBio2RDFFeature():
    fin = open(const.BIO2RDF_DRUG_TRIPLE_PATH)
    featureMap = dict()
    featureCount = dict()

    dDrug2Bio2RDFFeature = dict()
    currentDrug = ""
    currentBio2RDFFeature = []
    while True:
        line = fin.readline()
        if line == "":
            #fout.write("%s|%s\n" % (currentDrug, ",".join(int2StringArray(currentBio2RDFFeature))))
            dDrug2Bio2RDFFeature[currentDrug] = currentBio2RDFFeature
            break
        parts = line.strip().split("\t")
        if len(parts) != 3:
            print("Error")
            print(line)
            exit(-1)
        drugId = parts[0]
        if drugId != currentDrug:
            if currentDrug != "":
                #fout.write("%s|%s\n" % (currentDrug, ",".join(int2StringArray(currentBio2RDFFeature))))
                dDrug2Bio2RDFFeature[currentDrug] = currentBio2RDFFeature

            currentDrug = drugId
            currentBio2RDFFeature = []
        predicate = parts[1]
        obj = parts[2]

        isSkipped = False
        for skipPattern in PREDICATE_SKIP_PATTERNS:
            if predicate.__contains__(skipPattern):
                isSkipped = True
                break

        if isSkipped:
            continue

        feature = "%s|%s" % (predicate, obj)
        featureId = utils.get_update_dict_index(featureMap, feature)
        utils.add_dict_counter(featureCount,featureId)
        currentBio2RDFFeature.append(featureId)

    fin.close()

    #sorted = utils.sort_dict(featureCount)
    #print (sorted[-10:])





    newFeatureMap = dict()
    for featureId, cout in featureCount.items():
        if cout < MIN_FEATURE_COUNT:
            continue
        utils.get_update_dict_index(newFeatureMap,featureId)
    print("After filtering: ", len(newFeatureMap))

    fout = open(const.BIO2RDF_FEATURE_PATH, "w")

    for drugId, features in dDrug2Bio2RDFFeature.items():
        newFeatureAr = []
        for feature in features:
            newFeatureId = utils.get_dict(newFeatureMap,feature,-1)
            if newFeatureId != -1:
                newFeatureAr.append(newFeatureId)

        strArr = int2StringArray(newFeatureAr)
        fout.write("%s|%s\n" % (drugId, ",".join(strArr)))
    fout.close()


    fout = open("%s_Feature" % const.BIO2RDF_FEATURE_PATH, "w")

    revertNewFeatureMap = utils.reverse_dict(newFeatureMap)
    revertOldFeatuerMap = utils.reverse_dict(featureMap)
    for newFeatureMapId, oldFeatureMapId in revertNewFeatureMap.items():
        fout.write("%s|%s\n" % (newFeatureMapId, revertOldFeatuerMap[oldFeatureMapId]))
    fout.close()
    fout = open(const.BIO2RDF_INFO,"w")
    fout.write("Num feature: %s\n" % len(newFeatureMap))
    fout.close()

def loadAllBio2RDFFeature():
    fin = open(const.BIO2RDF_FEATURE_PATH)
    drug2BioRDFFeature = dict()
    while True:
        line = fin.readline()
        if line == "":
            break
        line = line.strip()
        parts = line.split("|")
        drug2BioRDFFeature[parts[0]] = parts[1]
    fin.close()
    return drug2BioRDFFeature


def exportLiuBio2RDFFeature():
    allDrug2BioRDFFeature = loadAllBio2RDFFeature()
    fin = open(const.LIU_ADR_PATH)
    fout = open(const.LIU_BIO2RDF_PATH, "w")
    while True:
        line = fin.readline()
        if line == "":
            break

        drugId = line.strip().split("|")[0]
        feature = utils.get_dict(allDrug2BioRDFFeature,drugId,"")
        fout.write("%s|%s\n" % (drugId,feature))
    fin.close()
    fout.close()


def exportAEOLUSBio2RDFFeature():
    allDrug2BioRDFFeature = loadAllBio2RDFFeature()
    fin = open(const.AEOLUS_ADR_PATH)
    fout = open(const.AEOLUS_BIO2RDF_PATH, "w")
    while True:
        line = fin.readline()
        if line == "":
            break

        drugId = line.strip().split("|")[0]
        feature = utils.get_dict(allDrug2BioRDFFeature,drugId,"")
        fout.write("%s|%s\n" % (drugId,feature))
    fin.close()
    fout.close()


if __name__ == "__main__":
    exportBio2RDFFeature()
    exportLiuBio2RDFFeature()
    exportAEOLUSBio2RDFFeature()
