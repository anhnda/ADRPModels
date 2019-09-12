import const
import utils

PREDICATE_SKIP_PATTERNS = ["identifier", "title", "description",
                           "drugbank-id", "x-cas", "namespace",
                           ":uri", "22-rdf-syntax-ns#type", "inDataset",
                           "rdf-schema#label", "owl#sameAs", "rdf-schema#seeAlso"]

def int2StringArray(intArr):
    strArr = []
    for i in intArr:
        strArr.append("%s"%i)
    return strArr

def exportBio2RDFFeature():
    fout = open(const.BIO2RDF_FEATURE_PATH, "w")
    fin = open(const.BIO2RDF_DRUG_TRIPLE_PATH)
    featureMap = dict()
    currentDrug = ""
    currentBio2RDFFeature = []
    while True:
        line = fin.readline()
        if line == "":
            fout.write("%s|%s\n" % (currentDrug, ",".join(int2StringArray(currentBio2RDFFeature))))
            break
        parts = line.strip().split("\t")
        if len(parts) != 3:
            print ("Error")
            print(line)
            exit(-1)
        drugId = parts[0]
        if drugId != currentDrug:
            if currentDrug != "":
                fout.write("%s|%s\n" % (currentDrug, ",".join(int2StringArray(currentBio2RDFFeature))))
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
        currentBio2RDFFeature.append(featureId)

    fin.close()
    fout.close()

    fout = open("%s_Feature"%const.BIO2RDF_FEATURE_PATH,"w")
    for k,v in featureMap.items():
        fout.write("%s|%s\n"%(k,v))
    fout.close()


if __name__ == "__main__":
    exportBio2RDFFeature()


