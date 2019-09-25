def obtainSetDrugId(path,idInd=0):
    fin = open(path)
    ids = set()
    while True:
        line = fin.readline()
        if line == "":
            break
        ids.add(line.split("|")[idInd])
    fin.close()
    return ids

def compare():
    path1 = "./AEOLUS_Data/AEOLUS_FinalDrugADR.tsv"
    path11 = "/media/anhnd/Storage/DocumentRoot/drugCidInfo.dat"
    path2 = "./Liu_Data/LiuBioRDFFeature.dat"

    ids1 = obtainSetDrugId(path11,1)
    ids2 = obtainSetDrugId(path2)
    matchC = 0
    for id in ids1:
        if id in ids2:
            matchC += 1

    print (matchC, len(ids1), len (ids2), (matchC * 1.0 / (len(ids1) + len(ids2) - matchC)))
    print (matchC * 1.0/ len(ids2))

if __name__ == "__main__":
    compare()
