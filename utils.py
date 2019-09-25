from io import open
import time

import numpy as np
from sklearn.externals import joblib
import os
from datetime import datetime


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def convertHexToBinString888(hexString):
    # scale = 16  ## equals to hexadecimal
    # num_of_bits = 888
    return bin(int(hexString, 16))[2:].zfill(888)


def convertBinString888ToArray(binString888):
    ar = np.ndarray(888, dtype=float)
    ar.fill(0)
    for i in range(887, -1, -1):
        if binString888[i] == "1":
            ar[i] = 1
    return ar


def convertHex888ToArray(hex888):
    return convertBinString888ToArray(convertHexToBinString888(hex888))


def get_dict(d, k, v=-1):
    try:
        v = d[k]
    except:
        pass
    return v


def get_insert_key_dict(d, k, v=0):
    try:
        v = d[k]
    except:
        d[k] = v
    return v


def add_dict_counter(d, k, v=1):
    try:
        v0 = d[k]
    except:
        v0 = 0
    d[k] = v0 + v


def sort_dict(dd):
    kvs = []
    for key, value in sorted(dd.items(), key=lambda p: (p[1], p[0])):
        kvs.append([key, value])
    return kvs[::-1]


def sum_sort_dict_counter(dd):
    cc = 0
    for p in dd:
        cc += p[1]
    return cc


def get_update_dict_index(d, k):
    try:
        current_index = d[k]
    except:
        current_index = len(d)
        d[k] = current_index
    return current_index


def get_dict_index_only(d, k):
    try:
        current_index = d[k]
    except:
        current_index = -1

    return current_index


def load_list_from_file(path):
    list = []
    fin = open(path)
    while True:
        line = fin.readline()
        if line == "":
            break
        list.append(line.strip())
    fin.close()
    return list


def reverse_dict(d):
    d2 = dict()
    for k, v in d.items():
        d2[v] = k
    return d2


def save_obj(obj, path):
    joblib.dump(obj, path)


def load_obj(path):
    return joblib.load(path)


def loadMapFromFile(path, sep="\t", keyPos=0, valuePos=1):
    fin = open(path)
    d = dict()
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split(sep)
        d[parts[keyPos]] = parts[valuePos]
    fin.close()
    return d


def loadMapSetFromFile(path, sep="\t", keyPos=0, valuePos=1, sepValue="", isStop=""):
    fin = open(path)
    dTrain = dict()

    if isStop != "":
        dTest = dict()

    d = dTrain

    while True:
        line = fin.readline()
        if line == "":
            break
        if isStop != "":
            if line.startswith(isStop):
                d = dTest
                continue
        parts = line.strip().split(sep)
        v = get_insert_key_dict(d, parts[keyPos], set())
        if sepValue == "":
            v.add(parts[valuePos])
        else:
            values = parts[valuePos]
            values = values.split(sepValue)
            for value in values:
                v.add(value)
    fin.close()
    if isStop != "":
        return dTrain, dTest
    return dTrain


def convertEpochtoTime(t):
    dt_object = datetime.fromtimestamp(t)
    return dt_object


def getTanimotoScore(ar1, ar2):
    c1 = np.sum(ar1)
    c2 = np.sum(ar2)
    bm = np.dot(ar1, ar2)
    return bm * 1.0 / (c1 + c2 - bm + 1e-10)


def getJaccardScore(ar1, ar2):
    c1 = np.sum(ar1)
    c2 = np.sum(ar2)
    bm = np.dot(ar1, ar2)
    return bm * 1.0 / (c1 + c2 - bm + 1e-10)


def get3WJaccardOnSets(set1, set2):
    len1 = len(set1)
    len2 = len(set2)
    nMatch = 0
    for s in set1:
        if s in set2:
            nMatch += 1
    return 3.0 * nMatch / (len1 + len2 + nMatch + 0.1)


def get3WJaccardOnArray(ar1,ar2):
    c1 = np.sum(ar1)
    c2 = np.sum(ar2)
    bm = np.dot(ar1, ar2)
    return bm * 3.0 / (c1 + c2 + bm + 1e-10)

def getF2Sim(ar1,ar2):
    v = ar1 - ar2
    return np.dot(v,v)

def getSimByType(ar1,ar2,tp=0):
    if tp == 0:
        return getJaccardScore(ar1,ar2)
    elif tp == 1:
        return get3WJaccardOnArray(ar1,ar2)
    elif tp ==2:
        return getTanimotoScore(ar1,ar2)
    else:
        print("Error: Known type %s" % tp)
        return 0