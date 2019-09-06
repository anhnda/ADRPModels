import re

def loadData():
    inp = open("results/Result")
    data = []
    while True:
        line  = inp.readline()
        line = line.strip()
        if line == "":
            break
        method = line
        values = inp.readline().strip().split(" ")
        AUC = {}
        AUC['mean'] = values[0]
        AUC['error'] = values[1]
        AUPR = {}

        AUPR['mean'] = values[2]
        AUPR['error'] = values[3]
        it = {}
        it['name'] = method
        it['AUC'] = AUC
        it['AUPR'] = AUPR
        data.append(it)
    return data


def plotBar(data,metric):


    import matplotlib
    matplotlib.rcParams.update({'font.size': 16})
    #matplotlib.rc('xtick', labelsize=20)
    #matplotlib.rc('ytick', labelsize=20)

    import matplotlib.pyplot as plt

    MethodList = []
    Means = []
    Errors = []
    for it in data:
        MethodList.append(it['name'])
        me = it[metric]
        Means.append(float(me['mean']))
        Errors.append(float(me['error']))
    import math
    import numpy as np
    v = np.asarray(Errors,dtype=float)
    v *= math.sqrt(5)
    print (v)



    #plt.scatter(x=Methods,y=values,c=c,s=3**2)
    #for i in xrange(len(Methods)):
    #    plt.bar([Methods[i],Methods[i]],[values[i]+stds[i],values[i]-stds[i]],c=c[i])
    plt.bar(MethodList,Means,width=0.3,yerr=Errors)
    plt.xlabel('Methods')
    plt.ylabel(metric)
    #plt.title(title)
    plt.grid(True,alpha=0.2,axis='y')
    #if ylims is not None:
    #    plt.ylim(ylims)
    plt.tight_layout()
    plt.savefig('./figs/%s.eps'%(metric))
    plt.show()


def plot():
    data = loadData()
    plotBar(data,"AUC")
    plotBar(data,"AUPR")
if __name__ == "__main__":
    plot()