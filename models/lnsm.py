import numpy as np
import utils, const
from numpy.linalg import pinv
from qpsolvers import solve_qp


def getRowLNSM(v, mInp, idx=-1):
    nObj = mInp.shape[0]
    ar = np.zeros(nObj)
    for i, inp in enumerate(mInp):
        ar[i] = utils.getTanimotoScore(v, inp)
    if idx >= 0:
        ar[idx] = -10
    args = np.argsort(ar)[::-1][:const.KNN]
    P = np.ndarray((const.KNN, const.KNN))
    for i in range(const.KNN):
        for j in range(i, const.KNN):
            P[i][j] = np.dot(v - mInp[args[i]], v - mInp[args[j]])
            P[j][i] = P[i][j]

    I = np.diag(np.ones(const.KNN))
    P = P + I
    q = np.zeros(const.KNN)
    gg = np.ndarray(const.KNN)
    gg.fill(-1)
    G = np.diag(gg)
    h = np.zeros(const.KNN)
    b = np.ones(1)
    A = np.ones(const.KNN)
    re = solve_qp(P, q, G, h, A, b)
    out = np.zeros(nObj)
    for i in range(const.KNN):
        out[args[i]] = re[i]
    return out


def learnLNSM(mInp, mOut):
    nObj = mInp.shape[0]
    simAr = []
    for i in range(nObj):
        lnsm = getRowLNSM(mInp[i], mInp, i)
        simAr.append(lnsm)
    W = np.vstack(simAr)

    I = np.diag(np.ones(nObj))
    W = W * const.ALPHA
    I = I - W
    Y = pinv(I)
    Y *= 1 - const.ALPHA
    Y = np.matmul(Y, mOut)
    return Y
