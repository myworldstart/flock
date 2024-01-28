import numpy as np


def normalizeStaining(I, Io, beta, alpha, HERef, maxCRef):

    if 'Io' not in locals() or not Io:
        Io = 240

    if 'beta' not in locals() or not beta:
        beta = 0.15

    if 'alpha' not in locals() or not alpha:
        alpha = 1

    if 'HERef' not in locals() or not HERef:
        HERef = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])

    if 'maxCRef' not in locals() or maxCRef is None:
        maxCRef = np.array([1.9705, 1.0308])

    h = I.size(0)
    w = I.size(1)

    I = I.astype(np.float64)
    I = np.reshape(I, (-1, 3))

    OD = -1*np.log((I + 1) / Io)

    ODhat = OD[~np.any(OD < beta, axis=1)]

    covariance_matrix = np.cov(ODhat, rowvar=False)
    D, V = np.linalg.eig(covariance_matrix)
    ind = np.argsort(D)
    D = D[ind]
    V = V[:,ind]

    That = ODhat*V[:, 1:3]
    phi = np.arctan2(That[:,1], That[:,0])

    minPhi = np.percentile(phi, alpha, interpolation='lower')
    maxPhi = np.percentile(phi, 100 - alpha, interpolation='lower')

    vMin = np.dot(V[:, 1:3], np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(V[:, 1:3], np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    if(vMin[0] > vMax[0]):
        HE = np.hstack((vMin, vMax))
    else:
        HE = np.stack((vMax, vMin))

    Y = np.reshape(OD, (-1, 3)).T

    C = np.dot(np.linalg.inv(HE), Y)
    maxC = np.percentile(C, 99, 2, interpolation='lower')

    np.divide(C, maxC, C)
    C = C * maxCRef
    Inorm = Io * np.exp(-np.dot(HERef, C))





