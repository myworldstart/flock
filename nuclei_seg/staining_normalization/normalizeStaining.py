import numpy as np
import matplotlib.pyplot as plt

def normalizeStaining(I, Io=None, beta=None, alpha=None, HERef=None, maxCRef=None):

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
        maxCRef = np.array([[1.9705], [1.0308]])

    h,w,_ = I.shape

    I = I.astype(np.float64)
    I = np.reshape(I, (-1, 3), order="F")

    OD = -1*np.log((I + 1) / Io)

    ODhat = OD[~np.any(OD < beta, axis=1)]

    covariance_matrix = np.cov(ODhat, rowvar=False)
    D, V = np.linalg.eig(covariance_matrix)
    ind = np.argsort(D)
    D = D[ind]
    V = V[:,ind]

    That = np.dot(ODhat, V[:, 1:3])
    phi = np.arctan2(That[:,1], That[:,0])

    minPhi = np.percentile(phi, alpha, interpolation='lower')
    maxPhi = np.percentile(phi, 100 - alpha, interpolation='lower')

    vMin = np.dot(V[:, 1:3], np.array([[np.cos(minPhi)], [np.sin(minPhi)]]))
    vMax = np.dot(V[:, 1:3], np.array([[np.cos(maxPhi)], [np.sin(maxPhi)]]))

    if(vMin[0] > vMax[0]):
        HE = np.hstack((vMin, vMax))
    else:
        HE = np.stack((vMax, vMin))

    Y = np.reshape(OD, (-1, 3)).T

    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    maxC = np.percentile(C, 99, 1, interpolation='lower')
    maxC = np.reshape(maxC, (-1, 1))

    C = np.divide(C, maxC)
    C = C * maxCRef

    Inorm = Io * np.exp(-np.dot(HERef, C))
    Inorm = np.reshape(Inorm.T, (h, w, 3))
    Inorm = Inorm.astype(int)

    HERef1 = -HERef[:, 1]
    HERef1 = np.reshape(HERef1, (-1, 1))
    C1 = C[1, :]
    C1 = np.reshape(C1, (1, -1))

    H = Io * np.log(np.dot(HERef1, C1))
    H = np.reshape(H.T, (h, w, 3))
    H = H.astype(int)

    E = Io * np.log(np.dot(HERef1, C1))
    E = np.reshape(E.T, (h, w, 3))
    E = E.astype(int)

    return Inorm, H, E


# if __name__ == '__main__':
#     img = cv2.imread('/home/xjb/Code1/matlab/flock/img/TMA 002-G6.png')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     crop_img = img[1132:1132+415, 623:623+511, :]
#     print(crop_img[0, 0, :])
#     plt.figure()
#     plt.imshow(crop_img)
#     plt.axis('off')
#     plt.show()
#     Inorm, _, _ = normalizeStaining(crop_img)






