import numpy as np
import pandas as pd
import cv2

def intersect2D(a, b):
    """
    Find row intersection between 2D numpy arrays, a and b.
    Returns another numpy array with shared rows
    """
    return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])

def Edgfind3(pln_org, sec_edg, pos_artf, spimp, artf_pln, im_sze):
    angle_degree = 35
    sp_spimp = spimp.copy()

    pos_edg = np.zeros(im_sze[:2])
    epos_edg = np.zeros(im_sze[:2])
    epos_edg2 = np.zeros(im_sze[:2])
    pos_edgl = np.zeros(im_sze[0])
    pos_edgr = np.zeros(im_sze[1])

    for i in range(pln_org.shape[0]):
        if np.all(pln_org[i, :] == 0):
            pos_edg[i, :] = 0
        else:
            pos_edgl[i] = np.argmax(pln_org[i, :] != 0)
            posEdgl = int(pos_edgl[i])
            pos_edg[i, posEdgl] = pln_org[i, posEdgl]
            pos_edgr[i] = np.argmax(pln_org[i, :] != 0)
            posEdgr = int(pos_edgr[i])
            pos_edg[i, posEdgr] = pln_org[i, posEdgr]

    nl = np.argmax(pos_edgl != 0)
    nr = np.argmax(pos_edgr != 0)

    d = np.array([9, 13, 20, 25, 29])
    dlen = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55])
    nll = nl + d
    nrr = nr + d
    ml = np.empty(len(d) * len(dlen))
    mr = np.empty(len(d) * len(dlen))
    c = 0

    for i in range(len(d)):
        for j in range(len(dlen)):
            ml[c] = (-dlen[j]) / (pos_edgl[nll[i]] - pos_edgl[nll[i] + dlen[j]])
            mr[c] = (-dlen[j]) / (pos_edgr[nrr[i]] - pos_edgr[nrr[i] + dlen[j]])
            c += 1

    ml1 = np.median(ml)
    mr1 = np.median(mr)
    indl = np.where(ml == ml1)[0]
    indr = np.where(mr == mr1)[0]

    if indl[0] <= 9:
        dl = 3
    elif 9 < indl[0] <= 18:
        dl = 7
    elif 19 <= indl[0] <= 27:
        dl = 9
    else:
        dl = 13

    if indr[0] <= 9:
        dr = 3
    elif 10 <= indr[0] <= 18:
        dr = 7
    elif 19 <= indr[0] <= 27:
        dr = 9
    else:
        dr = 13

    r = np.deg2rad(angle_degree)
    m = np.tan(r)
    ml2 = ml1 - m
    mr2 = mr1 + m
    pos_linl = np.zeros((im_sze[0] - nl, 2), dtype=int)
    pos_linr = np.zeros((im_sze[0] - nr, 2), dtype=int)
    pos_linll = np.zeros((im_sze[0] - nl, 2), dtype=int)
    pos_linrr = np.zeros((im_sze[0] - nr, 2), dtype=int)

    for i in range(im_sze[0] - nl):
        pos_edgll = int(np.round((-dl + i - 1 + (ml1 * pos_edgl[dl + nl])) / ml1))
        pos_linll[i, :] = [i + nl - 1, pos_edgll]

        if pos_edgll == 0:
            break
        else:
            epos_edg[i + nl - 1, pos_edgll] = im_sze[0]

        pos_edgll2 = np.round((i - 1 + (ml2 * pos_edgll)) / ml2)
        pos_linl[i, :] = [i + nl - 1, pos_edgll2]

        if pos_edgll2 == 0:
            break
        else:
            epos_edg2[i + nl - 1, int(pos_edgll2)] = im_sze[0]

    for i in range(im_sze[0] - nr):
        pos_edgrr = np.round((-dr + i - 1 + (mr1 * pos_edgr[nr + dr])) / mr1)
        pos_edgrr = int(pos_edgrr)
        pos_linrr[i, :] = [i + nr - 1, pos_edgrr]

        if pos_edgrr == 127 or pos_edgrr == im_sze[0]:
            break
        else:
            epos_edg[i + nr - 1, pos_edgrr] = im_sze[0]

        pos_edgrr2 = np.round((i - 1 + (mr2 * pos_edgrr)) / mr2)
        pos_linr[i, :] = [i + nr - 1, pos_edgrr2]
        pos_edgrr2 = int(pos_edgrr2)
        if pos_edgrr2 == 127 or pos_edgrr2 == im_sze[0]:
            break
        else:
            epos_edg2[i + nr - 1, pos_edgrr2] = im_sze[0]

    edgeoutl = pos_linll
    edgeoutr = pos_linrr

    ae = pos_artf
    ce1 = pos_linll
    ber = pos_linrr
    be = pos_linr
    ce = pos_linl

    if sec_edg == 2:
        df_ae = pd.DataFrame(ae)
        df_ce1 = pd.DataFrame(ce1)
        ClMerge = pd.merge(df_ae, df_ce1)
        Cl = ClMerge.to_numpy()

        Cl = intersect2D(ae, ce1)
        for i in range(Cl.shape[0]):
            sp_spimp[Cl[i, 0], : Cl[i, 1] + 1] = 0

        if Cl.size == 0:
            pos1, pos2 = np.where(sp_spimp != 0)
            x1 = np.min(pos1)
            y1 = np.min(pos2)
            y2 = np.max(pos2)
            x2 = np.max(pos1[pos2 == np.max(pos2)])
        else:
            sp_spimp[: Cl[0, 0], : Cl[0, 1] + 1] = 0
            x1 = Cl[0, 0]
            y1 = Cl[-1, 1]
            pos1, pos2 = np.where(spimp != 0)
            y2 = np.max(pos2)
            x2 = np.max(pos1[pos2 == np.max(pos2)])

        edgeout = pos_linl
        edgein = pos_linll
    else:
        Cr = intersect2D(ae, ber)

        #  TODO INTERESECT ROWS
        # Cr = np.intersect1d(ae, ber)
        for i in range(Cr.shape[0]):
            sp_spimp[Cr[i, 0]:, Cr[i, 1]] = 0

        if Cr.size == 0:
            pos1, pos2 = np.where(sp_spimp != 0)
            y1 = np.min(pos2)
            x2 = np.max(pos1[pos2 == np.min(pos2)])
            y2 = np.max(pos2)
            x1 = np.min(pos1)
        else:
            sp_spimp[: Cr[0, 0], Cr[0, 1]:] = 0
            x1 = Cr[0, 0]
            y2 = Cr[-1, 1]
            pos1, pos2 = np.where(sp_spimp != 0)
            y1 = np.min(pos2)
            x2 = np.max(pos1[pos2 == np.min(pos2)])

        edgeout = pos_linrr
        edgein = pos_linr

    sp_pln1 = pln_org + sp_spimp + epos_edg + epos_edg2
    sp_pln2 = cv2.add(pln_org, sp_spimp)
    pos_sp = np.array([[x1, x2], [y1, y2]])

    return sp_pln1, sp_pln2, pos_sp, sp_spimp, edgeout, edgein, edgeoutl, edgeoutr
