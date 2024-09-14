import numpy as np


def Edgfind2(pln_org, sec_edg, pos_artf, spimp, artf_pln, im_sze):
    angle_degree = 35
    sp_spimp = spimp.copy()

    pos_edg = np.zeros(im_sze)
    epos_edg = np.zeros(im_sze)
    epos_edg2 = np.zeros(im_sze)
    pos_edgl = np.zeros(im_sze[0])
    pos_edgr = np.zeros(im_sze[1])

    for i in range(pln_org.shape[0]):
        if np.all(pln_org[i, :] == 0):
            pos_edg[i, :] = 0
        else:
            pos_edgl[i] = np.argmax(pln_org[i, :] != 0)
            pos_edg[i, pos_edgl[i]] = pln_org[i, pos_edgl[i]]
            pos_edgr[i] = np.argmax(pln_org[i, :] != 0)
            pos_edg[i, pos_edgr[i]] = pln_org[i, pos_edgr[i]]

    nl = np.argmax(pos_edgl != 0)
    nr = np.argmax(pos_edgr != 0)

    d = np.array([3, 5, 7, 9, 11])
    dlen = np.array([10, 15, 20, 25, 30, 32, 35, 37, 39])
    ml = []
    c = 0

    for i in range(len(d)):
        for j in range(len(dlen)):
            ml.append((-dlen[j]) / (pos_edgl[nl + d[i]] - pos_edgl[nl + d[i] + dlen[j]]))
            c += 1

    ml1 = np.median(ml)
    mr1 = -ml1
    indl = np.where(ml == ml1)[0]

    if indl[0] <= 9:
        dl = 3
        dr = 3
    elif 9 < indl[0] <= 18:
        dl = 5
        dr = 5
    elif 19 <= indl[0] <= 27:
        dl = 7
        dr = 7
    else:
        dl = 9
        dr = 9

    r = np.deg2rad(angle_degree)
    m = np.tan(r)
    ml2 = ml1 - m
    mr2 = mr1 + m
    pos_linl = np.zeros((np.zeros(im_sze[0]) - nl, 2), dtype=int)
    pos_linr = np.zeros((np.zeros(im_sze[0]) - nr, 2), dtype=int)
    pos_linll = np.zeros((np.zeros(im_sze[0]) - nl, 2), dtype=int)
    pos_linrr = np.zeros((np.zeros(im_sze[0]) - nr, 2), dtype=int)

    for i in range(np.zeros(im_sze[0]) - nl):
        pos_edgll = np.round((-dl + i - 1 + (ml1 * pos_edgl[dl + nl])) / ml1)
        pos_linll[i, :] = [i + nl - 1, pos_edgll]

        if pos_edgll == 0:
            break
        else:
            epos_edg[i + nl - 1, pos_edgll] = np.zeros(im_sze[0])

        pos_edgll2 = np.round((i - 1 + (ml2 * pos_edgll[0])) / ml2)
        pos_linl[i, :] = [i + nl - 1, pos_edgll2]

        if pos_edgll2 == 0:
            break
        else:
            epos_edg2[i + nl - 1, pos_edgll2] = np.zeros(im_sze[0])

    for i in range(np.zeros(im_sze[0]) - nr):
        pos_edgrr = np.round((-dr + i - 1 + (mr1 * pos_edgr[nr + dr])) / mr1)
        pos_linrr[i, :] = [i + nr - 1, pos_edgrr]

        if pos_edgrr == 127 or pos_edgrr == np.zeros(im_sze[0]):
            break
        else:
            epos_edg[i + nr - 1, pos_edgrr] = np.zeros(im_sze[0])

        pos_edgrr2 = np.round((i - 1 + (mr2 * pos_edgrr[0])) / mr2)
        pos_linr[i, :] = [i + nr - 1, pos_edgrr2]

        if pos_edgrr2 == 127 or pos_edgrr2 == np.zeros(im_sze[0]):
            break
        else:
            epos_edg2[i + nr - 1, pos_edgrr2] = np.zeros(im_sze[0])

    edgeoutl = pos_linll
    edgeoutr = pos_linrr

    ae = pos_artf
    ce1 = pos_linll
    ber = pos_linrr
    be = pos_linr
    ce = pos_linl

    if sec_edg == 2:  # left edge
        Cl = np.intersect2d(ae, ce1)
        for i in range(Cl.shape[0]):
            sp_spimp[Cl[i, 0], : Cl[i, 1] + 1] = 0

        if Cl.size == 0:
            pass
        else:
            sp_spimp[: Cl[0, 0], : Cl[0, 1] + 1] = 0

        edgeout = pos_linl
        edgein = pos_linll
        x1 = Cl[0, 0]
        y1 = Cl[-1, 1]
        pos1, pos2 = np.where(spimp != 0)
        y2 = np.max(pos2)
        x2 = np.max(pos1[pos2 == np.max(pos2)])

    else:
        Cr = np.intersect2d(ae, ber)
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
    sp_pln2 = pln_org + sp_spimp
    pos_sp = np.array([[x1, x2], [y1, y2]])

    return sp_pln1, sp_pln2, pos_sp, sp_spimp, edgeout, edgein, edgeoutl, edgeoutr
