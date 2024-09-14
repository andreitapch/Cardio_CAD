import numpy as np


def Edgfindf(pln_org, sec_edg, pos_rtf, spimp, edgeoutl, edgeoutr):
    sp_spimp = np.copy(spimp)  # ellipsoid

    ae = pos_rtf
    ce1 = edgeoutl
    ber = edgeoutr

    if sec_edg == 2:  # left edge
        Cl = np.intersect1d(ae, ce1, return_indices=True)

        for i in range(Cl.shape[1]):
            sp_spimp[Cl[0, i], : Cl[1, i] + 1] = 0

        if Cl.size == 0:
            pos1, pos2 = np.where(spimp != 0)
            x1 = np.min(pos1)
            y1 = np.min(pos2)
            y2 = np.max(pos2)
            x2 = np.max(pos1[np.where(pos2 == y2)])

        else:
            sp_spimp[: Cl[0, 0] + 1, : Cl[1, 0] + 1] = 0
            x1 = Cl[0, 0]
            y1 = Cl[1, -1]
            pos1, pos2 = np.where(spimp != 0)
            y2 = np.max(pos2)
            x2 = np.max(pos1[np.where(pos2 == y2)])

    else:
        Cr = np.intersect1d(ae, ber, return_indices=True)

        for i in range(Cr.shape[1]):
            sp_spimp[Cr[i], Cr[0, i]:] = 0

        if Cr.size == 0:
            pos1, pos2 = np.where(sp_spimp != 0)
            y1 = np.min(pos2)
            x2 = np.max(pos1[np.where(pos2 == y1)])
            y2 = np.max(pos2)
            x1 = np.min(pos1)

        else:
            sp_spimp[: Cr[0, 0] + 1, Cr[1, 0]:] = 0
            x1 = Cr[0, 0]
            y2 = Cr[1, -1]
            pos1, pos2 = np.where(sp_spimp != 0)
            y1 = np.min(pos2)
            x2 = np.max(pos1[np.where(pos2 == y1)])

    sp_pln2 = pln_org + sp_spimp  # artificial image

    pos_sp = np.array([[x1, x2], [y1, y2]])  # pixels of ellipsoid (artifact position)

    return sp_pln2, pos_sp, sp_spimp
