import cv2
import numpy as np
from scipy import ndimage


def EllipArtGen2(pln_org, pos_mat, hd_list, vd_list, sec_edg, ang_vec, sigma1, sigma2, ang_rot, gry_max, filtX, filtY):
    sp_pln = pln_org.copy()

    hd = hd_list  # horizontal size
    vd = vd_list  # vertical size

    rct = np.zeros((vd, hd), dtype=np.uint8)  # a rectangle encompassing the ellipsoid
    elip_cent = [(hd - 1) / 2, (vd - 1) / 2]

    # Ellipsoid with Gaussian
    exponent1 = np.exp(-0.5 * ((np.arange(1, hd + 1) - (hd + 1) / 2) / sigma1) ** 2)
    exponent2 = np.exp(-0.5 * ((np.arange(1, vd + 1) - (vd + 1) / 2) / sigma2) ** 2)

    xc = (vd - 1) / 2  # center of ellipsoid (row)
    yc = (hd - 1) / 2  # center of ellipsoid (col)

    for i in range(vd):
        for j in range(hd):
            rct[i, j] = round(gry_max * exponent1[j] * exponent2[i])

    ang_stp = (ang_vec[0] - ang_vec[1]) / sp_pln.shape[1]

    if sec_edg == 1:  # right edge
        ang = ang_vec[0]
        ang = ang - ang_rot * ang_stp
    else:  # left edge
        ang = ang_vec[1]
        ang = ang + ang_rot * ang_stp

    rct = ndimage.rotate(rct, ang)

    # Superimpose the artifact pattern on the image
    row, col = sp_pln.shape
    if pos_mat[1] <= (hd + 1) / 2:
        col = round(sp_pln.shape[1] + (((hd + 1) / 2) - pos_mat[1]))
        spimp = np.full((row, col), np.nan, dtype=np.uint8) #np.zeros((row, col), dtype=np.uint8)
        spimp[(pos_mat[0] - xc):(pos_mat[0] + xc), ((hd + 1) // 2 - yc):((hd + 1) // 2 + yc)] = rct
    else:
        spimp = np.full((row, col), np.nan, dtype=np.uint8)
        # spimp = np.zeros((row, col), dtype=np.uint8)

        cRctRow = (rct.shape[0] / 2)
        cRctCol = (rct.shape[1] / 2)

        row_start = round(pos_mat[0] - cRctRow)
        row_end = round(pos_mat[0] + cRctRow)
        col_start = round(pos_mat[1] - cRctCol)
        col_end = round(pos_mat[1] + cRctCol)
        spimp[row_start:row_end, col_start:col_end] = rct
    #
    # plt.imshow(spimp, cmap='gray')
    # plt.scatter(filtX, filtY, color='blue')
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.axis('off')
    # plt.show()

    spimp = cv2.resize(spimp, (pln_org.shape[1], pln_org.shape[0]))

    x1, y1 = np.where(spimp != 0)
    pos_artf1 = np.column_stack((x1, y1))

    xm1, ym1 = np.where(spimp == np.max(spimp))
    xm = xm1[len(xm1) // 2]
    ym = ym1[len(ym1) // 2]

    pos_artfm = np.array([xm, ym])
    artImg = cv2.add(sp_pln, spimp)

    sp_pln = artImg
    return sp_pln, spimp, pos_artf1, pos_artfm
