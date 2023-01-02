#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
##import cupy as cp
import pdb
import sys
import time
import csv
from numba import jit

__all__ = ['frdist']

@jit(nopython=True, cache=True)
def linear_frechet(p, q):
    n_p = p.shape[0]
    n_q = q.shape[0]
    ca = np.zeros((n_p, n_q), dtype=np.float64)
    for i in range(n_p):
        for j in range(n_q):
            d = np.linalg.norm(p[i] - q[j])

            if i > 0 and j > 0:
                ca[i, j] = max(min(ca[i - 1, j],
                                   ca[i - 1, j - 1],
                                   ca[i, j - 1]), d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            else:
                ca[i, j] = d
    return ca[n_p - 1, n_q - 1]

@jit(nopython=True, cache=True)
def _c(ca, i, j, p, q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i] - q[j])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i - 1, 0, p, q), np.linalg.norm(p[i] - q[j]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j - 1, p, q), np.linalg.norm(p[i] - q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i - 1, j, p, q),
                _c(ca, i - 1, j - 1, p, q),
                _c(ca, i, j - 1, p, q)
            ),
            np.linalg.norm(p[i] - q[j])
        )
    else:
        ca[i, j] = np.inf

    return ca[i, j]


def frdist(p, q, compute_type='recursive'):
    """
    Computes the discrete FrÃ©chet distance between
    two curves. The FrÃ©chet distance between two curves in a
    metric space is a measure of the similarity between the curves.
    The discrete FrÃ©chet distance may be used for approximately computing
    the FrÃ©chet distance between two arbitrary curves,
    as an alternative to using the exact FrÃ©chet distance between a polygonal
    approximation of the curves or an approximation of this value.

    This is a Python 3.* implementation of the algorithm produced
    in Eiter, T. and Mannila, H., 1994. Computing discrete FrÃ©chet distance.
    Tech. Report CD-TR 94/64, Information Systems Department, Technical
    University of Vienna.
    http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf

    Function dF(P, Q): real;
        input: polygonal curves P = (u1, . . . , up) and Q = (v1, . . . , vq).
        return: Î´dF (P, Q)
        ca : array [1..p, 1..q] of real;
        function c(i, j): real;
            begin
                if ca(i, j) > âˆ’1 then return ca(i, j)
                elsif i = 1 and j = 1 then ca(i, j) := d(u1, v1)
                elsif i > 1 and j = 1 then ca(i, j) := max{ c(i âˆ’ 1, 1), d(ui, v1) }
                elsif i = 1 and j > 1 then ca(i, j) := max{ c(1, j âˆ’ 1), d(u1, vj) }
                elsif i > 1 and j > 1 then ca(i, j) :=
                max{ min(c(i âˆ’ 1, j), c(i âˆ’ 1, j âˆ’ 1), c(i, j âˆ’ 1)), d(ui, vj ) }
                else ca(i, j) = âˆž
                return ca(i, j);
            end; /* function c */

        begin
            for i = 1 to p do for j = 1 to q do ca(i, j) := âˆ’1.0;
            return c(p, q);
        end.

    Parameters
    ----------
    P : Input curve - two dimensional array of points
    Q : Input curve - two dimensional array of points

    Returns
    -------
    dist: float64
        The discrete FrÃ©chet distance between curves `P` and `Q`.

    Examples
    --------
    >>> from frechetdist import frdist
    >>> P=[[1,1], [2,1], [2,2]]
    >>> Q=[[2,2], [0,1], [2,4]]
    >>> frdist(P,Q)
    >>> 2.0
    >>> P=[[1,1], [2,1], [2,2]]
    >>> Q=[[1,1], [2,1], [2,2]]
    >>> frdist(P,Q)
    >>> 0
    """
    p = np.array(p, np.float64)
    q = np.array(q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:
        raise ValueError('Input curves are empty.')

    if len_p != len_q or len(p[0]) != len(q[0]):
        raise ValueError('Input curves do not have the same dimensions.')

    ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)
    if compute_type=='recursive':
        dist = _c(ca, len_p - 1, len_q - 1, p, q)
    if compute_type == 'iterative':
        dist = linear_frechet(p, q)
    return dist


if __name__ == '__main__':

    sys.setrecursionlimit(5000)

    data = csv.reader(open('/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/output/tracktor/online_SCT/global_tracks_clasp2_10FPS_R50.csv'))
    next(data)

    tracklets = []
    for d in data:
        tracklets.append(d)
    trk_array = np.array(tracklets)
    trk_array = np.array(trk_array[:, 2:].astype(float))
    tracklet_cam2 = trk_array[trk_array[:, 4] == 2]
    tracklet_cam5 = trk_array[trk_array[:, 4] == 5]
    linear_times = []
    recurs_time = []
    trk_len = 0
    trk_size = 320
    while trk_len<len(tracklet_cam5)-500:
        tic = time.time()
        d = frdist(tracklet_cam2[trk_len:trk_len+trk_size , 0:2], tracklet_cam5[trk_len:trk_len+trk_size , 0:2], compute_type='recursive')
        print('Recursive: dist: {:2f}, trsize: {}, ET: {}, '.format(d, trk_size, (time.time() - tic)))
        recurs_time.append(time.time() - tic)

        tic = time.time()
        d = frdist(tracklet_cam2[trk_len:trk_len+trk_size , 0:2], tracklet_cam5[trk_len:trk_len+trk_size , 0:2], compute_type='iterative')
        print('Linear: dist: {}, trsize: {}, ET: {}, '.format(d, trk_size, (time.time() - tic)))
        linear_times.append(time.time() - tic)
        trk_len+=trk_size
    print('Linear: Avg: {}, Peak: {}'.format(np.mean(linear_times), np.max(linear_times)))
    print('Recursive: Avg: {}, Peak: {}'.format(np.mean(recurs_time), np.max(recurs_time)))
