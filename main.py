#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from func import *
import time
from joblib import Parallel, delayed

AREA = 500.0    #area length[m]
CORE = 8        #number of CPU cores (for parallel computation)

'''for Wireless Channels'''
PTx     = 0.0               #transmission power [dBm]
PTx_mW  = 10.0**(PTx*0.1)   #transmission power [mW]
AWGN    = -172.0            #AWGN[dBm/Hz] (note: this value should be -174.0 in practice...)
AWGN_mW = 10.0**(AWGN*0.1)  #AWGN[mW/Hz]

loop = 1000

OPTIMIZER = "DIRECTED" 
## "DIRECTED": brute-force search (equivalent to Alg.2)
## "UNDIRECTED": assuming R_1=R_2=...=R_(ij) (used in Appendix)
DIRECTED_EXACT = False #False: equavalent to main proposed algorithm

def dosim(xi_target, eta, pout, worker, bw):
    '''gen node locations'''
    x, y = genLocation(worker, AREA)

    '''distance matrix (between nodes)'''
    dmat = genDistanceMatrix(x, y)

    '''gen desired topology (d_com: maximum communication distance vector)'''
    if OPTIMIZER == "DIRECTED":
        if DIRECTED_EXACT == True:
            '''for validation (fully brute force)'''
            ## rate adaptation: applied to all candidate vector
            d_com, connmat_desired = stochasticOptimizedTopology(x, y, xi_target, eta, pout, PTx_mW, AWGN_mW, bw)
        else:
            '''main proposed algorithm (Alg.2 + rate optimization)'''
            ## topology search: brute force (alg.2)
            ## rate adaptation: applied to output from alg.2
            d_com, connmat_desired = optimizeTopologyDirectedMod(x, y, xi_target, eta, PTx_mW, AWGN_mW, bw)

    if OPTIMIZER == "UNDIRECTED":
        '''assuming assuming R_1=R_2=...=R_(ij)'''
        ## approx=True: for Appendix
        d_com, connmat_desired = optimizeTopologyUndirectedMod(x, y, xi_target, eta, PTx_mW, AWGN_mW, bw, approx=True)

    '''calc transmission rates and normalized communication time'''
    rvec, status = rateAlloc(d_com, dmat, eta, pout, PTx_mW, AWGN_mW, bw)
    tcom = (1.0/rvec).sum()

    '''gen fading gain (over power domain)'''
    g = np.random.exponential(1.0, (worker, worker)) #fading
    prx = PTx_mW * pathloss(dmat, eta) * g #instantaneous Prx[mW]
    c = bw * np.log2(1.0 + prx/(AWGN_mW*bw)) #capacity matrix

    '''actual connectivity (equivalent to adjecent matrix with self loop)'''
    connmat = np.zeros([worker, worker])
    for i in range(worker):
        connmat[i] = rvec[i] <= c[i]
        connmat[i, i] = 1
    connmat *= connmat_desired

    '''eigenvalue computation'''
    ## broadcasting case (NOT used for the paper)
    w = convertConnmatToWeightMat(connmat)
    eig = np.linalg.eig(w)
    xi = np.sort(np.abs(eig[0]))[::-1][1]

    ## multicasting case (used for the paper)
    w = convertConnmatToWeightMat(connmat_desired)
    eig = np.linalg.eig(w)
    xi_desired = np.sort(np.abs(eig[0]))[::-1][1]


    # print(xi_target, eta, pout, bw, status, tcom, xi, sep=',')
    return tcom, xi, xi_desired, rvec.mean()

def printParameters():
    import datetime
    print("-----------------------------------------------")
    print("Datetime:", datetime.datetime.now())
    print("AREA:", AREA, "[m]")
    print("PTx:", PTx, "[dBm]")
    print("AWGN:", AWGN, "[dBm/Hz]")
    print("loop:", loop)
    print("CORE:", CORE)
    print("OPTIMIZER:", OPTIMIZER)
    print("-----------------------------------------------")

if __name__ == "__main__":
    printParameters()

    bws = np.array([1.4e6]) #bandwidth list
    print(bws, OPTIMIZER, DIRECTED_EXACT)

    for worker in [4]:
        for pout in [0.5]:
            for eta in [4.0, 3.0]:
                for xi_target in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    t = time.time()
                    for bw in bws:
                        result = Parallel(n_jobs=CORE, timeout=None)([delayed(dosim)(xi_target, eta, pout, worker, bw) for n in range(loop)])
                        result = np.array(result)
                        tcom = result[:, 0]
                        xi_out = result[:, 1]
                        xi_desired = result[:, 2]
                        r_mean = result[:, 3]
                        # for i in range(loop):
                        #     print(tcom[i], xi_out[i])
                        cnt_outage = (xi_out > xi_target).sum()
                        cnt_outage_for_desired = (xi_out > xi_desired).sum()
                        print(worker, bw, eta, xi_target, pout, tcom.mean(), r_mean.mean(), float(cnt_outage)/float(loop), float(cnt_outage_for_desired)/float(loop), sep=',')
                        # print(",,")
                    # print(time.time() - t)
