import numpy as np
import cvxpy as cp
import itertools
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import scipy

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)
  
def pathloss(d, eta):
    delta = 1.0e-1
    return (d+delta)**(-eta)

def genLocation(size, len_area):
    x = np.random.uniform(0.0, len_area, size)
    y = np.random.uniform(0.0, len_area, size)
    return x, y

'''Signal-to-Noise power Ratio in True Domain'''
def calcSNR(ptx, eta, d, awgn_mw, bw):
    prx = ptx * pathloss(d, eta)
    return prx / (awgn_mw*bw)

''' Channel Capacity '''
def calcCapacity(ptx, eta, d, awgn_mw, bw):
    snr = calcSNR(ptx, eta, d, awgn_mw, bw)
    return bw * np.log2(1.0+snr)

'''distance matrix'''
def genDistanceMatrix(x, y):
    return distance(x, y, x[:, np.newaxis], y[:, np.newaxis])


'''compute whether the topology is strongly connected'''
## a: adjacency matrix WITHOUT self loop
def DFS(a):
    n = len(a) #number of nodes
    seen = np.zeros(n, dtype=bool) #seen vector
    seen[0] = True
    tmp = np.where(a[0]==1)[0] 
    seen[tmp] = True
    td = tmp.tolist()

    while len(td) > 0:
        indx = td[len(td)-1] #stack structure for DFS
        td.pop(len(td)-1)
        t = np.where(a[indx]==1)[0]
        for w in t:
            if seen[w] == False:
                seen[w] = True
                td.append(w)
                
    return len(np.where(seen[:] == True)[0]) == len(seen)

'''Gen Adjacency matrix with self loop (based on communication distance)'''
def genConnmat(dmat, d_vec):
    connmat = np.array([(dmat[j] <= d_vec[j]) for j in range(len(dmat))]) * 1
    np.fill_diagonal(connmat, 1)
    return connmat

'''Gen Adjacency matrix with self loop (based on capacity)'''
def genConnmatRate(cmat, r_vec):
    connmat = np.array([(cmat[j] >= r_vec[j]) for j in range(len(cmat))]) * 1
    np.fill_diagonal(connmat, 1)
    return connmat

def convertConnmatToWeightMat(connmat):
    n_node = len(connmat)
    w = np.array([connmat[i] / connmat[i].sum() for i in range(n_node)])
    return w

def optimizeTopologyDirectedMod(x, y, xi_target, eta, ptx_mw, awgn_mw, bw):
    dmat = genDistanceMatrix(x, y)

    n = len(x)
    indx_opt = []
    t_min = np.inf

    p = itertools.product([i for i in range(n-1)], repeat=n)
    indx_kouho = np.array([np.argsort(dmat[i])[1:][::-1] for i in range(len(dmat))])
    ## "kouho" means "candidate"
    for v in p:
        d_vec = np.array([dmat[j, indx_kouho[j, v[j]]] for j in range(n)])
        connmat = genConnmat(dmat, d_vec)
        weight = convertConnmatToWeightMat(connmat)
        eig= np.linalg.eig(weight)
        a = np.sort(np.abs(eig[0]))[::-1]

        if (a[1] <= xi_target):
            adjmat = np.copy(connmat)
            np.fill_diagonal(adjmat, 0)

            if (DFS(adjmat) == True):
                rate = calcCapacity(ptx_mw, eta, d_vec, awgn_mw, bw)
                t = (1.0 / rate ).sum() #total communication time
                if t < t_min:
                    t_min = t
                    indx_opt = [indx_kouho[j, v[j]] for j in range(n)]

    d_min = np.array([dmat[j, indx_opt[j]] for j in range(len(indx_opt))])
    connmat_opt = genConnmat(dmat, d_min)

    return d_min, connmat_opt

#assuming R_1=R_2=...=R{ij}
def optimizeTopologyUndirectedMod(x, y, xi_target, eta, ptx_mw, awgn_mw, bw, approx=False):
    dmat = genDistanceMatrix(x, y)
    cmat = calcCapacity(ptx_mw, eta, dmat, awgn_mw, bw)
    n = len(x)
    
    t_min = np.inf

    r_kouho = np.sort(np.triu(cmat).flatten())
    r_kouho = r_kouho[r_kouho>0.0]

    if approx==True: #quantized solution (for Appendix)
        r_kouho = np.linspace(np.min(r_kouho), np.max(r_kouho), 2000)

    for r in r_kouho:
        r_vec = np.full(n, r)
        connmat = genConnmatRate(cmat, r_vec)

        adjmat = np.copy(connmat)
        np.fill_diagonal(adjmat, 0)

        graph = csr_matrix(adjmat)
        n_con = connected_components(graph, directed=False, return_labels=False)
        if (n_con==1): ## strongly connected case
            weight = convertConnmatToWeightMat(connmat)

            eig = scipy.linalg.eigvalsh(weight, check_finite=False, eigvals=(n-2, n-2))[0]
            if (eig <= xi_target):
                t = (1.0 / r_vec ).sum() #total communication time
                if t < t_min:
                    t_min = t
                    r_min = np.copy(r_vec)
        else:
            break

    connmat_opt = genConnmatRate(cmat, r_min)

    d_min = np.array([np.max(connmat_opt[i]*dmat[i]) for i in range(n)])

    return d_min, connmat_opt


def stochasticOptimizedTopology(x, y, xi_target, eta, pout, ptx_mw, awgn_mw, bw):
    dmat = genDistanceMatrix(x, y)
    n = len(x)

    '''brute force search'''
    indx_opt = []
    p = itertools.product([i for i in range(n-1)], repeat=n)
    indx_kouho = np.array([np.argsort(dmat[i])[1:][::-1] for i in range(len(dmat))])

    t_min = np.inf
    r_opt = np.zeros(n)
    for v in p:
        d_vec = np.array([dmat[j, indx_kouho[j, v[j]]] for j in range(n)])
        connmat = genConnmat(dmat, d_vec)
        weight = convertConnmatToWeightMat(connmat)
        eig= np.linalg.eig(weight)
        a = np.sort(np.abs(eig[0]))[::-1]

        adjmat = np.array([(dmat[j] <= d_vec[j]) for j in range(n)]) * 1
        for j in range(len(adjmat)):
            adjmat[j][j] = 0


        if (DFS(adjmat) == True) and (a[1] <= xi_target):
            rvec, status = rateAlloc(d_vec, dmat, eta, pout, ptx_mw, awgn_mw, bw)
            t = (1.0/rvec).sum()
            if t < t_min:
                t_min = t
                indx_opt = [indx_kouho[j, v[j]] for j in range(n)]
                r_opt = np.copy(rvec)

    d_min = np.array([dmat[j, indx_opt[j]] for j in range(len(indx_opt))])
    connmat_opt = genConnmat(dmat, d_min)

    # return d_min, connmat_opt, t_min, r_opt, status
    return d_min, connmat_opt

def rateAlloc(d_com, dmat, eta, pout, ptx_mw, awgn_mw, bw):
    d_tmp = []
    n = len(dmat)

    for i in range(len(dmat)):
        a = dmat[i, np.where(dmat[i]<=d_com[i])][0]
        d_tmp.append(a[a>0])

    b = np.zeros(n)
    for i in range(n):
        tmp = calcSNR(ptx_mw, eta, d_tmp[i], awgn_mw, bw)
        b[i] = (1.0/tmp).sum()

    r = cp.Variable(n, pos=True)
    obj = cp.Minimize(cp.sum(cp.inv_pos(r)))

    constraints = [-r <= 0, cp.log(1.0 - pout) - cp.sum(b) + cp.sum(cp.exp(r*cp.log(2.0) + cp.log(b))) <= 0]
    #note: all SNRs should be larger than 1.0
    
    prob = cp.Problem(obj, constraints)
    result = prob.solve(solver='SCS', verbose=False)

    return (r.value) * bw, prob.status