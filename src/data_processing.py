import numpy as np
import itertools
import logging

def loadVCMeta(path, mappings=None):
    data = np.loadtxt(path, delimiter='\t', usecols=[2,3], skiprows=1, dtype=str)
    if mappings is None:
        gender_mapping = {}
        nationality_mapping = {}
        mappings = [gender_mapping,nationality_mapping]
    else:
        [gender_mapping, nationality_mapping] = mappings

    for i in range(data.shape[0]):
        data[i] = np.array([gender_mapping.setdefault(data[i][0], len(gender_mapping.keys())),
                            nationality_mapping.setdefault(data[i][1], len(nationality_mapping.keys()))])
    data = data.astype('int32')
    return data, mappings

def idFromPath(x):
    return int(x.split('/')[0][3:])-1

def loadVCIDdata(path, id_mapping, subset='test'):
    data = np.loadtxt(path, dtype=str)
    data = data[data[:,0].astype(int) == 3]
    data = data[:,1]    
    np.random.seed(0)
    np.random.shuffle(data)
    ids = np.array([idFromPath(x) for x in data])
    covariates = np.array([id_mapping[x] for x in ids])
    return ids, covariates

def loadVCVeriData(path, id_mapping):
    data =np.loadtxt(path, dtype=str)
    np.random.seed(0)
    np.random.shuffle(data)
    labels = data[:,[0]].astype('int32').reshape(-1)
    pairs = data[:,1:]
    cov_pairs = np.array([[id_mapping[idFromPath(x[0])], id_mapping[idFromPath(x[1])]] for x in pairs])
    return labels, cov_pairs

def genGallery(covRange, nCovSamps):
    covs = itertools.product(*([range(r[0],r[1]+1) for r in covRange])) 
    covs = np.array([list(x) for x in covs])
    G = np.repeat(covs, nCovSamps, axis=0)  
    return G

def genQueries(G, nQueries):
    data = np.concatenate((np.arange(G.shape[0]).reshape(-1,1), G), axis=1)
    Q = np.repeat(data, nQueries, axis=0)
    np.random.shuffle(Q)
    ids = Q[:,[0]].reshape(-1)
    Q = Q[:,range(1,Q.shape[1])]
    return ids,Q

def buildToyDataset(covRange, nIdsPerCov, nGalleryEntriesPerId):
    VC = itertools.product(*([range(r[0],r[1]+1) for r in covRange]))   
    VC = np.array([list(x) for x in VC])

    id_map = np.repeat(VC, nIdsPerCov, axis=0)
    ids = np.arange(id_map.shape[0]).reshape(-1,1)
    
    g_data = np.concatenate((ids, id_map), axis=1)
    g_data = np.repeat(g_data, nGalleryEntriesPerId, axis=0)
    np.random.seed(0)
    np.random.shuffle(g_data)

    g_ids = g_data[:,0].astype('int32')
    G = g_data[:,1:]
    
    return VC, id_map, G, g_ids

def buildFloatingToyDataset(covRange, nIdsPerCov, nGalleryEntriesPerId):
    VC, id_map, G, g_ids = buildToyDataset(covRange, nIdsPerCov, nGalleryEntriesPerId)
    return VC, id_map.astype('float64'), G.astype('float64'), g_ids

def oneHot2weight(v):
    v += 1
    s = np.sum(v)
    for i in range(v.shape[0]):
        v[i] = v[i]/s
    return v

def buildConToyDataset(covRange, nIdsPerCov, nGalleryEntriesPerId):
    VC, id_map, G, g_ids = buildToyDataset(covRange, nIdsPerCov, nGalleryEntriesPerId)
    N = len(id_map)
    cov_data = []
    for j, (_, maxval) in enumerate(covRange):
        data_array = np.zeros((N, maxval + 1))
        for i in range(N):
            data_array[i, id_map[i][j]] = 1
            data_array[i] = oneHot2weight(data_array[i])
        cov_data.append(data_array)
    return VC, cov_data, G, g_ids

def main():
    logger = logging.getLogger("Main")
    data = buildConToyDataset([(0,1),(0,5)], 3, 10)
    logger.debug("Result: %s", data)
    pass

if __name__ == '__main__':
    # id_mapping, cov_mappings = loadVCMeta('vox1_meta.csv')
    # ids, covariates = loadVCIDdata('iden_split.txt', id_mapping)
    # labels, pairs = loadVCVeriData('veri_test.txt', id_mapping)
    # VC, id_map, G, g_ids = buildToyDataset([(0,1),(0,5)], 3, 10)
    # print (VC.shape, id_map.shape, G.shape, g_ids.shape )
    logging.basicConfig(level=logging.DEBUG,
    format="%(name)s: %(message)s")
    main()

    
    
    