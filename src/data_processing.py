import numpy as np
import itertools
import logging
from pprint import pprint
import pandas as pd

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

def _get_collision_rate(data):
    collisions = []
    for j in range(data.shape[1]):
        col = data[:,j].tolist()
        vals = set(col)
        coll = 0.0
        for v in vals:
            coll += col.count(v)
        collisions.append(coll/len(vals) - 1)
    # print(collisions)

    data_tuple = [tuple(r) for r in data.tolist()]
    vc = set(data_tuple)
    collisions = 0.0
    for v in vc:
        collisions += data_tuple.count(v) - 1
    return (collisions/len(vc))
    
def get_all_sublists(L, sublists):
    if L == []:
        return sublists    
    sublists = get_all_sublists(L[1:], [x + [L[0]] for x in sublists]) + get_all_sublists(L[1:], sublists)    
    return sublists

def loadSREMeta(path, quantization_bucket_size=10, n_ids=-1, seed=0):
    selected_cols = ['subjid', 'sex', 'year_of_birth', 'native_language', 'smoker', 'height_cm', 'weight_kg']
    selected_cols = ['subjid', 'sex', 'native_language', 'year_of_birth', 'smoker']
    # selected_cols = ['subjid', 'sex', 'year_of_birth']
    data = pd.read_csv(path)    
    data = data[selected_cols]    
    
    np.random.seed(seed)
    np.random.shuffle(data.values)

    # data = data.sample(frac=1.0, random_state=seed)

    if n_ids == -1:
        n_ids = len(data)    

    mappings = []

    for col in selected_cols[1:]:        
        mapping = {}
        values = data[col].values
        if col in ['year_of_birth', 'height_cm', 'weight_kg']:
            values = values[~np.isnan(values)]

            if col == 'year_of_birth':
                min_yob = values.min()
                print values.min(), values.max()
            if col == 'height_cm':
                min_height = values.min()
            else:
                min_weight = values.min()

            values = (values - values.min()) / quantization_bucket_size
            values = values.round().astype('int32')
            
        elif col == 'smoker':
            values[values != 'Y'] = 'N'

        for i in range(len(values)):
            if type(values[i]) == str or not np.isnan(values[i]):
                mapping.setdefault(values[i], len(mapping))
        mappings.append(mapping)
    
    covariates = []
    id2idx = {}
    mappings = [id2idx] + mappings
    
    for i in range(len(data)):
        if len(covariates) >= n_ids:
            break
        row = data.iloc[i]
        if len([v for v in row.values if type(v) != str and np.isnan(v)]) == 0:
            id2idx.setdefault(row['subjid'], len(id2idx))
            cov = []
            for k,col in enumerate(selected_cols):
                v = row[col]
                if col == 'year_of_birth':    
                    v = (v - min_yob)/quantization_bucket_size
                    v = round(v)
                if col == 'height_cm':
                    v = (v - min_height)/quantization_bucket_size
                    v = round(v)
                if col == 'weight_kg':                    
                    v = (v - min_weight)/quantization_bucket_size
                    v = round(v)
                cov.append(mappings[k][v])
            covariates.append(cov)
    covariates = sorted(covariates, key=lambda x: x[0])
    covariates = np.array(covariates)
    # print covariates[:5]
    # print mappings
    covariates = covariates[:, 1:]

    combs = get_all_sublists(range(covariates.shape[1]),[[]])
    combs_cr = []
    for c in combs:
        cr = _get_collision_rate(covariates[:,c])
        combs_cr.append(([selected_cols[i+1] for i in c], cr))
    pprint(sorted(combs_cr, key=lambda x: x[1]))
    return covariates, mappings

def loadSREDvecs(mdvec_path, fdvec_path):
    mvecs = np.load(mdvec_path)
    fvecs = np.load(fdvec_path)
    return np.concatenate((mvecs, fvecs), axis=0)

def loadSREIds(mid_path, fid_path):
    mid = np.load(mid_path)
    fid = np.load(fid_path)
    return np.concatenate((mid, fid), axis=0)

def loadSREIDData(ids, id_map, mappings):
    id_mapping = mappings[0]
    ids = np.array([mappings[0][id] for id in ids if id in id_mapping])

    covariates = np.array([id_map[i] for i in ids])
    
    np.random.seed(0)
    shuffled_idxs = np.arange(len(ids))
    np.random.shuffle(shuffled_idxs)

    covariates = covariates[shuffled_idxs]
    ids = ids[shuffled_idxs]
    
    # print ids[:5]
    # print covariates.shape

    return ids, covariates

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
    # main()
    id_map, mappings = loadSREMeta('../SRE/raw/NIST_SRE08_speaker.csv')
    # ids = loadSREIds('../SRE/sre08/sre08_male_sids.npy', '../SRE/sre08/sre08_female_sids.npy').astype('int32')
    # loadSREIDData(ids, id_map, mappings)
    
    
    