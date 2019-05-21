import numpy as np
from src.data_processing import loadSREMeta, loadSREDvecs, loadSREIds

def computeAP(ranking):
    precisions = []
    correct_counts = 0
    for i, hit in enumerate(ranking):
        if hit:
            correct_counts += 1.0
            precisions.append(correct_counts / (i+1))
    return np.mean(precisions)

def computeRR(ranking):
    return 1.0/(np.where(ranking)[0][0]+1)

def computeRankingMetrics(Q, qids, G, gids):
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    G = G / np.linalg.norm(G, axis=1, keepdims=True)

    sim = Q.dot(G.T)

    APs = []
    RRs = []
    for i, scores in enumerate(sim):
        sorted_idx = sorted(range(sim.shape[1]), key=lambda j: scores[j], reverse=True)
        ranking = gids[sorted_idx] == qids[i]     
        ap = computeAP(ranking)
        rr = computeRR(ranking)
        APs.append(ap)
        RRs.append(rr)
    
    map = np.mean(APs)
    mrr = np.mean(RRs)
    print(map, mrr)
    
if __name__ == '__main__':
    _, mappings = loadSREMeta('SRE/raw/NIST_SRE08_speaker.csv', n_ids=10)
    ids = loadSREIds('SRE/sre08/sre08_male_sids.npy', 'SRE/sre08/sre08_female_sids.npy').astype('int32')
    dvecs = loadSREDvecs('SRE/sre08/sre08_male_dvecs.npy', 'SRE/sre08/sre08_female_dvecs.npy')

    id_list = ids.tolist()
    id_mapping = mappings[0]
    id_idx = [i for i,id in enumerate(id_list) if id in id_mapping]
    
    ids = ids[id_idx]
    dvecs = dvecs[id_idx]
    
    qidxs = []
    for id in id_mapping:
        qidx = np.where(ids == id)[0]
        if len(qidx) > 1:
            qidxs.append(qidx[0])

    Q = dvecs[qidxs]
    G = np.delete(dvecs, qidxs, axis=0)
    gids = np.delete(ids, qidxs, axis=0)
    computeRankingMetrics(Q, ids[qidxs], G, gids)
    