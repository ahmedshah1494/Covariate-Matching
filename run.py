import numpy as np
from src.data_processing import *
from src.experiments import *
from src.continuous_experiments import ContinuousClassificationExperiment, ContinuousVerificationExperiment
import logging

def runVeriTest(toy=True, true_prob=0.9, naive=False, verbose=False, FR=None):
    id_map, _ = loadVCMeta('data/vox1_meta.csv')
    if toy:
        VC, id_map, gdata, ids = buildToyDataset([(0,1),(0,5)], 3, 5)
        qset = []
        labels = []
        for i in range(len(gdata)):
            neg =   gdata[ids != ids[i]]
            pos = gdata[ids == ids[i]]
            for j in range(len(pos)):
                qset.append(np.array([gdata[i], pos[j]]))
                labels.append(1)

                qset.append(np.array([gdata[i], neg[j]]))
                labels.append(0)
        qset = np.array(qset)
        # qset = np.array([np.array(x) for x in itertools.product(gdata, gdata)])
        # labels = [x == y for (x,y) in itertools.product(ids, ids)]
    else:
        labels, qset = loadVCVeriData('data/veri_test.txt', id_map)
        labels = labels[:500]
        qset = qset[:500]
    Q = qset[:,[0]].squeeze()
    G = qset[:,[1]].squeeze()
    rangeVC = [np.arange(min(G[:,i].min(), Q[:,i].min()), max(G[:,i].max()+1, Q[:,i].max()+1)) for i in range(G.shape[1])]
    VC = np.array([[x,y] for x,y in itertools.product(*rangeVC)])
    e = VerificationExperiment(VC, G, Q, true_prob=true_prob, FR=FR)
    qset = zip(range(Q.shape[0]), range(G.shape[0]))
    return e.test(qset, labels=labels, naive=naive, verbose=verbose)

def runIDTest(dataset='toy', true_prob=0.9, naive=False, verbose=False):
    id_map, _ = loadVCMeta('data/vox1_meta.csv')
    if dataset == 'toy':
        VC, id_map, gdata, ids = buildToyDataset([(0,1),(0,5)], 3, 5)
    elif dataset =='sre':
        id_map, mappings = loadSREMeta('SRE/raw/NIST_SRE08_speaker.csv')        
        ids = loadSREIds('SRE/sre08/sre08_male_sids.npy', 'SRE/sre08/sre08_female_sids.npy').astype('int32')        
        ids, gdata = loadSREIDData(ids, id_map, mappings)
    else:
        ids, gdata = loadVCIDdata('data/iden_split.txt', id_map)    
    G, Q = id_map, gdata    
    e = UniqueMatchExperiment(G, Q, true_prob=true_prob)
    # print np.exp(e.P_H(np.array([[1,2],[1,2],[1,2],[1,2]]), np.array([[1,2],[0,2],[1,1],[0,1]])))
    return e.test(ids, verbose=verbose, naive=naive)

def runRankingTest(dataset='toy', true_prob=0.9, naive=False, verbose=False):    
    if dataset == 'toy':
        VC, id_map, gdata, ids = buildToyDataset([(0,1),(0,5)], 3, 5)
    elif dataset =='sre':
        id_map, mappings = loadSREMeta('SRE/raw/NIST_SRE08_speaker.csv', n_ids=10)        
        ids = loadSREIds('SRE/sre08/sre08_male_sids.npy', 'SRE/sre08/sre08_female_sids.npy').astype('int32')        
        ids, gdata = loadSREIDData(ids, id_map, mappings)
    else:
        id_map, _ = loadVCMeta('data/vox1_meta.csv')
        ids, gdata = loadVCIDdata('data/iden_split.txt', id_map)
    gdata = gdata
    ids = ids
    Q, G = id_map, gdata
    print Q.shape, G.shape
    global P
    P = Pool(24)

    e = RankingExperiment(G, Q, true_prob=true_prob)
    e.test(ids, verbose=verbose, naive=naive, pool=P)

def runContinuousClassificationTest(toy=True):
    id_map, _ = loadVCMeta('data/vox1_meta.csv')
    id_map = id_map.astype('float64')
    rangeVC = [(0,1),(0,5)]
    if toy:
        VC, id_map, gdata, ids = buildFloatingToyDataset(rangeVC, 3, 5)
    else:
        ids, gdata = loadVCIDdata('data/iden_split.txt', id_map)    
        gdata = gdata.astype('float64')
    G, Q = id_map, gdata 
    e = ContinuousClassificationExperiment(rangeVC, G, Q)
    # print np.exp(e.P_H(np.array([[1,2],[1,2],[1,2],[1,2]]), np.array([[1,2],[0,2],[1,1],[0,1]])))
    return e.test(ids, verbose=False, naive=False)

def runContinuousVeriTest(toy=True, true_prob=0.9):
    id_map, _ = loadVCMeta('data/vox1_meta.csv')
    if toy:
        rangeVC = [(0,1),(0,5)]
        VC, id_map, gdata, ids = buildToyDataset([(0,1),(0,5)], 3, 5)
        qset = []
        labels = []
        for i in range(len(gdata)):
            neg =   gdata[ids != ids[i]]
            pos = gdata[ids == ids[i]]
            for j in range(len(pos)):
                qset.append(np.array([gdata[i], pos[j]]))
                labels.append(1)

                qset.append(np.array([gdata[i], neg[j]]))
                labels.append(0)
        qset = np.array(qset)
        # print(labels)
        # qset = np.array([np.array(x) for x in itertools.product(gdata, gdata)])
        # labels = [x == y for (x,y) in itertools.product(ids, ids)]
    else:
        labels, qset = loadVCVeriData('data/veri_test.txt', id_map)
    Q = qset[:,[0]].squeeze()
    G = qset[:,[1]].squeeze()
    rangeVC = [(min(G[:,i].min(), Q[:,i].min()), max(G[:,i].max()+1, Q[:,i].max()+1)) for i in range(G.shape[1])]
    VC = np.array([[x,y] for x,y in itertools.product(*rangeVC)])
    e = ContinuousVerificationExperiment(rangeVC, G, Q)
    qset = zip(range(Q.shape[0]), range(G.shape[0]))

    return e.test(qset, labels=labels, naive=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
    format="%(name)s: %(message)s")

    # print(runContinuousClassificationTest(toy=True))
    # print(runContinuousVeriTest(toy=True, true_prob=0.9))

    # P = [1.0, 0.9, 0.75, 0.5]
    # fr = [0.0, 0.185, 0.429, 0.731]
    # for i in range(len(P)):
    #     print(P[i])
    #     print(runVeriTest(toy=False, true_prob=P[i], naive=True, FR=fr[i]))
    print(runRankingTest(dataset='sre', naive=False, verbose=True, true_prob=0.9))
    # print (runIDTest(dataset='vc', true_prob=0.999))
# print(runContinuousClassificationTest(toy=True))
# gender + native_lang ('MAP =', 0.15840527923672099, 'MRR =', 0.15270920705703317)
# gender + yob + native_lang ('MAP =', 0.49011995579787204, 'MRR =', 0.5264470642625982)
# gender + yob + native_lang + smoker ('MAP =', 0.9343528806657337, 'MRR =', 0.9333333333333332)
# gender + yob + native_lang + smoker (15) ('MAP =', 0.8147924875668049, 'MRR =', 0.8666666666666667)
# gender + yob + native_lang + smoker (p=1.) ('MAP =', 0.9378377291505823, 'MRR =', 0.9333333333333332)
# gender + yob + native_lang + smoker (p=.75) ('MAP =', 0.9138550868301847, 'MRR =', 0.9333333333333332)