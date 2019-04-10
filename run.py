import numpy as np
from src.data_processing import *
from src.experiments import *

def runVeriTest(toy=True, true_prob=0.9):
	id_map, _ = loadVCMeta('data/vox1_meta.csv')
	if toy:
		VC, id_map, gdata, ids = buildToyDataset([(0,1),(0,5)], 3, 5)
		qset = []
		labels = []
		for i in range(len(gdata)):
			neg = 	gdata[ids != ids[i]]
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
	Q = qset[:,[0]].squeeze()
	G = qset[:,[1]].squeeze()
	rangeVC = [np.arange(min(G[:,i].min(), Q[:,i].min()), max(G[:,i].max()+1, Q[:,i].max()+1)) for i in range(G.shape[1])]
	VC = np.array([[x,y] for x,y in itertools.product(*rangeVC)])
	e = VerificationExperiment(VC, G, Q, true_prob=true_prob)
	qset = zip(range(Q.shape[0]), range(G.shape[0]))
	return e.test(qset, labels=labels, naive=True)

def runIDTest((toy, true_prob)):	
	id_map, _ = loadVCMeta('data/vox1_meta.csv')
	if toy:
		VC, id_map, gdata, ids = buildToyDataset([(0,1),(0,5)], 3, 5)
	else:
		ids, gdata = loadVCIDdata('data/iden_split.txt', id_map)	
	G, Q = id_map, gdata	
	e = UniqueMatchExperiment(G, Q, true_prob=true_prob)
	# print np.exp(e.P_H(np.array([[1,2],[1,2],[1,2],[1,2]]), np.array([[1,2],[0,2],[1,1],[0,1]])))
	return e.test(ids, verbose=True, naiive=False)

def runRankingTest(toy=True, true_prob=0.9):
	id_map, _ = loadVCMeta('data/vox1_meta.csv')
	if toy:
		VC, id_map, gdata, ids = buildToyDataset([(0,1),(0,5)], 3, 5)
	else:
		ids, gdata = loadVCIDdata('data/iden_split.txt', id_map)
	gdata = gdata[:500]
	ids = ids[:500]
	Q, G = id_map, gdata

	global P
	P = Pool(12)

	e = RankingExperiment(G, Q, true_prob=true_prob)
	e.test(ids, verbose=False, naive=True, pool=P)




print runRankingTest(True, 0.9)