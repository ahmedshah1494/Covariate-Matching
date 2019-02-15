import numpy as np
import itertools

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
	return int(x.split('/')[0][4:])-1
def loadVCIDdata(path, id_mapping):
	data = np.loadtxt(path, usecols=1, dtype=str)
	np.random.shuffle(data)
	ids = np.array([idFromPath(x) for x in data])
	covariates = np.array([id_mapping[x] for x in ids])
	return ids, covariates

def loadVCVeriData(path, id_mapping):
	data =np.loadtxt(path, dtype=str)
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

if __name__ == '__main__':
	id_mapping, cov_mappings = loadVCMeta('vox1_meta.csv')
	ids, covariates = loadVCIDdata('iden_split.txt', id_mapping)
	labels, pairs = loadVCVeriData('veri_test.txt', id_mapping)
	G = genGallery([(0,1),(0,35)], 2)
	Q = genQueries(G, 2)
	print labels[:5], pairs[:5]
	
	