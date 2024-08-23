import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import networkx as nx
from tqdm import tqdm
import os
import pickle as pkl

def checkPath(path):
	if not os.path.exists(path):
		os.mkdir(path)
	return

class DataHandler:
	def __init__(self):
		if args.data == 'yelp':
			predir = '../../Datasets/sparse_yelp/'
		elif args.data == 'gowalla':
			predir = '../../Datasets/sparse_gowalla/'
		elif args.data == 'amazon':
			predir = '../../Datasets/sparse_amazon/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'
		self.ppr_savePath = os.path.join(self.predir, f'ppr_scores/')
		checkPath(self.ppr_savePath)

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeAdj_2(self, allOneAdj):
		allOneAdj_2 = t.sparse.mm(allOneAdj, allOneAdj).coalesce()
		rows = allOneAdj_2._indices()[0, :]
		cols = allOneAdj_2._indices()[1, :]
		diag = (rows == cols)
		rows = rows[t.logical_not(diag)]
		cols = cols[t.logical_not(diag)]
		values = t.ones_like(rows, dtype=t.float32)
		return t.sparse.FloatTensor(t.stack([rows, cols], dim=0), values, allOneAdj.shape).cuda()

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0

		print('Generating PPR.')
		G = nx.from_scipy_sparse_array(mat)
		for h in tqdm(G.nodes(), ncols=50, leave=False):
			ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(h)}.pkl')
			if os.path.exists(ent_ppr_savePath):
				pass
			else:
				# with default setting to generate ppr scores
				h_ppr_scores = nx.pagerank(G, personalization={h: 1}, backend="cugraph")
				pkl.dump(h_ppr_scores, open(ent_ppr_savePath, 'wb'))
		print('Finished.')

		mat = self.normalizeAdj(mat)
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()

	def makeAllOne(self, torchAdj):
		idxs = torchAdj._indices()
		vals = t.ones_like(torchAdj._values())
		shape = torchAdj.shape
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()

	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		args.user, args.item = trnMat.shape
		self.torchBiAdj = self.makeTorchAdj(trnMat)
		self.allOneAdj = self.makeAllOne(self.torchBiAdj)
		self.adj_2 = self.makeAdj_2(self.allOneAdj)
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)
	
class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
