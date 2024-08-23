from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
import pickle as pkl
import os
from tqdm import tqdm
from torch_sparse import spmm

init = nn.init.xavier_uniform_

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gcn_layer)])
		self.gtLayers = nn.Sequential(*[GTLayer() for i in range(args.gt_layer)])

	def getEgoEmbeds(self):
		return t.concat([self.uEmbeds, self.iEmbeds], axis=0)

	def forward(self, encoderAdj, decoderAdj=None):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
		embedsLst = [embeds]
		for i, gcn in enumerate(self.gcnLayers):
			embeds = gcn(encoderAdj, embedsLst[-1])
			embedsLst.append(embeds)
		if decoderAdj is not None:
			for gt in self.gtLayers:
				embeds = gt(decoderAdj, embedsLst[-1])
				embedsLst.append(embeds)
		embeds = sum(embedsLst)
		return embeds[:args.user], embeds[args.user:]

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return t.spmm(adj, embeds)


class GTLayer(nn.Module):
	def __init__(self):
		super(GTLayer, self).__init__()
		self.qTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
		self.kTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
		self.vTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
		self.filter = nn.Parameter(init(t.empty(args.user+args.item, args.head)))
		
	def forward(self, adj, embeds):
		indices = adj._indices()
		rows, cols = indices[0, :], indices[1, :]
		rowEmbeds = embeds[rows]
		colEmbeds = embeds[cols]

		qEmbeds = (rowEmbeds @ self.qTrans).view([-1, args.head, args.latdim // args.head])
		kEmbeds = (colEmbeds @ self.kTrans).view([-1, args.head, args.latdim // args.head])
		vEmbeds = (colEmbeds @ self.vTrans).view([-1, args.head, args.latdim // args.head])

		att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
		att = t.clamp(att, -10.0, 10.0)
		att += self.filter[cols]
		expAtt = t.exp(att)
		tem = t.zeros([adj.shape[0], args.head]).cuda()
		attNorm = (tem.index_add_(0, rows, expAtt))[rows]
		att = expAtt / (attNorm + 1e-8)

		resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, args.latdim])
		tem = t.zeros([adj.shape[0], args.latdim]).cuda()
		resEmbeds = tem.index_add_(0, rows, resEmbeds)
		return resEmbeds

class LocalGraph(nn.Module):
	def __init__(self):
		super(LocalGraph, self).__init__()
	
	def makeNoise(self, scores):
		noise = t.rand(scores.shape).cuda()
		noise = -t.log(-t.log(noise))
		return t.log(scores) + noise

	def getPPRscores(self, ent):
		if args.data == 'yelp':
			predir = '../../Datasets/sparse_yelp/ppr_scores/'
		elif args.data == 'gowalla':
			predir = '../../Datasets/sparse_gowalla/ppr_scores/'
		elif args.data == 'amazon':
			predir = '../../Datasets/sparse_amazon/ppr_scores/'
		ent_ppr_savePath = os.path.join(predir, f'{int(ent)}.pkl')
		scores = pkl.load(open(ent_ppr_savePath, 'rb'))
		return scores

	def getPPREntropy(self, allOneAdj, adj_2, embeds, pprMat, entropy):
		indices = adj_2._indices()
		row_indices = indices[0]
		entropy_list = t.empty(0).cuda()
		ppr_idx = t.empty(0).long().cuda()
		ppr_value = t.empty(0).cuda()
		entropy_num = t.sparse.sum(adj_2, dim=-1).to_dense().view([-1, 1])

		for i in tqdm(range(len(embeds)), ncols=50, leave=False):
			# getPPR
			ppr_list = list(self.getPPRscores(i).values())
			ppr = t.tensor(ppr_list, dtype=t.float32).cuda()

			# PPR_entropy
			if entropy_num[i] > 0:
				entropy = -(ppr * t.log(ppr.clamp(min=1e-8))).sum()
			else:
				entropy = t.mean(entropy_list).unsqueeze(0)
			entropy_list = t.cat([entropy_list, entropy], dim=0)

			# PPR_weights
			row_mask = (row_indices == i)
			selected_indices = indices[:, row_mask]
			ppr_idx = t.cat([ppr_idx, selected_indices], dim=1)
			new_values = ppr[selected_indices[1]]
			new_values_softmax = t.softmax(new_values / args.tau, dim=-1)
			ppr_value = t.cat([ppr_value, new_values_softmax], dim=0)

		assert not t.isnan(entropy_list).any()
		assert not t.isnan(ppr_value).any()
		pprMat = t.sparse.FloatTensor(ppr_idx, ppr_value, allOneAdj.shape).cuda()

		with open('ppr_' + args.data + '.pkl', 'wb') as f:
			pkl.dump(pprMat, f)
		with open('entropy_' + args.data + '.pkl', 'wb') as f:
			pkl.dump(entropy_list, f)

		return pprMat, entropy_list

	def forward(self, allOneAdj, adj_2, embeds, pprMat, entropy):
		if pprMat is None or entropy is None:
			pprMat,entropy = self.getPPREntropy(allOneAdj, adj_2, embeds, pprMat, entropy)

		subgraphEmbeds = t.spmm(pprMat, embeds)
		
		entropy_num = (t.sparse.sum(adj_2, dim=-1).to_dense()+1).view([-1, 1])
		entropy_mean = (t.spmm(adj_2, entropy)+entropy)/(entropy_num)
		entropy_mean_nor = ((entropy_mean - entropy_mean.min()) / (entropy_mean.max() - entropy_mean.min())).squeeze(1)

		entropy_num = t.sparse.sum(adj_2, dim=-1).to_dense().view([-1, 1])
		entropy_mean = t.spmm(adj_2, entropy) / (entropy_num+1e-8)
		entropy_mean_l = ((2 * entropy_mean * entropy+1e-8)/(entropy_mean**2+entropy**2+1e-8)).squeeze(1)

		orphan = t.where(entropy_num==0)[0]

		subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)
		embeds = F.normalize(embeds, p=2)
		scores = t.sigmoid(t.sum(subgraphEmbeds * embeds, dim=-1)*entropy_mean_l+entropy_mean_nor)
		scores = self.makeNoise(scores)
		_, seeds = t.topk(scores, args.seedNum)
		seeds = t.cat([seeds, orphan])
		scores = t.clamp(scores, -10.0, 10.0)
		return scores, seeds

class RandomMaskSubgraphs(nn.Module):
	def __init__(self):
		super(RandomMaskSubgraphs, self).__init__()
		self.flag = False
	
	def normalizeAdj(self, adj):
		degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
		newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
		rowNorm, colNorm = degree[newRows], degree[newCols]
		newVals = adj._values() * rowNorm * colNorm
		return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

	def forward(self, adj, seeds):
		rows = adj._indices()[0, :]
		cols = adj._indices()[1, :]
		maskNodes = [seeds]
		for i in range(args.maskDepth):
			curSeeds = seeds if i == 0 else nxtSeeds
			nxtSeeds = list()
			for seed in curSeeds:
				rowIdct = (rows == seed)
				colIdct = (cols == seed)
				idct = t.logical_or(rowIdct, colIdct)
				if i != args.maskDepth - 1:
					mskRows = rows[idct]
					mskCols = cols[idct]
					nxtSeeds.append(mskRows)
					nxtSeeds.append(mskCols)
				rows = rows[t.logical_not(idct)]
				cols = cols[t.logical_not(idct)]
			if len(nxtSeeds) > 0:
				nxtSeeds = t.unique(t.concat(nxtSeeds))
				maskNodes.append(nxtSeeds)
		sampNum = int((args.user + args.item) * args.keepRate)
		sampedNodes = t.randint(args.user + args.item, size=[sampNum]).cuda()
		if self.flag == False:
			l1 = adj._values().shape[0]
			l2 = rows.shape[0]
			print('-----')
			print('LENGTH CHANGE', '%.2f' % (l2 / l1), l2, l1)
			tem = t.unique(t.concat(maskNodes))
			print('Original SAMPLED NODES', '%.2f' % (tem.shape[0] / (args.user + args.item)), tem.shape[0], (args.user + args.item))
		maskNodes.append(sampedNodes)
		maskNodes = t.unique(t.concat(maskNodes))
		if self.flag == False:
			print('AUGMENTED SAMPLED NODES', '%.2f' % (maskNodes.shape[0] / (args.user + args.item)), maskNodes.shape[0], (args.user + args.item))
			self.flag = True
			print('-----')
		encoderAdj = self.normalizeAdj(t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.ones_like(rows).cuda(), adj.shape))

		temNum = maskNodes.shape[0]
		temRows = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).cuda()]
		temCols = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).cuda()]
		newRows = t.concat([temRows, temCols, t.arange(args.user+args.item).cuda(), rows])
		newCols = t.concat([temCols, temRows, t.arange(args.user+args.item).cuda(), cols])
		# filter duplicated
		hashVal = newRows * (args.user + args.item) + newCols
		hashVal = t.unique(hashVal)
		newCols = hashVal % (args.user + args.item)
		newRows = ((hashVal - newCols) / (args.user + args.item)).long()
		decoderAdj = t.sparse.FloatTensor(t.stack([newRows, newCols], dim=0), t.ones_like(newRows).cuda().float(), adj.shape)
		return encoderAdj, decoderAdj