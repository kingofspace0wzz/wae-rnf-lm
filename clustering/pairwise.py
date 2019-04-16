import torch

def pairwise_distance(data1, data2=None, device=-1):
	if data2 is None:
		data2 = data1 

	if device!=-1:
		data1, data2 = data1.to(device), data2.to(device)

	#N*1*M
	A = data1.unsqueeze(dim=1)

	#1*N*M
	B = data2.unsqueeze(dim=0)

	dis = (A-B)**2.0
	#return N*N matrix for pairwise distance
	dis = dis.sum(dim=-1).squeeze()
	return dis

def group_pairwise(X, groups, device=0, fun=lambda r,c: pairwise_distance(r, c).cpu()):
	group_dict = {}
	for group_index_r, group_r in enumerate(groups):
		for group_index_c, group_c in enumerate(groups):
			R, C = X[group_r], X[group_c]
			if device!=-1:
				R = R.to(device)
				C = C.to(device)
			group_dict[(group_index_r, group_index_c)] = fun(R, C)
	return group_dict
