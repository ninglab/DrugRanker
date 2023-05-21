import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.nn import GlobalAttention, SAGPooling

import numpy as np
import ot

from features.featurization import get_atom_fdim, get_bond_fdim
from utils.nn_utils import index_select_ND
from torch_geometric.nn import MLP


class MPNN(nn.Module):
	def __init__(self, args, atom_fdim = None, bond_fdim = None):
		super(MPNN, self).__init__()
		self.atom_fdim = atom_fdim or get_atom_fdim()
		self.bond_fdim = bond_fdim or get_bond_fdim(atom_messages=args.atom_messages)
		self.atom_messages = args.atom_messages
		self.hidden_dim = args.mol_out_size
		self.steps = args.message_steps
		self.layers_per_message = 1
		self.device = args.device
		self.pooling = args.pooling
		self.use_features_only = args.use_features_only
		self.feature_gen = args.feature_gen
		self.agg_emb = args.agg_emb

		self.activation = nn.ReLU()
		self.dropout_layer = nn.Dropout(p=args.dropout)

		input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
		self.W_i = nn.Linear(input_dim, self.hidden_dim, bias=False)
		if self.atom_messages:
			w_m_input_size = self.hidden_dim + self.bond_fdim
		else:
			w_m_input_size = self.hidden_dim

		self.W_m = nn.Linear(w_m_input_size, self.hidden_dim, bias=False)

		mol_dim = self.hidden_dim
		if 'hier-cat' in self.pooling:
			mol_dim *= self.steps

		self.W_a = nn.Linear(mol_dim + self.atom_fdim, mol_dim)

		if self.pooling in ['v-attention', 'hier-cat-v-attention']:
			self.W_v = nn.Linear(mol_dim, mol_dim)

		elif 'attention' in self.pooling: #or self.pooling == 'x-attention':
			self.W_att_1 = nn.Linear(mol_dim, args.attn_dim)
			self.W_att_2 = nn.Linear(args.attn_dim, 1)

		self.cached_zero_vector = nn.Parameter(torch.zeros(mol_dim), requires_grad=False)


	def aggregate(self, message, a2b, b2a, b2revb, a2a, f_bonds):
		if self.atom_messages:
			nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
			nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
			nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
			message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
		else:
			# m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
			neigh_message = index_select_ND(message, a2b)	# num_atoms x max_num_bonds x hidden_dim
			agg_message   = neigh_message.sum(dim = 1)	# num_atoms x hidden_dim
			rev_message   = message[b2revb]			# num_bonds x hidden_dim
			message	      = agg_message[b2a] - rev_message	# num_bonds x hidden_dim
		return message

	def update(self, input, message):
		message = self.W_m(message)
		message = self.activation(input + message)
		message = self.dropout_layer(message)
		return message

	def _readout(self, atom_h, smiles=None, ic=None, draw=False):
		scores = torch.from_numpy(np.ones(atom_h.shape[0])/atom_h.shape[0])
		if 'mean' in self.pooling:
			return atom_h.mean(dim=0), scores

		elif 'sum' in self.pooling:
			return atom_h.sum(dim=0), scores

		elif 'max' in self.pooling:
			return atom_h.max(dim=0)[0], scores

		##TODO: this is deprecated: remove
		elif self.pooling == 'attention':
			scores = torch.softmax(torch.matmul(atom_h, atom_h.T), dim=1)
			x = torch.matmul(scores, atom_h)
			return torch.mean(x, dim=0), scores.flatten()

		elif self.pooling in ['v-attention', 'hier-cat-v-attention']: # virtual super node
			vnode = torch.tanh(self.W_v(torch.sum(atom_h, dim=0))/atom_h.shape[0])
			scores = torch.sigmoid(atom_h*vnode)
			x = scores*atom_h
			return torch.sum(x, dim=0), scores.flatten()

		elif 'attention' in self.pooling:
			temp_h = self.activation(self.W_att_1(atom_h))
			scores = torch.sigmoid(self.W_att_2(temp_h), dim=0) # type: ignore
			x = (1+scores)*atom_h
			return torch.sum(x, dim=0), scores.flatten()
		else:
			raise ValueError('Invalid pooling type {}'.format(self.pooling))


	def readout(self, atom_h, a_scope, mol_graph):
		mol_vecs = []
		node_scores = []
		for i, (a_start, a_size) in enumerate(a_scope):
			if a_size == 0:
				mol_vecs.append(self.cached_zero_vector)
			else:
				temp_h = atom_h.narrow(0, a_start, a_size)
				mol_vec, scores = self._readout(temp_h)#, mol_graph.smiles_batch[i],\
												#mol_graph.ic_batch[i], draw)
				mol_vecs.append(mol_vec)
				node_scores.append(scores)

		mol_vecs = torch.stack(mol_vecs, dim = 0)
		return mol_vecs, node_scores


	def cross_readout(self, atom1_h, a1_scope, atom2_h, a2_scope, mol1_graph, mol2_graph):
		mol1_vecs, mol2_vecs = [], []

		# node-level cross attention with updates per node emb
		for (a1_start, a1_size), (a2_start, a2_size) in zip(a1_scope, a2_scope):
			if a1_size == 0:
				mol1_vecs.append(self.cached_zero_vector)
			if a2_size == 0:
				mol2_vecs.append(self.cached_zero_vector)
			else:
				temp1_h = atom1_h.narrow(0, a1_start, a1_size) 	# atom emb matrix for G1
				temp2_h = atom2_h.narrow(0, a2_start, a2_size)  # atom emb matrix for G2

				# score atoms in G1 based on G2
				sim_matrix = torch.sigmoid(torch.matmul(temp1_h, temp2_h.T))

				## Sinkhorn normalization
				### marginal weights `a` and `b` on the nodes must sum to 1
				mol1_vec, a = self._readout(temp1_h) # type: ignore
				mol2_vec, b = self._readout(temp2_h) # type: ignore
				S = ot.sinkhorn(a, b, 1-sim_matrix, 1, stopThr=1e-4)  ##TODO: check reg parameter in OT
				temp1_hp = torch.matmul(S, temp2_h) # type: ignore
				temp2_hp = torch.matmul(S.T, temp1_h) # type: ignore

				mol1_vecp, _ = self._readout(temp1_h + temp1_hp) # type: ignore
				mol2_vecp, _ = self._readout(temp2_h + temp2_hp) # type: ignore
				if self.agg_emb == 'self':
					mol1_vecs.append(mol1_vecp)
					mol2_vecs.append(mol2_vecp)
				elif self.agg_emb == 'concat':
					mol1_vecs.append(torch.concat((mol1_vec, mol1_vecp))) # type: ignore
					mol2_vecs.append(torch.concat((mol2_vec, mol2_vecp))) # type: ignore

		mol1_vecs = torch.stack(mol1_vecs, dim = 0)
		mol2_vecs = torch.stack(mol2_vecs, dim = 0)
		return mol1_vecs, mol2_vecs


	def _forward(self, mol_graph, features=None, draw=False, node_featurizer=False):

		f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
		f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device),\
							a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

		a2a = None
		if self.atom_messages:
			a2a = mol_graph.get_a2a().to(self.device)

		# Input
		if self.atom_messages:
			input = self.W_i(f_atoms)  # num_atoms x hidden_dim
		else:
			input = self.W_i(f_bonds)  # num_bonds x hidden_dim

		message = self.activation(input)
		message_timesteps = [message]

		for step in range(self.steps - 1):
			aggregated_message = self.aggregate(message, a2b, b2a, b2revb, a2a, f_bonds)
			message = self.update(input, aggregated_message)
			message_timesteps.append(message)

		if self.pooling == 'hier-sum':
			message_timesteps = torch.stack(message_timesteps, dim=0)
			message = message_timesteps.sum(dim=0)
		if 'hier-cat' in self.pooling:
			message = torch.concat(message_timesteps, dim=1) # type: ignore

		a2x = a2a if self.atom_messages else a2b
		incoming_a_message = index_select_ND(message, a2x) # type: ignore
		a_message = incoming_a_message.sum(dim=1)
		a_input = torch.cat([f_atoms, a_message], dim=1)
		atom_h = self.activation(self.W_a(a_input))
		atom_h = self.dropout_layer(atom_h)

		# if featurizer is set, output node embedding matrix
		if node_featurizer: #self.pooling == 'x-attention':
			return atom_h, a_scope
			#return atom_h_timesteps[-1], a_scope

		#readout
		mol_vecs, node_scores = self.readout(atom_h, a_scope, mol_graph)

		if self.feature_gen:
			features_batch = torch.from_numpy(np.stack(features)).float().to(self.device) # type: ignore
			mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)

		return mol_vecs, node_scores


	def forward(self, mol1_graph, features1=None, mol2_graph=None, features2=None):

		# when mol2_graph is None, we are using ranknet framework
		if mol2_graph is None:
			mol_vecs, _ = self._forward(mol1_graph, features1)
			return mol_vecs

		atom1_h, a1_scope = self._forward(mol1_graph, features1, node_featurizer=True)
		atom2_h, a2_scope = self._forward(mol2_graph, features2, node_featurizer=True)

		return self.cross_readout(atom1_h, a1_scope, atom2_h, a2_scope, mol1_graph, mol2_graph)


