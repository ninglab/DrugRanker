#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : ae.py
# Author            : Vishal Dey <dey.78@osu.edu>
# Date              : Tue 22 Feb 2022 21:50:16
# Last Modified Date: Tue 22 Feb 2022 21:50:16
# Last Modified By  : Vishal Dey <dey.78@osu.edu>

import numpy as np
import torch.nn as nn


class AE(nn.Module):
	def __init__(self, args):
		super().__init__()
		hidden_size = 4096
		self.encoder = nn.Sequential(
						nn.Linear(args.ae_in_size, hidden_size),
						nn.ReLU(),
						nn.Dropout(args.dropout),
						nn.Linear(hidden_size, hidden_size//4),
						nn.ReLU(),
						nn.Dropout(args.dropout),
						nn.Linear(hidden_size//4, args.ae_out_size),
						)

		self.decoder = nn.Sequential(
						nn.Linear(args.ae_out_size, hidden_size//4),
						nn.ReLU(),
						nn.Dropout(args.dropout),
						nn.Linear(hidden_size//4, hidden_size),
						nn.ReLU(),
						nn.Dropout(args.dropout),
						nn.Linear(hidden_size, args.ae_in_size),
						)


	def forward(self, x, use_encoder_only=False):
		if use_encoder_only:
			return self.encoder(x)

		enc = self.encoder(x)
		dec = self.decoder(enc)
		return dec
