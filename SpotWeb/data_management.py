"""
This file is used in SpotWeb as is. It is part of the CVXportfolio software.

Copyright 2017 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pandas as pd

class data_management():	
	
	def __init__(self):
		self.me = 1. 

	def null_checker(self, obj):
		"""Check if obj contains NaN."""
		#if (isinstance(obj, pd.xarray) or
		if (isinstance(obj, pd.DataFrame) or
			isinstance(obj, pd.Series)):
			if np.any(pd.isnull(obj)):
				raise ValueError('Data object contains NaN values', obj)
			elif np.isscalar(obj):
				if np.isnan(obj):
					raise ValueError('Data object contains NaN values', obj)
				else:
					raise TypeError('Data object can only be scalar or Pandas.')


	def non_null_data_args(self, f):
		def new_f(*args, **kwds):
			for el in args:
				self.null_checker(el)
			for el in kwds.values():
				self.null_checker(el)
			return f(*args, **kwds)
		return new_f


	def time_matrix_locator(self, obj, t, as_numpy=False):
		"""Retrieve a matrix from a time indexed Panel, or a static DataFrame."""
		if isinstance(obj, pd.Panel):
			res = obj.iloc[obj.axes[0].get_indexer([t], method='pad')[0]]
			return res.values if as_numpy else res
		elif isinstance(obj, pd.DataFrame):
			return obj.values if as_numpy else obj
		else:  # obj not pandas
			raise TypeError('Expected Pandas DataFrame or Panel, got:', obj)


	def time_locator(self, obj, t, as_numpy=False):
		"""Retrieve data from a time indexed DF, or a Series or scalar."""
		if isinstance(obj, pd.DataFrame):
			index = obj.axes[0]
			if isinstance(index, pd.MultiIndex):
				res = obj.loc[t, :]
			else:
				res = obj.iloc[obj.axes[0].get_indexer([t], method='pad')[0]]
			return res.values if as_numpy else res
		elif isinstance(obj, pd.Series):
			index = obj.index
			if isinstance(index, pd.MultiIndex):
				res = obj.loc[t]
			else:
				res = obj
			return res.values if as_numpy else res
		elif np.isscalar(obj):
			return obj
		else:
			raise TypeError(
		    	'Expected Pandas DataFrame, Series, or scalar. Got:', obj)
