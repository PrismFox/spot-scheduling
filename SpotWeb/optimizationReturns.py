"""
Created on May 29, 2019

@author: Ahmed Ali-Eldin
SpotWeb Copyright (c) 2019 The SpotWeb team, led by Dr. Ahmed Ali-Eldin and Prof. Prashant Shenoy at UMass Amherst. 
All Rights Reserved.
# 
# This product is licensed to you under the Apache 2.0 license (the "License").
# You may not use this product except in compliance with the Apache 2.0
# License.
# 
# This product may include a number of subcomponents with separate copyright
# notices and license terms. Your use of these subcomponents is subject to the
# terms and conditions of the subcomponent's license, as noted in the LICENSE
# file.

The code is based the code of CVXPortfolio by Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.
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


import cvxpy as cvx
#from expression import Expression
import data_management
#from pymc3.examples.custom_dists import alpha

__all__ = ['ReturnsForecast', 'MPOReturnsForecast',
           'MultipleReturnsForecasts']

dm=data_management.data_management()

class BaseReturnsModel(cvx.Expression):
    pass


class ReturnsForecast(BaseReturnsModel):
    """A single return forecast.

    Attributes:
      alpha_data: A dataframe of return estimates.
      delta_data: A confidence interval around the estimates.
      half_life: Number of days for alpha auto-correlation to halve.
    """

    def __init__(self, returns, delta=0., gamma_decay=None, name=None):
        dm.null_checker(returns)
        self.returns = returns
        dm.null_checker(delta)
        self.delta = delta
        self.gamma_decay = gamma_decay
        self.name = name

    def weight_expr(self, t, wplus, z=None, v=None):
        """Returns the estimated alpha.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = cvx.multiply(
            dm.time_locator(self.returns, t, as_numpy=True), wplus)
        alpha -= cvx.multiply(
            dm.time_locator(self.delta, t, as_numpy=True), cvx.abs(wplus))
        return cvx.sum(alpha)

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """

        alpha = self.weight_expr(t, wplus)
        if tau > t and self.gamma_decay is not None:
            alpha *= (tau - t).days**(-self.gamma_decay)
        return alpha


class MPOReturnsForecast(BaseReturnsModel):
    """A single alpha estimation.

    Attributes:
      alpha_data: A dict of series of return estimates.
    """

    def __init__(self, alpha_data):
        self.alpha_data = alpha_data

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """

        return self.alpha_data[(t, tau)].values.T * wplus


    def weight_expr(self, t, wplus, z=None, v=None):
            print("WTF, not implemented ")
            pass


class MultipleReturnsForecasts(BaseReturnsModel):
    """A weighted combination of alpha sources.

    Attributes:
      alpha_sources: a list of alpha sources.
      weights: An array of weights for the alpha sources.
    """

    def __init__(self, alpha_sources, weights):
        self.alpha_sources = alpha_sources
        self.weights = weights

    def weight_expr(self, t, wplus, z=None, v=None):
        """Returns the estimated alpha.

        Args:
            t: time estimate is made.
            wplus: An expression for holdings.
            tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = 0
        for idx, source in enumerate(self.alpha_sources):
            alpha += source.weight_expr(t, wplus) * self.weights[idx]
        return alpha

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = 0
        for idx, source in enumerate(self.alpha_sources):
            alpha += source.weight_expr_ahead(t,
                                              tau, wplus) * self.weights[idx]
        return alpha
