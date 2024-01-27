"""

Created on Nov 28, 2018

@author: Ahmed Ali-Eldin
SpotWeb Copyright (c) 2019 The SpotWeb team, led by Ahmed Ali-Eldin and Prashant Shenoy at UMass Amherst.
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
Licensed under the Apache License, Version 2.0 

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file implements the MPO optimization policies.
"""

from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import logging
import cvxpy as cvx
from osqp import *
from optimizationCosts import BaseCost
from optimizationReturns import BaseReturnsModel
from optimizationConstraints import BaseConstraint
import data_management
import time
from datetime import datetime
dm=data_management.data_management()


__all__ = ['Hold', 'FixedTrade', 'PeriodicRebalance', 'AdaptiveRebalance',
           'SinglePeriodOpt', 'MultiPeriodOpt', 'ProportionalTrade',
           'RankAndLongShort']


class BasePolicy(object):
    """ Base class for a trading policy. """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.costs = []
        self.constraints = []
        
    @abstractmethod
    def get_trades(self, portfolio, t=datetime.utcnow().date()):
        """Trades list given current portfolio and time t.
        """
        return NotImplemented

    def _nulltrade(self, portfolio):
        return pd.Series(index=portfolio.index, data=0.)

    def get_rounded_trades(self, portfolio, prices, t):
        """Get trades vector as number of shares, rounded to integers."""
        return np.round(self.get_trades(portfolio,
                                        t) / dm.time_locator(prices, t))[:-1]


class SinglePeriodOpt(BasePolicy):
    """Single-period optimization policy.

    Implements the model developed in chapter 4 of our paper
    https://stanford.edu/~boyd/papers/cvx_portfolio.html
    """

    def __init__(self,  costs, constraints,return_forecast=None, solver=None,
                 solver_opts=None):
        super(SinglePeriodOpt, self).__init__()

        for cost in costs:
            print(type(cost), type(BaseCost))
            assert isinstance(cost, BaseCost)
            self.costs.append(cost)

        for constraint in constraints:
            logging.info(type(constraint))
            assert isinstance(constraint, BaseConstraint)
            self.constraints.append(constraint)

        self.solver = solver
        self.solver_opts = {} if solver_opts is None else solver_opts

    def get_trades(self, portfolio, t=None):
        """
        Get optimal trade vector for given portfolio at time t.

        Parameters
        ----------
        portfolio : pd.Series
            Current portfolio vector.
        t : pd.timestamp
            Timestamp for the optimization.
        """

        if t is None:
            t = datetime.utcnow().date()

        value = sum(portfolio)
        w = portfolio / value
        z = cvx.Variable(w.size)  # TODO pass index
        wplus = w.values + z

        if isinstance(self.return_forecast, BaseReturnsModel):
            alpha_term = self.return_forecast.weight_expr(t, wplus)
        else:
            alpha_term = cvx.sum(cvx.multiply(
                dm.time_locator(self.return_forecast, t, as_numpy=True), wplus))

        assert(alpha_term.is_concave())

        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, wplus, z, value)
            costs.append(cost_expr)
            constraints += const_expr

        constraints += [item for item in (con.weight_expr(t, wplus, z, value)
                                          for con in self.constraints)]

        for el in costs:
            assert (el.is_convex())

        for el in constraints:
            assert (el.is_dcp())

        self.prob = cvx.Problem(
            cvx.Minimize(alpha_term - sum(costs)),
            [cvx.sum(z) == 0] + constraints)
        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)

            if self.prob.status == 'unbounded':
                logging.error(
                    'The problem is unbounded. Defaulting to no trades')
                return self._nulltrade(portfolio)

            if self.prob.status == 'infeasible':
                logging.error(
                    'The problem is infeasible. Defaulting to no trades')
                return self._nulltrade(portfolio)

            return pd.Series(index=portfolio.index, data=(z.value * value))
        except cvx.SolverError:
            logging.error(
                'The solver %s failed. Defaulting to no trades' % self.solver)
            return self._nulltrade(portfolio)


class MultiPeriodOpt(SinglePeriodOpt):

    def __init__(self, trading_times,
                 lookahead_periods=5, *args, **kwargs):
        """
        trading_times: list, all times at which get_trades will be called
        lookahead_periods: int or None. if None uses all remaining periods
        """
        self.lookahead_periods = lookahead_periods
        self.trading_times = trading_times
        self.prevz=[]
        with open("SimO.out","w") as foo:
            pass
        super(MultiPeriodOpt, self).__init__(*args, **kwargs)

    def get_trades(self, portfolio, t=datetime.utcnow().date()):

        value = sum(portfolio)
        assert (value > 0)
        w = cvx.Constant(portfolio.values / value)

        prob_arr = []
        z_vars = []
        t1=time.time()
        for tau in \
                self.trading_times[self.trading_times.index(t):
                                   self.trading_times.index(t) +
                                   self.lookahead_periods]:
            z = cvx.Variable(w.shape)
            #wplus = z - w
            wplus = z
            #r_adj = cvx.Variable(w.shape)
            self.prevz=wplus
            #self.prevz=w

            costs, constr = [], []
            for cost in self.costs:
                cost_expr, const_expr = cost.weight_expr_ahead(tau, wplus, z, value,self.lookahead_periods)
                costs.append(cost_expr)
                constr += const_expr

            obj = cvx.sum(costs)
            constr += [cvx.sum(z) >= 0]
            constr += [con.weight_expr(t, wplus, z, value)
                       for con in self.constraints]
            prob = cvx.Problem(cvx.Minimize(obj), constr)
            prob_arr.append(prob)
            z_vars.append(z)
            #w += wplus
            w=wplus

        sumprob=cvx.sum(prob_arr)
        maxiters=800000000
        solution=sumprob.solve()#solver=cvx.SCS, max_iters=maxiters, verbose=True)
        t2=time.time()
        with open("SimO.out","a") as foo:
            if z.value is None:
                logging.error(f"Solver failed, with only high precision but {maxiters} iterations, returning previous value")
                logging.info("input to the solver", w.shape,t,z.value, t2-t1,self.lookahead_periods)
                print(w.size,t, None, z.value, t2-t1, self.lookahead_periods,sumprob.status,sumprob.value,file=foo)
                return pd.Series(index=portfolio.index,data=self.prevz)
            else:
                xx=z_vars[0].value # * value
                xx=xx.tolist()
                #xx=[ll[0] for ll in xx]
                logging.info("input to the solver", w.size,t,xx, t2-t1,self.lookahead_periods)
                self.prevz=xx
                print(w.size,t, sum(z.value),z.value.tolist(), t2-t1, self.lookahead_periods,sumprob.status,sumprob.value,file=foo)
                return pd.Series(index=portfolio.index,data=xx)
