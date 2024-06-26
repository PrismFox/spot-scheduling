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
Licensed under the Apache License, Version 2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file implements the cost models for both server provisioning and SLA violations.
"""

import cvxpy as cvx
from cvxpy import *
import numpy as np
import copy
#from expression import Expression
import data_management 
import datetime as dt
import logging

__all__ = ['HcostModelServers','TcostModelServers']

dm=data_management.data_management()
#logging.basicConfig(level=logging.DEBUG, filename='simulation.log', filemode='w')

class BaseCost(cvx.Expression):

    def __init__(self, *args, **kwargs):
        self.gamma = 1.  # it is changed by gamma * BaseCost()
        print('init', type(self))
        self.args = args
        

    def weight_expr(self, t, w_plus, z, value):
        cost, constr = self._estimate(t, w_plus, z, value)
        return self.gamma * cost, constr

    def weight_expr_ahead(self, tau, w_plus, z, value,LA):
        cost, constr = self._estimate_ahead(tau, w_plus, z, value,LA)
        return self.gamma * cost, constr

    def __mul__(self, other):
        """Read the gamma parameter as a multiplication."""
        newobj = copy.copy(self)
        newobj.gamma *= other
        return newobj

    def __rmul__(self, other):
        """Read the gamma parameter as a multiplication."""
        return self.__mul__(other)    

class DistanceLatencyModel(BaseCost):
    """A model for costs caused by distance-based latency.
    """

    def __init__(self, regional_arrival_rate, instance_capacity, distance, d_weight):
        dm.null_checker(regional_arrival_rate)
        self.Lambda_r = regional_arrival_rate
        dm.null_checker(instance_capacity)
        self.r = instance_capacity
        self.distance = distance # [L_s, L_r]
        self.d_weight = d_weight
        super().__init__(regional_arrival_rate, instance_capacity)

    def _estimate_ahead(self, tau, w_plus, n, value,LA):
        return self._estimate(tau, w_plus, n, value,LA)

    def _estimate(self, t, w_plus, n, value,LA):
        consts = []

        regional_load = cvx.Variable((n.shape[0] * n.shape[1], dm.time_locator(self.Lambda_r, t).shape[0]), integer = True)
        #share_of_regional_requests = cvx.Variable((n.shape[0] * n.shape[1], dm.time_locator(self.Lambda_r, t).shape[0]), nonneg = True) # [L_s * i, L_r]
        #regional_load = cvx.multiply(share_of_regional_requests, np.expand_dims(dm.time_locator(self.Lambda_r, t), axis=0))
        consts.append(cvx.sum(regional_load, axis = 0) == dm.time_locator(self.Lambda_r, t))
        consts.append(regional_load >= 0)
        consts.append(regional_load <= np.expand_dims(dm.time_locator(self.Lambda_r, t), axis=0))
        #consts.append(cvx.sum(share_of_regional_requests, axis=0) >= 1)
        #consts.append(cvx.sum(share_of_regional_requests, axis=0) <= 1.25)
        #consts.append(share_of_regional_requests >= 0)
        #consts.append(share_of_regional_requests <= 1)
        consts.append(cvx.sum(regional_load, axis=1) <= cvx.reshape(cvx.multiply(self.r, n), n.shape[0] * n.shape[1]))
        return cvx.sum(cvx.multiply(self.d_weight, cvx.multiply(np.repeat(self.distance, n.shape[1]), cvx.sum(regional_load, axis=1)))), consts
    
    def value_expr(self, t, h_plus, u,LA): #TODO
        return 1


class RegionBasedModelServers(BaseCost):
    """A model for SLA violation costs.
    """

    def __init__(self, penalty,L, pricePerReq,probFail,arrivalRate,oracle=True):
        dm.null_checker(probFail)
        self.failure = probFail
        dm.null_checker(arrivalRate)
        self.Lambda = arrivalRate
        self.penalty=penalty
        self.L=L
        self.oracle=oracle
        self.pricePerReq = pricePerReq
        super().__init__(penalty, L, pricePerReq,probFail,arrivalRate,oracle)
        
    def _estimate(self, t, w_plus, n, value,LA):
        """Estimate SLA violation costs.

        Args:
          t: time of estimate
          wplus: holdings
          tau: time to estimate (default=t)
        """
        constr = []
        xyz=0

        try:
            if LA==1: 
                third_term = dm.time_locator(self.failure, t).T.multiply(dm.time_locator(self.Lambda, t).multiply(self.L)).T#.tolist()[0])
                second_term = dm.time_locator(self.pricePerReq,t).T.multiply(dm.time_locator(self.Lambda, t)).T#.tolist()[0])    #Provisioning cost

            else:
                third_term = dm.time_locator(self.failure, t+dt.timedelta(hours=1)).T.multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).multiply(self.L)).T#.tolist()[0])
                second_term = dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)).T.multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1))).T#.tolist()[0])    #Provisioning cost
            third_term*=self.penalty
            third_term+=second_term
            third_term*=1
            logging.info(("Total", third_term.to_numpy()))
            xyz+=1
        except Exception as e:
                logging.error(("Exception in estimate", t, xyz))
                logging.error(str(e))
                raise e
        #if np.isscalar(third_term):
        if third_term.isna().values.any():
            constr += [n == 0]
            third_term = 0
            logging.error("SLA violation Costs converged to zero due to NANs")
        #else:  # it is a pd series
        #    no_trade = third_term.index[third_term.isnull()]
        #    logging.info(("no_trade", no_trade))
        #    third_term[no_trade] = 0
        #    logging.info("Third Term is not a scalar")
    
        #constr += [z[i] == 0 for i in second_term.index.get_indexer(second_term[second_term == 0].index)]
        served_requests = cvx.multiply(n, self.r)
        share_of_requests = (served_requests.T / dm.time_locator(self.Lambda, t)).T
    
        #b = cvx.minimum(z + 1 - 2*cvx.ceil(z), 0)
        #constr += [cvx.sum(cvx.multiply(b, second_term)) == 0]

        third_term[third_term<0]=0
        try:
            self.expression = cvxpy.multiply(third_term, cvx.abs(share_of_requests))
        except TypeError:
            self.expression = third_term.multiply(cvx.abs(w_plus))
            logging.error("Error when Multiplying Third Term with allocation")
        return cvxpy.sum(self.expression), []


    def _estimate_ahead(self, tau, w_plus, n, value,LA):
        return self._estimate(tau, w_plus, n, value,LA)

    def value_expr(self, t, h_plus, u,LA):
        u[u<=0]=0
        u_nc = u

        if LA==1: 
            if dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0]- dm.time_locator(self.Lambda, t).tolist()[0]>0:
                third_term = dm.time_locator(self.failure, t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).multiply(self.L).tolist()[0])\
                +dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0]- dm.time_locator(self.Lambda, t).tolist()[0]    #Late requests and failure cost
                second_term = dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0])    #Provisioning cost
                logging.debug(("Debugging",LA,t, third_term*u, second_term*u, sum(third_term*u),sum(second_term*u)))
                logging.info(("error in prediction",dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0]- dm.time_locator(self.Lambda, t).tolist()[0],"LLLL", self.L,'PPR',dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)),'FR',dm.time_locator(self.failure, t+dt.timedelta(hours=1))))
            else:
                third_term = dm.time_locator(self.failure, t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).multiply(self.L).tolist()[0])
                second_term = dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0])    #Provisioning cost
                logging.debug(("Debugging", LA,t, third_term*u, second_term*u, sum(third_term*u),sum(second_term*u),"LLLLL",self.L))


        else:
            third_term = dm.time_locator(self.failure, t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).multiply(self.L).tolist()[0])
            second_term = dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0])    #Provisioning cost
            logging.debug(("Debugging",LA,t, third_term*u, second_term*u, sum(third_term*u),sum(second_term*u), "LLLLL",self.L))
            logging.info(("error in prediction",dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0]- dm.time_locator(self.Lambda, t).tolist()[0],"LLLL", self.L,'PPR',dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)),'FR',dm.time_locator(self.failure, t+dt.timedelta(hours=1))))

        third_term*=self.penalty
        third_term+=second_term

        served_requests = u_nc*self.r
        share_of_requests = (served_requests.T / dm.time_locator(self.Lambda, t)).T
        self.last_cost = third_term*share_of_requests  #np.abs(u_nc) * self.penalty *(self.L*dm.time_locator(self.failure, t).multiply(

        return  sum(self.last_cost)

    def optimization_log(self, t):
        return self.expression

    def simulation_log(self, t):
        return self.last_cost

class HcostModelServers(BaseCost):
    """A model for SLA violation costs.
    """

    def __init__(self, penalty,L, pricePerReq,probFail,arrivalRate,instance_capacity,oracle=True):
        dm.null_checker(probFail)
        self.failure = probFail
        dm.null_checker(arrivalRate)
        self.Lambda = arrivalRate
        self.penalty=penalty
        self.L=L
        self.r=instance_capacity
        self.oracle=oracle
        self.pricePerReq = pricePerReq
        super().__init__(penalty, L, pricePerReq,probFail,arrivalRate,oracle)
        
            

    def _estimate(self, t, w_plus, n, value,LA):
        """Estimate SLA violation costs.

        Args:
          t: time of estimate
          wplus: holdings
          tau: time to estimate (default=t)
        """
        constr = []
        xyz=0

        try:
            if LA==1: 
                third_term = dm.time_locator(self.failure, t).T.multiply(dm.time_locator(self.Lambda, t).multiply(self.L)).T#.tolist()[0])
                second_term = dm.time_locator(self.pricePerReq,t).T.multiply(dm.time_locator(self.Lambda, t)).T#.tolist()[0])    #Provisioning cost

            else:
                third_term = dm.time_locator(self.failure, t+dt.timedelta(hours=1)).T.multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).multiply(self.L)).T#.tolist()[0])
                second_term = dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)).T.multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1))).T#.tolist()[0])    #Provisioning cost
            third_term*=self.penalty
            third_term+=second_term
            third_term*=1
            logging.info(("Total", third_term.to_numpy()))
            xyz+=1
        except Exception as e:
                logging.error(("Exception in estimate", t, xyz))
                logging.error(str(e))
                raise e
        #if np.isscalar(third_term):
        if third_term.isna().values.any():
            constr += [n == 0]
            third_term = 0
            logging.error("SLA violation Costs converged to zero due to NANs")
        #else:  # it is a pd series
        #    no_trade = third_term.index[third_term.isnull()]
        #    logging.info(("no_trade", no_trade))
        #    third_term[no_trade] = 0
        #    logging.info("Third Term is not a scalar")
    
        #constr += [z[i] == 0 for i in second_term.index.get_indexer(second_term[second_term == 0].index)]
        share_of_requests = cvx.multiply(n, (self.r.T / dm.time_locator(self.Lambda, t).values).T)
    
        #b = cvx.minimum(z + 1 - 2*cvx.ceil(z), 0)
        #constr += [cvx.sum(cvx.multiply(b, second_term)) == 0]

        third_term[third_term<0]=0
        try:
            self.expression = cvxpy.multiply(third_term, cvx.abs(share_of_requests))
        except TypeError:
            self.expression = third_term.multiply(cvx.abs(w_plus))
            logging.error("Error when Multiplying Third Term with allocation")
        return cvxpy.sum(self.expression), []


    def _estimate_ahead(self, tau, w_plus, n, value,LA):
        return self._estimate(tau, w_plus, n, value,LA)

    def value_expr(self, t, h_plus, u,LA):
        u[u<=0]=0
        u_nc = u

        if LA==1: 
            if dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0]- dm.time_locator(self.Lambda, t).tolist()[0]>0:
                third_term = dm.time_locator(self.failure, t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).multiply(self.L).tolist()[0])\
                +dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0]- dm.time_locator(self.Lambda, t).tolist()[0]    #Late requests and failure cost
                second_term = dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0])    #Provisioning cost
                logging.debug(("Debugging",LA,t, third_term*u, second_term*u, sum(third_term*u),sum(second_term*u)))
                logging.info(("error in prediction",dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0]- dm.time_locator(self.Lambda, t).tolist()[0],"LLLL", self.L,'PPR',dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)),'FR',dm.time_locator(self.failure, t+dt.timedelta(hours=1))))
            else:
                third_term = dm.time_locator(self.failure, t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).multiply(self.L).tolist()[0])
                second_term = dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0])    #Provisioning cost
                logging.debug(("Debugging", LA,t, third_term*u, second_term*u, sum(third_term*u),sum(second_term*u),"LLLLL",self.L))


        else:
            third_term = dm.time_locator(self.failure, t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).multiply(self.L).tolist()[0])
            second_term = dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0])    #Provisioning cost
            logging.debug(("Debugging",LA,t, third_term*u, second_term*u, sum(third_term*u),sum(second_term*u), "LLLLL",self.L))
            logging.info(("error in prediction",dm.time_locator(self.Lambda, t+dt.timedelta(hours=1)).tolist()[0]- dm.time_locator(self.Lambda, t).tolist()[0],"LLLL", self.L,'PPR',dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)),'FR',dm.time_locator(self.failure, t+dt.timedelta(hours=1))))

        third_term*=self.penalty
        third_term+=second_term

        served_requests = u_nc*self.r
        share_of_requests = served_requests / dm.time_locator(self.Lambda, t)
        self.last_cost = third_term*share_of_requests  #np.abs(u_nc) * self.penalty *(self.L*dm.time_locator(self.failure, t).multiply(

        return  sum(self.last_cost)

    def optimization_log(self, t):
        return self.expression

    def simulation_log(self, t):
        return self.last_cost


class TcostModelServers(BaseCost):
    """A model for server provisioning costs.
    """
    def __init__(self, arrival=200, pricePerReq=1.,oracle=True):
        MachinePrices=super(TcostModelServers, self).__init__()
        dm.null_checker(arrival)
        self.Lambda = arrival
        dm.null_checker(pricePerReq)
        self.pricePerReq = pricePerReq
        self.oracle=oracle
        super().__init__(arrival, pricePerReq,oracle)
        

    def _estimate(self, t, w_plus, z, value):
        try:
            z = z[z.index != self.cash_key]
            z = z.values
        except AttributeError:
            z = z[:-1]  # TODO fix when cvxpy pandas ready
    
        constr = []
        xyz=0
        try:
            second_term = dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t).tolist()[0])    #Provisioning cost
            xyz+=1
        except Exception as e:
                logging.error(e)

        if np.isscalar(second_term):
            if np.isnan(second_term):
                constr += [z == 0]
                second_term = 0
            else:  # it is a pd series
                    no_trade = second_term.index[second_term.isnull()]
                    second_term[no_trade] = 0
                    constr += [z[second_term.index.get_loc(tick)] == 0
                               for tick in no_trade]
        self.expression = second_term[:-1].multiply(cvx.abs(z))
        return sum(self.expression), []
        
    def value_expr(self, t, h_plus, u):
        '''u is the trades (not cash)
        hplus is the  current portfolio+the trades
        '''

        u_nc = u.iloc[:-1]
        self.tmp_tcosts = (np.abs(u_nc) * dm.time_locator(self.pricePerReq,t+dt.timedelta(hours=1)).multiply(dm.time_locator(self.Lambda, t).tolist()[0]))

        return self.tmp_tcosts.sum()

    def optimization_log(self, t):
        try:
            return self.expression.value
        except AttributeError:
            return np.nan

    def simulation_log(self, t):
        # TODO find another way
        return self.tmp_tcosts

    def _estimate_ahead(self, t, tau, w_plus, z, value):
        """Returns the estimate at time t of tcost at time tau.
        Gets called in the Base_cost class
        """
        return self._estimate(t, w_plus, z, value)

    def est_period(self, t, tau_start, tau_end, w_plus, z, value):
        """Returns the estimate at time t of tcost over given period.
        """
        K = (tau_end - tau_start).hours   #Changed to hours instead of days.
        tcost, constr = self.eight_expr(t, None, z / K, value)
        return tcost * K, constr
