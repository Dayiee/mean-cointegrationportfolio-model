from Leybourne import Leybourne
import numpy as np
import pandas as pd

import utils

class Simulation(object):
    def __init__(self, *args, **kwargs):
        #TODO
        pass

    def generate_portfolio(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()

class PeriodicSimulation(object):
    def __init__(self, periods, portfolio_model):
        self.periods = periods
        self.portfolio_model = portfolio_model

    def run(self, **params):
        records = []
        for i,period in enumerate(self.periods):
            utils.logPrint("Simulating on the %d period."%(i+1))
            portfolio = self.portfolio_model.fit(**period, **params)
            gain      = self.portfolio_model.eval(**period)
            records.append({**portfolio, **gain})
        return records


class StockembSimulation(PeriodicSimulation):
    def __init__(self, *args, **kwargs):
        #TODO
        pass


class CointegSimulation(PeriodicSimulation):
    def __init__(self, *args, **kwargs):
        #TODO
        pass




