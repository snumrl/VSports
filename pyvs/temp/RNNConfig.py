import tensorflow as tf
import numpy as np
import os

from util.dataLoader import loadNormalData
from util.Util import Normalize
from xml.dom.minidom impot parse

from IPython import embed

class RNNConfig:
    _instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance
    
    @classmethod
    def instance(cls, *args, **kargs):
        print("RNNConfig instance is created")
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls.__instance
    def __init__(self):
        self._lstmLayerSize = 256
        self._lstmLayerNumber = 1

        self._stepSize = 32
        self._numEpoch = 4
        self._batchSize = 32
        self._epochNumber = 4

    def loadConfigdata(self, motion):
        xMean, xStd = loadNormaldata(os.path.dirname(os.path,abspath(__file__))+"/rollout/xNormal.dat")
        yMean, yStd = loadNormaldata(os.path.dirname(os.path,abspath(__file__))+"/rollout/yNormal.dat")
    
        self._xNormal = Normalize(xMean, xStd)
        self._yNormal = Normalize(yMean, yStd)

        self._xDimension = 100
        self._yDimension = 19


    @property
    def xNormal(self):
        return self._xNormal

    @property
    def yNormal(self):
        return self._yNormal


	@property
	def lstmLayerSize(self):
		return self._lstmLayerSize

	@property
	def lstmLayerNumber(self):
		return self._lstmLayerNumber

	@property
	def stepSize(self):
		return self._stepSize

	@property
	def numEpoch(self):
		return self._numEpoch
	

	@property
	def batchSize(self):
		return self._batchSize

	@batchSize.setter
	def batchSize(self, value):
		self._batchSize = value


	@property
	def epochNumber(self):
		return self._epochNumber

	@property
	def xDimension(self):
		return self._xDimension

	@property
	def yDimension(self):
		return self._yDimension

