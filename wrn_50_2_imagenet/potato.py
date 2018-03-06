from future import *

from IPython.lib.deepreload import reload as dreload
import PIL, os, numpy as np, math, collections, threading, json, bcolz, random, scipy, cv2
import random, pandas as pd, pickle, sys, itertools, string, sys, re, datetime, time, shutil
import seaborn as sns, matplotlib
import IPython, graphviz, sklearn_pandas, sklearn, warnings
from abc import abstractmethod
from glob import glob, iglob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import chain
from functools import partial
from collections import Iterable, Counter, OrderedDict
from isoweek import Week
from pandas_summary import DataFrameSummary
from IPython.lib.display import FileLink
from PIL import Image, ImageEnhance, ImageOps
from sklearn import metrics, ensemble, preprocessing
from operator import itemgetter, attrgetter

from matplotlib import pyplot as plt, rcParams, animation
from ipywidgets import interact, interactive, fixed, widgets

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = "/dogscats/"

arch=wrn
sz=224
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz), bs=16) 
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.01, 2)

