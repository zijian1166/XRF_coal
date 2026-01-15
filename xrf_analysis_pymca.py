#!/usr/bin/env python3
"""
python3 xrf_analysis_pymca.py \
  --path 煤炭压片_006-3_2.mca \
  --plot spectrum_pymca.png
"""
import argparse
from io import StringIO
from pathlib import Path
import sys

import numpy as np

from PyMca5.PyMcaIO import ConfigDict
from PyMca5.PyMcaIO import specfilewrapper as specfile
from PyMca5.PyMcaPhysics.xrf import ClassMcaTheory
DEFAULT_CFG = """

[xrfmc]
program = None

[xrfmc.setup]
p_polarisation = 0.995
source_sample_distance = 100.0
slit_distance = 100.0
slit_width_x = 0.005
slit_width_y = 0.005
source_size_x = 0.0005
source_size_y = 0.0001
source_diverg_x = 0.0001
source_diverg_y = 0.0001
nmax_interaction = 4
layer = 1
collimator_height = 0.0
collimator_diameter = 0.0
histories = 100000

[attenuators]
kapton = 0, -, 0.0, 0.0, 1.0
atmosphere = 0, -, 0.0, 0.0, 1.0
deadlayer = 0, Si1, 2.33, 0.002, 1.0
absorber = 0, -, 0.0, 0.0, 1.0
window = 0, -, 0.0, 0.0, 1.0
contact = 0, Au1, 19.37, 1e-06, 1.0
Filter 6 = 0, -, 0.0, 0.0, 1.0
Filter 7 = 0, -, 0.0, 0.0, 1.0
BeamFilter0 = 0, -, 0.0, 0.0, 1.0
BeamFilter1 = 0, -, 0.0, 0.0, 1.0
Detector = 0, Si1, 2.33, 0.5, 1.0
Matrix = 0, MULTILAYER, 0.0, 0.0, 45.0, 45.0, 0, 90.0

[peaks]
Fe = K
Ca = K
Na = K
Mg = K
K = K
Al = K
Si = K
P = K
S = K
Ti = K
V = K
Mn = K

[fit]
deltaonepeak = 0.01
strategy = SingleLayerStrategy
strategyflag = 0
fitfunction = 0
continuum = 0
fitweight = 1
stripalgorithm = 1
linpolorder = 5
exppolorder = 6
stripconstant = 1.0
snipwidth = 30
stripiterations = 20000
stripwidth = 25
stripfilterwidth = 1
stripanchorsflag = 0
maxiter = 10
deltachi = 0.001
xmin = 20
xmax = 880
linearfitflag = 0
use_limit = 0
stripflag = 1
escapeflag = 1
sumflag = 0
scatterflag = 1
hypermetflag = 1
stripanchorslist = 0, 0, 0, 0
energy = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
energyweight = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
energyflag = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
energyscatter = 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

[concentrations]
usematrix = 0
useattenuators = 1
usemultilayersecondary = 0
usexrfmc = 0
mmolarflag = 0
flux = 10000000000.0
time = 1.0
area = 30.0
distance = 10.0
reference = Auto
useautotime = 0

[detector]
detele = Si
nthreshold = 4
zero = -0.03183
deltazero = 0.1
fixedzero = 0
gain = 0.009194
deltagain = 0.001
fixedgain = 0
noise = 0.1
deltanoise = 0.05
fixednoise = 0
fano = 0.114
deltafano = 0.114
fixedfano = 0
sum = 1e-08
deltasum = 1e-08
fixedsum = 0
ignoreinputcalibration = 0

[peakshape]
st_arearatio = 0.05
deltast_arearatio = 0.03
fixedst_arearatio = 0
st_sloperatio = 0.5
deltast_sloperatio = 0.49
fixedst_sloperatio = 0
lt_arearatio = 0.02
deltalt_arearatio = 0.015
fixedlt_arearatio = 0
lt_sloperatio = 10.0
deltalt_sloperatio = 7.0
fixedlt_sloperatio = 0
step_heightratio = 0.0001
deltastep_heightratio = 5e-05
fixedstep_heightratio = 0
eta_factor = 0.02
deltaeta_factor = 0.02
fixedeta_factor = 0

[materials]

[materials.Air]
Comment = Dry Air (Near sea level) density=0.001204790 g/cm3
Thickness = 1.0
Density = 0.0012048
CompoundFraction = 0.000124, 0.75527, 0.23178, 0.012827, 3.2e-06
CompoundList = C1, N1, O1, Ar1, Kr1

[materials.Goethite]
Comment = Mineral FeO(OH) density from 3.3 to 4.3 density=4.3 g/cm3
CompoundList = Fe1O2H1
CompoundFraction = 1.0
Density = 4.3
Thickness = 0.1

[materials.Mylar]
Comment = Mylar (Polyethylene Terephthalate) density=1.40 g/cm3
Density = 1.4
CompoundFraction = 0.041959, 0.625017, 0.333025
CompoundList = H1, C1, O1

[materials.Kapton]
Comment = Kapton 100 HN 25 micron density=1.42 g/cm3
Density = 1.42
Thickness = 0.0025
CompoundFraction = 0.628772, 0.066659, 0.304569
CompoundList = C1, N1, O1

[materials.Teflon]
Comment = Teflon density=2.2 g/cm3
Density = 2.2
CompoundFraction = 0.240183, 0.759817
CompoundList = C1, F1

[materials.Viton]
Comment = Viton Fluoroelastomer density=1.8 g/cm3
Density = 1.8
CompoundFraction = 0.009417, 0.280555, 0.710028
CompoundList = H1, C1, F1

[materials.Water]
Comment = Water density=1.0 g/cm3
CompoundFraction = 1.0
CompoundList = H2O1
Density = 1.0

[materials.Gold]
Comment = Gold
CompoundFraction = 1.0
CompoundList = Au
Density = 19.37
Thickness = 1e-06

[userattenuators]

[userattenuators.UserFilter0]
use = 0
name = UserFilter0
comment = 
energy = 0.0, 0.001
transmission = 0.0, 1.0

[userattenuators.UserFilter1]
use = 0
name = UserFilter1
comment = 
energy = 0.0, 0.001
transmission = 0.0, 1.0

[multilayer]
Layer0 = 0, -, 0.0, 0.0
Layer1 = 0, -, 0.0, 0.0
Layer2 = 0, -, 0.0, 0.0
Layer3 = 0, -, 0.0, 0.0
Layer4 = 0, -, 0.0, 0.0
Layer5 = 0, -, 0.0, 0.0
Layer6 = 0, -, 0.0, 0.0
Layer7 = 0, -, 0.0, 0.0
Layer8 = 0, -, 0.0, 0.0
Layer9 = 0, -, 0.0, 0.0

[tube]
transmission = 0
voltage = 30.0
anode = Ag
anodethickness = 0.0002
anodedensity = 10.5
window = Be
windowthickness = 0.0125
windowdensity = 1.848
filter1 = He
filter1thickness = 0.0
filter1density = 0.000179
alphax = 90.0
alphae = 90.0
deltaplotting = 0.1
"""


def read_pmca_header(path):
    header = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        if s == "<<DATA>>":
            break
        if " - " in s:
            key, val = s.split(" - ", 1)
            header[key.strip()] = val.strip()
    return header


def load_default_config():
    config = ConfigDict.ConfigDict()
    config.readfp(StringIO(DEFAULT_CFG))
    return config


def parse_calib(calib_text):
    parts = [p.strip() for p in calib_text.split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("Need at least zero,gain")
    zero = float(parts[0])
    gain = float(parts[1])
    return zero, gain


def find_first_mca(path):
    sf = specfile.Specfile(path)
    for scan in range(len(sf)):
        if sf[scan].nbmca():
            return sf[scan].mca(1)
    return None


def update_peaks(config, elements):
    peaks = {}
    for el in elements:
        peaks[el] = "K"
    config["peaks"] = peaks


def extract_param_map(result):
    params = result.get("parameters", [])
    fitted = result.get("fittedpar", [])
    if len(params) != len(fitted):
        return {}
    return {name: value for name, value in zip(params, fitted)}


def write_group_csv(path, result):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "fitarea", "sigmaarea", "mcaarea", "statistics"])
        for group in result.get("groups", []):
            g = result.get(group, {})
            w.writerow(
                [
                    group,
                    f"{g.get('fitarea', 0.0):.6g}",
                    f"{g.get('sigmaarea', 0.0):.6g}",
                    f"{g.get('mcaarea', 0.0):.6g}",
                    f"{g.get('statistics', 0.0):.6g}",
                ]
            )


def main():
    parser = argparse.ArgumentParser(description="Analyze .mca using PyMca")
    parser.add_argument(
        "--path",
        default="Data/Original_data/煤炭压片_004_1.mca",
        help="Path to .mca file",
    )
    parser.add_argument(
        "--config",
        default="",
        help="PyMca config file path (.cfg). If omitted, uses PyMca test config.",
    )
    parser.add_argument(
        "--calib",
        default="",
        help="Energy calibration zero,gain in keV/channel (overrides config)",
    )
    parser.add_argument(
        "--fix-calib",
        action="store_true",
        help="Fix zero/gain during fit when --calib is provided",
    )
    parser.add_argument(
        "--xmin",
        type=int,
        default=None,
        help="Fit range lower bound (channel index)",
    )
    parser.add_argument(
        "--xmax",
        type=int,
        default=None,
        help="Fit range upper bound (channel index)",
    )
    parser.add_argument(
        "--elements",
        default="",
        help="Comma-separated element list for K lines (overrides config peaks)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Top N groups to display by fitted area",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV output for fitted group areas",
    )
    parser.add_argument(
        "--plot",
        default="",
        help="Optional plot output path (PNG)",
    )
    parser.add_argument(
        "--plot-elements",
        default="",
        help="Optional plot output path for colored element areas (PNG)",
    )
    args = parser.parse_args()

    y = find_first_mca(args.path)
    if y is None:
        print("No MCA data found in file.", file=sys.stderr)
        return 1

    header = read_pmca_header(args.path)
    live_time = float(header.get("LIVE_TIME", "0") or 0.0)

    x = np.arange(y.size, dtype=float)

    if args.config:
        config = ConfigDict.ConfigDict()
        config.read(args.config)
    else:
        config = load_default_config()

    if args.elements:
        elements = [e.strip() for e in args.elements.split(",") if e.strip()]
        if elements:
            update_peaks(config, elements)

    if args.calib:
        zero, gain = parse_calib(args.calib)
        config["detector"]["zero"] = zero
        config["detector"]["gain"] = gain
        if args.fix_calib:
            config["detector"]["fixedzero"] = 1
            config["detector"]["fixedgain"] = 1

    xmin = 0 if args.xmin is None else max(0, args.xmin)
    xmax = int(y.size - 1 if args.xmax is None else min(y.size - 1, args.xmax))
    if xmin >= xmax:
        xmin = 0
        xmax = int(y.size - 1)
    config["fit"]["xmin"] = xmin
    config["fit"]["xmax"] = xmax

    mcaFit = ClassMcaTheory.ClassMcaTheory()
    config = mcaFit.configure(config)
    if live_time > 0:
        mcaFit.setData(x, y, xmin=xmin, xmax=xmax, time=live_time)
    else:
        mcaFit.setData(x, y, xmin=xmin, xmax=xmax)

    mcaFit.estimate()
    fitresult, result = mcaFit.startfit(digest=1)

    param_map = extract_param_map(result)
    print("PyMca fit complete")
    print(f"Channels: {y.size}")
    if live_time > 0:
        print(f"Live time: {live_time:.3f} s")
    if "chisq" in result:
        print(f"Chi-square: {result['chisq']:.6g}")
    if "niter" in result:
        print(f"Iterations: {result['niter']}")
    if "lastdeltachi" in result:
        print(f"Last delta chi: {result['lastdeltachi']:.6g}")
    if "Zero" in param_map and "Gain" in param_map:
        print(f"Fitted Zero/Gain: {param_map['Zero']:.6g}, {param_map['Gain']:.6g}")

    groups = result.get("groups", [])
    rows = []
    for group in groups:
        g = result.get(group, {})
        rows.append((group, g.get("fitarea", 0.0), g.get("sigmaarea", 0.0), g.get("mcaarea", 0.0)))
    rows.sort(key=lambda r: r[1], reverse=True)
    print("\nTop fitted groups (by fit area)")
    print("group\tfitarea\tsigmaarea\tmcaarea")
    for group, fitarea, sigmaarea, mcaarea in rows[: args.top]:
        print(f"{group}\t{fitarea:.6g}\t{sigmaarea:.6g}\t{mcaarea:.6g}")

    if args.csv:
        write_group_csv(args.csv, result)
        print(f"\nSaved group table to {args.csv}")

    ydata = result.get("ydata", y)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available, skip plotting.")
        else:
            xplot = result.get("energy", x)
            plt.figure(figsize=(10, 4.5))
            plt.plot(xplot, ydata, lw=1.0, color="#2b2b2b", label="data")
            plt.plot(xplot, result.get("yfit", y), lw=1.0, color="#e67e22", label="fit")
            plt.plot(xplot, result.get("continuum", np.zeros_like(y)), lw=1.0, color="#7f8c8d", ls="--", label="continuum")
            max_y = float(np.max(ydata)) if ydata is not None and len(ydata) else 0.0
            min_label_height = 0.0
            for group in result.get("groups", []):
                y_group = result.get(f"y{group}")
                if y_group is None:
                    continue
                y_group = np.asarray(y_group)
                if y_group.size == 0:
                    continue
                idx = int(np.argmax(y_group))
                peak_y = float(y_group[idx])
                if peak_y < min_label_height:
                    continue
                peak_x = float(xplot[idx])
                plt.plot([peak_x, peak_x], [0, peak_y], color="#d35400", alpha=0.3, lw=0.8)
                label = group.split()[0] if group else group
                label_y = peak_y + max(peak_y * 0.15, max_y * 0.05)
                plt.text(
                    peak_x,
                    label_y,
                    label,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    color="#d35400",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.0),
                )
            plt.xlabel("Energy (keV)" if "energy" in result else "Channel")
            plt.ylabel("Counts")
            plt.title(Path(args.path).name)
            plt.grid(alpha=0.2, linestyle="--")
            plt.legend(loc="upper right", fontsize=8, frameon=False)
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            print(f"Saved plot to {args.plot}")

    if args.plot or args.plot_elements:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available, skip plotting element areas.")
        else:
            elements_path = args.plot_elements
            if not elements_path:
                plot_path = Path(args.plot)
                suffix = plot_path.suffix or ".png"
                elements_path = str(plot_path.with_name(f"{plot_path.stem}_elements{suffix}"))
            xplot = result.get("energy", x)
            yfit = result.get("yfit", y)
            continuum = result.get("continuum", np.zeros_like(y))
            plt.figure(figsize=(10, 4.5))
            plt.plot(xplot, ydata, lw=1.0, color="#2b2b2b", label="data")
            plt.plot(xplot, yfit, lw=1.0, color="#111111", alpha=0.8, label="fit")
            plt.plot(xplot, continuum, lw=1.0, color="#7f8c8d", ls="--", label="continuum")
            colors = [
                "#e74c3c",
                "#e67e22",
                "#f1c40f",
                "#2ecc71",
                "#1abc9c",
                "#3498db",
                "#9b59b6",
                "#e84393",
                "#16a085",
                "#d35400",
                "#2980b9",
                "#8e44ad",
            ]
            groups = result.get("groups", [])
            for i, group in enumerate(groups):
                if group.strip().upper() == "K K":
                    continue
                y_group = result.get(f"y{group}")
                if y_group is None:
                    continue
                label = group.split()[0] if group else group
                plt.fill_between(
                    xplot,
                    continuum,
                    continuum + y_group,
                    color=colors[i % len(colors)],
                    alpha=0.45,
                    linewidth=0.0,
                    label=label,
                )
            plt.xlabel("Energy (keV)" if "energy" in result else "Channel")
            plt.ylabel("Counts")
            plt.title(Path(args.path).name + " (elements)")
            plt.grid(alpha=0.2, linestyle="--")
            plt.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
            plt.tight_layout()
            plt.savefig(elements_path, dpi=150)
            print(f"Saved element area plot to {elements_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
