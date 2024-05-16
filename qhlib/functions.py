#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 05/2024
# Author: Carmelo Mordini <carmelo.mordini@unipd.it>


import numpy as np


def rabi_flop(t, Omega, Delta):
    return (
        Omega**2
        / (Omega**2 + Delta**2)
        * (1 + np.cos(np.sqrt(Omega**2 + Delta**2) * t))
        / 2
    )
