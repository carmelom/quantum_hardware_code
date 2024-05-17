#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 05/2024
# Author: Carmelo Mordini <carmelo.mordini@unipd.it>


import numpy as np
from scipy.stats import poisson


def rabi_flop_delta(t, Omega, Delta):
    return 0.5 * (
        Omega**2
        / (Omega**2 + Delta**2)
        * (1 + np.cos(np.sqrt(Omega**2 + Delta**2) * t))
    )


def rabi_flop(t, Omega):
    return 0.5 * (1 + np.cos(Omega * t))


def coherent_state_populations(n, alpha2):
    return poisson.pmf(n, alpha2)


def thermal_state_populations(n, nbar):
    return np.exp(n * np.log(nbar) - (n + 1) * np.log(nbar + 1))


def thermal_state_nmax(nbar, pn_threshold=1e-6):
    return int(
        np.ceil(
            -(np.log10(nbar + 1) + np.log10(pn_threshold))
            / (np.log10(nbar + 1) - np.log10(nbar))
        )
    )


def thermal_carrier_rabi_flop(t, Omega, eta, nbar, nmax):
    n = np.arange(nmax).reshape(-1, 1)
    t = t.reshape(1, -1)
    x = Omega * t * (1 - eta**2 * (n + 0.5))
    return 0.5 + 0.5 * np.sum(thermal_state_populations(n, nbar) * np.cos(x), axis=0)


def thermal_rsb_rabi_flop(t, Omega, eta, nbar, nmax):
    n = np.arange(nmax).reshape(-1, 1)
    t = t.reshape(1, -1)
    x = np.sqrt(n) * eta * Omega * t
    return 0.5 + 0.5 * np.sum(thermal_state_populations(n, nbar) * np.cos(x), axis=0)


def thermal_bsb_rabi_flop(t, Omega, eta, nbar, nmax):
    n = np.arange(nmax).reshape(-1, 1)
    t = t.reshape(1, -1)
    x = np.sqrt(n + 1) * eta * Omega * t
    return 0.5 + 0.5 * np.sum(thermal_state_populations(n, nbar) * np.cos(x), axis=0)


def coherent_bsb_rabi_flop(t, Omega, eta, alpha2, nmax):
    n = np.arange(nmax).reshape(-1, 1)
    t = t.reshape(1, -1)
    x = np.sqrt(n + 1) * eta * Omega * t
    return 0.5 + 0.5 * np.sum(coherent_state_populations(n, alpha2) * np.cos(x), axis=0)
