#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 05/2024
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

from typing import Literal
import numpy as np
from qutip import (
    tensor,
    destroy,
    qeye,
    basis,
    sigmaz,
    sigmap,
    sigmam,
    thermal_dm,
    mesolve,
    Qobj,
)


def sigma_phi(phi):
    return np.exp(1j * phi) * sigmap() + np.exp(-1j * phi) * sigmam()


def sigma_op_j(sigma_op, j, N_ions):
    """Returns the operator sigma_op acting on the j-th ion in a chain of N_ions ions."""
    ops = [qeye(2) for _ in range(N_ions)]
    ops[j] = sigma_op
    return tensor(*ops)


def sdf_hamiltonian_j(
    j,
    N_ions,
    N_ho,
    phi_s,
    phi_m,
    arg_names=dict(Omega="Omega", eta="eta", delta="delta"),
):
    a = destroy(N_ho) * np.exp(-1j * phi_m)
    a_dag = a.dag()
    sigma = sigma_op_j(sigma_phi(phi_s), j, N_ions)
    H_sdf = [
        [
            tensor(sigma, a),
            "1j * {eta} * {Omega} / 2 * exp(1j * {delta} * t)".format(**arg_names),
        ],
        [
            tensor(sigma, a_dag),
            "-1j * {eta} * {Omega} / 2 * exp(-1j * {delta} * t)".format(**arg_names),
        ],
    ]
    return H_sdf


class SDFSolver:
    def __init__(
        self,
        N_ho,
        phi_s=0,
        phi_m=0,
        sideband_imbalance=False,
        center_line_detuning=False,
    ):
        # All frequencies in MHz, all times in us

        self._id = id(self)
        self.rhs_name = f"cqobjevo_ls_{self._id}"

        N_ions = 1
        self.N_ho = N_ho
        self.phi_s = phi_s
        self.phi_m = phi_m
        self.sideband_imbalance = sideband_imbalance
        self.center_line_detuning = center_line_detuning

        self.H = []
        self.H += sdf_hamiltonian_j(0, N_ions, N_ho, phi_s, phi_m)

        if sideband_imbalance:
            self.H += sdf_hamiltonian_j(
                0,
                N_ions,
                N_ho,
                phi_s + np.pi / 2,
                phi_m - np.pi / 2,
                arg_names=dict(Omega="alpha_Omega", eta="eta", delta="delta"),
            )

        if center_line_detuning:
            self.H += [[tensor(sigmaz(), qeye(N_ho)), "delta_c / 2"]]

        # projector operators
        id_s = qeye(2)

        g = basis(2, 0)
        gg = g * g.dag()
        e = basis(2, 1)
        ee = e * e.dag()

        _, (phi_minus, phi_plus) = sigma_phi(phi_s).eigenstates()
        phi_minus_proj = phi_minus * phi_minus.dag()
        phi_plus_proj = phi_plus * phi_plus.dag()

        id_m = qeye(N_ho)
        a = destroy(N_ho)
        a_dag = a.dag()
        x = a + a_dag
        p = 1j * (a_dag - a)

        self.e_ops = {
            "g": tensor(gg, id_m),
            "e": tensor(ee, id_m),
            "x": tensor(id_s, x),
            "p": tensor(id_s, p),
            "a": tensor(id_s, a),
            "n": tensor(id_s, a_dag * a),
            "a_minus": tensor(phi_minus_proj, a),
            "a_plus": tensor(phi_plus_proj, a),
        }

        # self.options = Options()
        # self.options.rhs_with_state = True
        # self.options.rhs_reuse = True
        # self.options.rhs_filename = self.rhs_name
        # qutip.rhs_generate(self.H, c_ops=[], args=self.default_args, name=self.rhs_name)

    def make_psi0(self, spin: Literal["x", "z"], n: int | float, thermal: bool = False):
        """
        Create the initial state `psi0` for a given spin direction and motional state.

        Args:
            spin (Literal["x", "z"]): Specifies the spin direction.
                - "x" for spin in the x-direction.
                - "z" for spin in the z-direction.
            n (int or float): initial fock state, or nbar if thermal=True.
            thermal (bool, optional): If True, create a thermal density matrix for the energy state.
                If False, create a pure state. Default is False.

        Returns:
            Qobj: A tensor product of the spin state and the energy state.
        """

        if thermal:
            psi_m = thermal_dm(self.N_ho, n)
        else:
            psi_m = basis(self.N_ho, n) * basis(self.N_ho, n).dag()
        if spin == "z":
            psi_s = basis(2, 0) * basis(2, 0).dag()
        elif spin == "x":
            px = (basis(2, 0) + basis(2, 1)).unit()
            psi_s = px * px.dag()
        else:
            raise ValueError(f"Invalid spin direction: {spin}. Must be 'x' or 'z'.")
        return tensor(psi_s, psi_m)

    def make_args(
        self,
        Omega: float,
        eta: float,
        delta: float,
        alpha_Omega: float = None,
        delta_c: float = None,
    ) -> dict:
        """
        Create a dictionary of arguments for a given set of parameters.

        Args:
            Omega (float): Rabi frequency.
            eta (float): Lamb-Dicke parameter.
            delta (float): Detuning.
            alpha_Omega (float, optional): Imbalance parameter for the Rabi frequency.
                Required if `sideband_imbalance` is True.
            delta_c (float, optional): Detuning for the center line.
                Required if `center_line_detuning` is True.

        Returns:
            dict: Dictionary containing the arguments.

        Raises:
            ValueError: If `sideband_imbalance` is True and `alpha_Omega` is not specified.
            ValueError: If `center_line_detuning` is True and `delta_c` is not specified.
        """
        args = {"Omega": Omega, "eta": eta, "delta": delta}
        if self.sideband_imbalance:
            if alpha_Omega is None:
                raise ValueError(
                    "alpha_Omega must be specified for sideband_imbalance=True"
                )
            args["alpha_Omega"] = alpha_Omega
        if self.center_line_detuning:
            if delta_c is None:
                raise ValueError(
                    "delta_c must be specified for center_line_detuning=True"
                )
            args["delta_c"] = delta_c
        return args

    def run(self, t: np.ndarray, psi0: Qobj, args: dict, **kwargs):
        result = mesolve(
            self.H,
            psi0,
            t,
            c_ops=None,
            e_ops=self.e_ops,
            args=args,
            # options=self.options,
            **kwargs,
        )
        return result


# 2-ion stuff
#
# p2 = tensor(gg, gg)
# p1 = tensor(gg, ee) + tensor(ee, gg)
# p0 = tensor(ee, ee)

# self.e_ops = {
#     "p2": tensor(p2, id_m),
#     "p1": tensor(p1, id_m),
#     "p0": tensor(p0, id_m),
#     "x": tensor(id_s, id_s, a + a_dag),
#     "p": tensor(id_s, id_s, 1j * (a_dag - a)),
#     "n": tensor(id_s, id_s, a_dag * a),
#     "x1g": tensor(gg, id_s, a + a_dag),
#     "p1g": tensor(gg, id_s, 1j * (a_dag - a)),
#     "x1e": tensor(ee, id_s, a + a_dag),
#     "p1e": tensor(ee, id_s, 1j * (a_dag - a)),
# }
