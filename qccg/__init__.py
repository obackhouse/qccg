"""
qccg: quantum chemistry code generation tools
"""


from qccg import base, index, tensor, contraction, read, write


# Register for dummy indices:

import bisect
from typing import Iterable
from qccg.index import DummyIndex

dummy_register = {}
spin = "ghf"

def register_dummies(dummies: Iterable[DummyIndex]):
    """
    Register a list of dummies globally.
    """

    import bisect

    global dummy_register

    for index in dummies:
        if index.spin is not None:
            index = index.copy(spin=None)
        if index.occupancy not in dummy_register:
            dummy_register[index.occupancy] = []
        if index not in dummy_register[index.occupancy]:
            # FIXME any point in making this O(1)?
            bisect.insort(dummy_register[index.occupancy], index)

def set_spin(spin_type: str):
    """
    Set the spin type.
    """

    global spin

    assert spin_type in ("rhf", "uhf", "ghf")

    spin = spin_type

def clear():
    """
    Clear the global variables.
    """

    global dummy_register, spin

    dummy_register = {}
    spin = "ghf"

del Iterable, DummyIndex
