"""
Indices
"""

from typing import Union, Iterable
import qccg
from qccg.base import AlgebraicBase


class AIndex(AlgebraicBase):
    """
    Abstract base class for an index.

    Attributes
    ----------
    character : str
        Character representation of the index.
    occupany : str
        Occupany of the index, one of {"o", "v", "O", "V", "b", "x"}.
    spin : int
        Spin of the index, 0 for alpha and 1 for beta. `None` is used
        to indicate a spin orbital, and 2 indicates a restricted orbital.
    """

    def __init__(self, character: str, occupancy: str, spin: Union[int, str]):
        self.character = character
        self.occupancy = self._convert_occupancy(occupancy)
        self.spin = self._convert_spin(spin)

    @staticmethod
    def _convert_occupancy(occupancy: str):
        if occupancy in ("occupied", "o", "hole", "h"):
            return "o"
        elif occupancy in ("virtual", "v", "particle", "p"):
            return "v"
        elif occupancy in ("occupied active", "O", "hole active", "H"):
            return "O"
        elif occupancy in ("virtual active", "V", "particle active", "P"):
            return "V"
        elif occupancy in ("boson", "b"):
            return "b"
        elif occupancy in ("auxiliary", "x"):
            return "x"
        else:
            raise ValueError("occupancy = %s" % spin)

    @staticmethod
    def _convert_spin(spin: Union[int, str]):
        if spin in ("alpha", "a", "α", 0):
            return 0
        elif spin in ("beta", "b", "β", 1):
            return 1
        elif spin in ("none", "n", None):
            return None
        elif spin in ("both", "restricted", "r", 2):
            return 2
        else:
            raise ValueError("spin = %s", spin)

    def copy(self, **kwargs):
        """
        Return a copy of the index, optionally replacing attributes.
        """

        character = kwargs.pop("character", self.character)
        occupancy = kwargs.pop("occupancy", self.occupancy)
        spin = kwargs.pop("spin", self.spin)

        index = self.__class__(character, occupancy, spin)

        return index

    def _sort_key(self):
        """
        Return a tuple representation of the object for memoisation,
        hashing and sorting.
        """

        return (
                "bovOVx".index(self.occupancy),
                self.character,
                self.summed,
                5 if self.spin is None else self.spin,
        )

    def __repr__(self):
        out = self.character
        if self.occupancy == self.occupancy.upper():
            out = out.upper()
        if self.spin in (0, 1):
            out += "ab"[self.spin]
        return out


class DummyIndex(AIndex):
    """
    Class for a dummy index, i.e. one that is summed over in an
    expression and therefore does not appear in the LHS.
    """

    summed = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        qccg.register_dummies((self,))


class ExternalIndex(AIndex):
    """
    Class for an external index, i.e. one that is not summed over in
    an expression and therefore appears in the LHS.
    summed = False
    """

    summed = False


def index_factory(
        cls: AIndex,
        characters: Iterable[str],
        occupancy: Iterable[str],
        spin: Iterable[Union[str, int]],
):
    """
    Factory to return indices.
    """

    if len(occupancy) == 1 and len(characters) > 1:
        occupancy = [occupancy[0] for i in range(len(characters))]

    if len(spin) == 1 and len(characters) > 1:
        spin = [spin[0] for i in range(len(characters))]

    assert len(characters) == len(occupancy) == len(spin)

    return tuple(cls(*tup) for tup in zip(characters, occupancy, spin))


