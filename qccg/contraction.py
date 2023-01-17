"""
Contractions
"""

import itertools
from typing import Union, Iterable, Any
from numbers import Number
from collections import defaultdict
import qccg
from qccg.base import AlgebraicBase
from qccg.misc import flatten


class Expression(AlgebraicBase):
    """
    Expression of contractions.
    """

    def __init__(self, contractions: Iterable["Contraction"], simplify: bool = True):
        contractions = tuple(
                Contraction((contraction,))
                if not hasattr(contraction, "tensors") else contraction
                for contraction in contractions
        )
        self.contractions = contractions
        if simplify:
            self._simplify()

    def _simplify(self):
        """
        Simplify the sum in-place. This does the following:

            a) Canonicalise the contractions
            b) Collect like contractions and factorise accordingly
            c) Sort the contractions
        """

        # Canonicalise the contractions
        contractions = []
        for contraction in self.contractions:
            new_contractions = contraction.canonicalise()
            contractions += list(new_contractions)

        contractions = flatten(contractions)

        # Collect contractions
        contractions_map = defaultdict(list)
        for contraction in contractions:
            contractions_map[contraction.tensors].append(contraction.factor)
        contractions = [
                Contraction((sum(factors), *tensors))
                for tensors, factors in contractions_map.items()
        ]

        # Remove contractions that are zero
        contractions = [contraction for contraction in contractions if contraction.factor != 0]

        # Sort the contractions
        contractions = tuple(sorted(contractions))

        # Collect result
        self.contractions = contractions

    def expand_spin_orbitals(self):
        """
        Expand any spin orbitals (those with unspecified spin) into sums
        over their alpha and beta components.
        """

        contractions = []
        for contraction in self.contractions:
            contractions += list(contraction.expand_spin_orbitals().contractions)

        contractions = flatten(contractions)

        if qccg.spin == "rhf":
            # Relabel spinned indices with restricted ones
            contractions = [
                    Contraction((contraction.factor, *[
                        tensor.copy(indices=tuple(
                            index.copy(spin=2)
                            if index.spin in (0, 1) else index
                            for index in tensor.indices
                        ))
                        for tensor in contraction.tensors
                    ]))
                    for contraction in contractions
            ]

        expression = self.__class__(contractions)

        return expression

    def __repr__(self):
        return "\n + ".join(["%r" % contraction for contraction in self.contractions])

    def __mul__(self, other):
        return self.__class__(tuple(contraction * other for contraction in self.contractions))

    def __add__(self, other):
        if isinstance(other, self.__class__):
            contractions = (*self.contractions, *other.contractions)
        else:
            contractions = (*self.contractions, other)
        return self.__class__(contractions)


class Contraction(AlgebraicBase):
    """
    Contraction between integers, floats and tensors.
    """

    def __init__(self, terms: Iterable[Any]):
        factors = [term for term in terms if isinstance(term, Number)]
        tensors = [term for term in terms if not isinstance(term, Number)]
        self.factor = 1.0
        for factor in factors:
            self.factor *= factor
        self.tensors = tuple(tensors)

    @property
    def terms(self):
        return (self.factor,) + self.tensors

    @property
    def dummies(self):
        dummies = {}
        for tensor in self.tensors:
            for dummy in tensor.dummies:
                dummies[dummy] = None
        return tuple(dummies.keys())

    @property
    def externals(self):
        externals = {}
        for tensor in self.tensors:
            for external in tensor.externals:
                externals[external] = None
        return tuple(externals.keys())

    @property
    def indices(self):
        indices = {}
        for tensor in self.tensors:
            for index in tensor.indices:
                indices[index] = None
        return tuple(indices.keys())

    def canonicalise_dummies(self):
        """
        Reset the dummy indices in the contraction into a canonical
        form.
        """

        dummies = self.dummies
        occupancies = set(index.occupancy for index in dummies)

        # Build a map of the dummy indices to swap
        dummy_map = {}
        for occupancy in occupancies:
            cache = set()
            old = []
            for index in dummies:
                if index.occupancy == occupancy:
                    index = index.copy(spin=None)
                    if index not in cache:
                        old.append(index)
                        cache.add(index)
            new = qccg.dummy_register[occupancy][:len(old)].copy()
            for spin in (None, 0, 1):
                for o, n in zip(old, new):
                    dummy_map[o.copy(spin=spin)] = n.copy(spin=spin)

        # Substitute the indices in the tensors
        tensors = tuple(tensor.substitute_indices(dummy_map) for tensor in self.tensors)

        return self.__class__((self.factor,) + tensors)

    def canonicalise(self):
        """
        Canonicalise the contraction. Dummy indices can be freely
        swapped. Also canonicalises the individual tensors. Returns a
        new `Contraction`.

            a) Canonicalise the tensors
            b) Canonicalise the contraction
            c) Collect factors
            d) Sort the terms
        """

        # Canonicalise the tensors
        tensors, factors = zip(*[tensor.canonicalise() for tensor in self.tensors])

        # Collect results
        contractions = []
        for tensors, factors in zip(
                itertools.product(*tensors),
                itertools.product(*factors),
        ):
            factor = self.factor
            for f in factors:
                factor *= f
            tensors = tuple(sorted(tensors))
            contractions.append(self.__class__((factor, *tensors)))

        # Canonicalise the dummy indices across the full contraction
        contractions = tuple(contraction.canonicalise_dummies() for contraction in contractions)

        return contractions

    def expand_spin_orbitals(self):
        """
        Expand any spin orbitals (those with unspecified spin) into sums
        over their alpha and beta components.
        """

        # FIXME this can be made more efficient by applying the same index
        # substitutions to the entire contraction!

        # Expand the tensors
        tensors, factors = zip(*[tensor.expand_spin_orbitals() for tensor in self.tensors])

        # Collect results
        contractions = []
        for tensors, factors in zip(
                itertools.product(*tensors),
                itertools.product(*factors),
        ):
            spins = {}
            good = True
            for tensor in tensors:
                for index in tensor.indices:
                    spinless = index.copy(spin=None)
                    if spinless in spins:
                        if spins[spinless] != index.spin:
                            good = False
                    else:
                        spins[spinless] = index.spin
                    if not good:
                        break
                if not good:
                    break

            if good:
                factor = self.factor
                for f in factors:
                    factor *= f
                tensors = tuple(sorted(tensors))
                contractions.append(self.__class__((factor, *tensors)))

        return Expression(contractions)

    def _sort_key(self):
        """
        Return a key for sorting objects of this type.
        """

        return flatten((
                len(self.tensors),
                tuple(zip(*tuple(tensor._sort_key() for tensor in self.tensors))),
                abs(self.factor),
                self.factor > 0,
        ))

    def __repr__(self):
        return " ".join(["%r" % term for term in self.terms])

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            terms = (*self.terms, *other.terms)
        else:
            terms = (*self.terms, other)
        return self.__class__(terms)

    def __add__(self, other) -> Expression:
        if isinstance(other, Expression):
            contractions = (self, *other.contractions)
        else:
            contractions = (self, other)
        return Expression(contractions)
