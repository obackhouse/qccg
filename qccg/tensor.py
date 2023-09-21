"""
Tensors
"""

import itertools
from collections import defaultdict
from typing import Union, Iterable
from ebcc.util import permutations_with_signs
import qccg
from qccg.base import AlgebraicBase
from qccg.misc import flatten
from qccg.index import AIndex, DummyIndex, ExternalIndex
from qccg.contraction import Contraction, Expression

ASSUME_REAL = True


class ATensor(AlgebraicBase):
    """
    Abstract base class for a tensor.
    """

    _symbol = None
    _sectors = None
    _perms = None

    def __init__(self, symbol: str, indices: Iterable[AIndex]):
        self.symbol = symbol
        self.indices = tuple(indices)
        self.parent = None

        self.check_sanity()

    def check_sanity(self):
        """Hook for subclasses to check their own sanity.
        """

        pass

    @property
    def rank(self):
        """Rank of the tensor.
        """

        return len(self.indices)

    @property
    def is_spin_orbital(self):
        """Return True if the tensor contains any spin orbitals.
        """

        return any(index.spin is None for index in self.indices)

    @property
    def is_unrestricted(self):
        """Return True if the tensor contains any unrestricted indices.
        """

        return any(index.spin in (0, 1) for index in self.indices)

    @property
    def is_restricted(self):
        """Return True if the tensor contains any restricted indices.
        """

        return any(index.spin == 2 for index in self.indices)

    def copy(self, **kwargs):
        """
        Return a copy of the tensor, optionally replacing attributes.
        """

        indices = kwargs.pop("indices", self.indices)

        args = []
        if self._symbol is None:
            args.append(kwargs.pop("symbol", self.symbol))
        if self._sectors is None:
            args.append(indices)
        else:
            i = 0
            for sector in self._sectors:
                n = len(getattr(self, sector))
                args.append(indices[i:i+n])
                i += n

        tensor = self.__class__(*args)

        return tensor

    def permute_indices(self, perm: Iterable[int]):
        """
        Return a copy with a permutation assigned to the indices.
        """

        indices = tuple(self.indices[x] for x in perm)

        return self.copy(indices=indices)

    def substitute_indices(self, index_map: dict):
        """
        Return a copy with indices exchanged according to a mapping.
        """

        indices = tuple(index_map.get(index, index) for index in self.indices)

        return self.copy(indices=indices)

    @property
    def perms(self):
        """Generate the symmetry-allowed permutations of the tensor,
        yielding a permutation tuple along with a sign indicating
        the phase (1 for symmetric and -1 for antisymmetric) of the
        permutation.
        """

        if self._perms is not None:
            for perm in self._perms:
                yield perm
        else:
            yield (tuple(range(self.rank)), 1)

    @perms.setter
    def perms(self, lst):
        """Allow the permutations to be set manually.
        """

        self._perms = lst

    def canonicalise(self):
        """
        Canonicalise the indices according to the minimum equivalent
        representation according to the permutations. Returns a tuple
        of tensors along with the phases of the canonicalisation,
        allowing the expansion of linear combinations to be considered
        canonical.
        """

        best = sign = None
        for perm, phase in self.perms:
            permuted_tensor = self.permute_indices(perm)
            if best is None or permuted_tensor < best:
                best, sign = permuted_tensor, phase

        return (best,), (sign,)

    @property
    def dummies(self):
        return tuple(index for index in self.indices if isinstance(index, DummyIndex))

    @property
    def externals(self):
        return tuple(index for index in self.indices if isinstance(index, ExternalIndex))

    def expand_spin_orbitals(self):
        """
        Expand any spin orbitals (those with unspecified spin) into a
        list of possible alpha and beta components, and a list of phases.
        """

        raise NotImplementedError  # Subclass only

    def _sort_key(self):
        """
        Return a key for sorting objects of this type.
        """

        # Allow a penalty slot to support additional characteristics
        # in subclasses
        penalty = 0

        return flatten((
                self.rank,
                self.symbol,
                penalty,
                tuple(zip(*tuple(index._sort_key() for index in self.indices))),
        ))

    def __repr__(self):
        index_string = ",".join(["%r" % x for x in self.indices])
        return "%s_{%s}" % (self.symbol, index_string)

    def __mul__(self, other) -> Contraction:
        return Contraction((self, other))

    def __add__(self, other) -> Expression:
        return Contraction((1 * self, 1 * other))


class Scalar(ATensor):
    """
    Scalar value.
    """

    def __init__(self, symbol: str):
        super().__init__(symbol, [])

    def copy(self):
        return self.__class__(self.symbol)

    def canonicalise(self):
        return self.copy()

    def expand_spin_orbitals(self):
        raise ValueError

    def __repr__(self):
        return self.symbol


class Fock(ATensor):
    """
    Fock matrix.
    """

    _symbol = "f"

    def __init__(self, indices: Iterable[AIndex], real: bool = ASSUME_REAL):
        super().__init__(self._symbol, indices)
        self.real = real

    def check_sanity(self):
        ATensor.check_sanity(self)
        assert self.rank == 2
        assert all(index.occupancy.lower() in ("o", "v") for index in self.indices)

    @property
    def perms(self):
        if self._perms is not None:
            for perm in self._perms:
                yield perm
        else:
            # Fock matrix is symmetric under transposition for any spin
            yield ((0, 1), 1)
            if self.real:
                yield ((1, 0), 1)

    def expand_spin_orbitals(self):
        """
        Expand any spin orbitals (those with unspecified spin) into a
        list of possible alpha and beta components, and a list of phases.
        """

        if qccg.spin == "ghf":
            return (self,), (1,)

        fixed_spins = {
                i: index.spin for i, index in enumerate(self.indices)
                if index.spin is not None
        }
        tensors = defaultdict(int)

        for spins in [
                (0, 0),
                (1, 1),
        ]:
            indices = tuple(index.copy(spin=spin) for index, spin in zip(self.indices, spins))

            # If we have fixed spins, skip this one if it's not valid:
            if any(indices[i].spin != spin for i, spin in fixed_spins.items()):
                continue

            tensor = self.copy(indices=indices)
            tensors[tensor] += 1

        return tuple(tensors.keys()), tuple(tensors.values())


class CDERI(ATensor):
    """
    Cholesky decomposed electronic repulsion integral.
    """

    _symbol = "v"

    def __init__(self, indices: Iterable[AIndex], real: bool = ASSUME_REAL):
        super().__init__(self._symbol, indices)
        self.real = real

    def check_sanity(self):
        ATensor.check_sanity(self)
        assert self.rank == 3
        assert all(index.occupancy.lower() in ("o", "v") for index in self.indices[1:])
        assert all(index.spin in (0, 1, 2) for index in self.indices[1:])
        assert self.indices[0].occupancy == "x"

    @property
    def perms(self):
        if self._perms is not None:
            for perm in self._perms:
                yield perm
        else:
            # Cholesky decomposed ERIs are symmetric under permutation
            # of their non-auxiliary indices
            yield ((0, 1, 2), 1)
            yield ((0, 2, 1), 1)


class ERI(ATensor):
    """
    Electronic repulsion integral.
    """

    _symbol = "v"

    def __init__(self, indices: Iterable[AIndex], real: bool = ASSUME_REAL):
        super().__init__(self._symbol, indices)
        self.real = real

    def check_sanity(self):
        ATensor.check_sanity(self)
        assert self.rank == 4
        assert all(index.occupancy.lower() in ("o", "v") for index in self.indices)

    @property
    def perms(self):
        if self._perms is not None:
            for perm in self._perms:
                yield perm
        else:
            if self.is_spin_orbital:
                # Physicists' notation, antisymmetrised
                yield ((0, 1, 2, 3),  1)
                yield ((0, 1, 3, 2), -1)
                yield ((1, 0, 2, 3), -1)
                yield ((1, 0, 3, 2),  1)
                if self.real:
                    yield ((2, 3, 0, 1),  1)
                    yield ((3, 2, 0, 1), -1)
                    yield ((2, 3, 1, 0), -1)
                    yield ((3, 2, 1, 0),  1)
            else:
                ## Physicists' notation, bare
                #yield ((0, 1, 2, 3), 1)
                #yield ((1, 0, 3, 2), 1)
                #if self.real:
                #    yield ((0, 3, 2, 1), 1)
                #    yield ((1, 2, 3, 0), 1)
                #    yield ((2, 1, 0, 3), 1)
                #    yield ((2, 3, 0, 1), 1)
                #    yield ((3, 0, 1, 2), 1)
                #    yield ((3, 2, 1, 0), 1)
                # Chemists' notation
                yield ((0, 1, 2, 3), 1)
                yield ((2, 3, 0, 1), 1)
                if self.real:
                    yield ((0, 1, 3, 2), 1)
                    yield ((1, 0, 2, 3), 1)
                    yield ((1, 0, 3, 2), 1)
                    yield ((2, 3, 1, 0), 1)
                    yield ((3, 2, 0, 1), 1)
                    yield ((3, 2, 1, 0), 1)

    def expand_spin_orbitals(self):
        """
        Expand any spin orbitals (those with unspecified spin) into sums
        over their alpha and beta components. Also expands the antisymmetry.
        """

        if qccg.spin == "ghf":
            return (self,), (1,)

        fixed_spins = {
                i: index.spin for i, index in enumerate(self.indices)
                if index.spin is not None
        }
        tensors = defaultdict(int)

        for spins, direct, exchange in [
                ((0, 0, 0, 0), True, True),
                ((1, 1, 1, 1), True, True),
                ((0, 1, 0, 1), True, False),
                ((1, 0, 1, 0), True, False),
                ((0, 1, 1, 0), False, True),
                ((1, 0, 0, 1), False, True),
        ]:
            indices = tuple(index.copy(spin=spin) for index, spin in zip(self.indices, spins))

            # If we have fixed spins, skip this one if it's not valid:
            if any(indices[i].spin != spin for i, spin in fixed_spins.items()):
                continue

            tensor = self.copy(indices=indices)

            if direct:
                tensors[tensor.permute_indices((0, 2, 1, 3))] += 1

            if exchange:
                tensors[tensor.permute_indices((0, 3, 1, 2))] -= 1

        return tuple(tensors.keys()), tuple(tensors.values())

    def expand_cderi(self, aux_index: AIndex):
        """
        Expand as a density fitting approximation.
        """

        return (
                CDERI((aux_index, self.indices[0], self.indices[1])),
                CDERI((aux_index, self.indices[2], self.indices[3])),
        )

    def _sort_key(self):
        """
        Return a key for sorting objects of this type.
        """

        # Add a penalty for ba
        if self.is_spin_orbital:
            penalty = int(tuple(i.spin for i in self.indices) == (1, 0, 1, 0))
        else:
            penalty = int(tuple(i.spin for i in self.indices) == (1, 1, 0, 0))

        return flatten((
                self.rank,
                self.symbol,
                penalty,
                tuple(zip(*tuple(index._sort_key() for index in self.indices))),
        ))


class RDM1(Fock):
    """
    1RDM matrix.
    """

    _symbol = "rdm1_f"


class RDM2(ATensor):
    """
    2RDM matrix.
    """

    _symbol = "rdm2_f"

    def __init__(self, indices: Iterable[AIndex], real: bool = ASSUME_REAL):
        super().__init__(self._symbol, indices)
        self.real = real

    @property
    def perms(self):
        if self._perms is not None:
            for perm in self._perms:
                yield perm
        else:
            if not self.is_restricted:
                yield ((0, 1, 2, 3),  1)
                yield ((0, 1, 3, 2), -1)
                yield ((1, 0, 2, 3), -1)
                yield ((1, 0, 3, 2),  1)
            else:
                yield (tuple(range(self.rank)), 1)

    def check_sanity(self):
        assert self.rank == 4
        assert all(index.occupancy.lower() in ("o", "v") for index in self.indices)
        ATensor.check_sanity(self)

    def expand_spin_orbitals(self):
        """
        Expand any spin orbitals (those with unspecified spin) into sums
        over their alpha and beta components. Also expands the antisymmetry.
        """

        if qccg.spin == "ghf":
            return (self,), (1,)

        fixed_spins = {
                i: index.spin for i, index in enumerate(self.indices)
                if index.spin is not None
        }
        tensors = defaultdict(int)

        for lower in itertools.product(range(2), repeat=2):
            for upper in set(itertools.permutations(lower)):
                spins = tuple(lower) + tuple(upper)
                indices = tuple(index.copy(spin=spin) for index, spin in zip(self.indices, spins))

                # If we have fixed spins, skip this one if it's not valid:
                if any(indices[i].spin != spin for i, spin in fixed_spins.items()):
                    continue

                tensor = self.copy(indices=indices)
                tensors[tensor] += 1

        tensors, factors = list(tensors.keys()), list(tensors.values())

        if qccg.spin == "uhf":
            # Expand the antisymmetry where spin allows
            for i, tensor in enumerate(tensors):
                spins = tuple(index.spin for index in tensor.indices)

                if not all(s in (0, 1) for s in spins):
                    continue

                new_tensors = []
                new_factors = []
                for perm, sign in [((0, 1, 2, 3), 1), ((1, 0, 2, 3), -1)]:
                    indices = tuple(tensor.indices[p] for p in perm)
                    spins_perm = tuple(index.spin for index in indices)

                    if spins == spins_perm:
                        new_tensors.append(tensor.permute_indices(perm))
                        new_factors.append(factors[i] * sign)

                tensors[i] = new_tensors
                factors[i] = new_factors

        tensors = flatten(tensors)
        factors = flatten(factors)

        return tuple(tensors), tuple(factors)

    def expand_spin_orbitals(self):
        """
        Expand any spin orbitals (those with unspecified spin) into sums
        over their alpha and beta components. Also expands the antisymmetry.
        """

        if qccg.spin == "ghf":
            return (self,), (1,)

        fixed_spins = {
                i: index.spin for i, index in enumerate(self.indices)
                if index.spin is not None
        }
        tensors = defaultdict(int)

        for spins, direct, exchange in [
                ((0, 0, 0, 0), True, True),
                ((1, 1, 1, 1), True, True),
                ((0, 1, 0, 1), True, False),
                ((1, 0, 1, 0), True, False),
                ((0, 1, 1, 0), False, True),
                ((1, 0, 0, 1), False, True),
        ]:
            indices = tuple(index.copy(spin=spin) for index, spin in zip(self.indices, spins))

            # If we have fixed spins, skip this one if it's not valid:
            if any(indices[i].spin != spin for i, spin in fixed_spins.items()):
                continue

            tensor = self.copy(indices=indices)

            if direct:
                tensors[tensor.permute_indices((0, 1, 2, 3))] += 1

            if exchange:
                tensors[tensor.permute_indices((0, 1, 3, 2))] -= 1

        return tuple(tensors.keys()), tuple(tensors.values())

    def canonicalise(self):
        """
        Canonicalise the indices according to the minimum equivalent
        representation according to the permutations. Returns a new
        tensor along with the phase of the canonicalisation.

        Also handles spin permutations for amplitudes.
        """

        tensors = [self]
        factors = [1]

        #if qccg.spin == "rhf":
        #    # Spin flip if needed
        #    for i, tensor in enumerate(tensors):
        #        spins = tuple(index.spin for index in tensor.indices)

        #        if any(spin is None for spin in spins):
        #            # Should this not be happening if we haven't expanded spin
        #            # orbitals? i.e. if spinned indices are explicitly used
        #            continue

        #        if sum(s == 1 for s in spins) > sum(s == 0 for s in spins):
        #            indices = tuple(
        #                    index.copy(spin=(index.spin + 1) % 2)
        #                    for index in tensor.indices
        #            )
        #            tensors[i] = tensor.copy(indices=indices)

        #    # Expand same-spin contributions in linear combinations of
        #    # mixed spin components
        #    for i, tensor in enumerate(tensors):
        #        if tensor.rank > 2:
        #            if all(index.spin == 0 for index in tensor.indices):
        #                spins = []
        #                for k in range(tensor.rank // 2):
        #                    spin = [(1 if j % 2 else 0) for j in range(tensor.rank // 2)]
        #                    spin += [(1 if k == j else 0) for j in range(tensor.rank // 2)]
        #                    spins.append(spin)

        #                new_tensors = []
        #                for spin in spins:
        #                    indices = tuple(
        #                            index.copy(spin=s)
        #                            for index, s in zip(tensor.indices, spin)
        #                    )
        #                    new_tensors.append(tensor.copy(indices=indices))

        #                tensors[i] = new_tensors
        #                factors[i] = [1] * len(new_tensors)

        tensors = flatten(tensors)
        factors = flatten(factors)

        # Now perform the usual canonicalisation:
        for i, tensor in enumerate(tensors):
            tensors[i], factors[i] = ATensor.canonicalise(tensor)

        tensors = flatten(tensors)
        factors = flatten(factors)

        return tuple(tensors), tuple(factors)

    def _sort_key(self):
        """
        Return a key for sorting objects of this type. Adds a penalty
        for neighbouring indices with the same spin.
        """

        # Add a penalty for when the ordering of the spin cases is
        # not the same in covariant and contravariant indices
        penalty = 2 * (
                + int(self.indices[0].spin != self.indices[2].spin)
                + int(self.indices[1].spin != self.indices[3].spin)
        )

        # We want mixed-spin cases to have alterating spin
        # i.e. abab, abaaba, etc.
        pattern = tuple(([0, 1] * self.rank)[:self.rank // 2])
        penalty += (
                + int(tuple(index.spin for index in self.indices[:2]) != pattern)
                * int(tuple(index.spin for index in self.indices[2:]) != pattern)
        )

        return flatten((
                self.rank,
                self.symbol,
                penalty,
                tuple(zip(*tuple(index._sort_key() for index in self.indices))),
        ))


class Delta(Fock):
    """
    Delta function.
    """

    _symbol = "delta"


class FermionicAmplitude(ATensor):
    """
    Fermionic amplitude tensor.

    Requires specification of upper and lower indices.
    """

    _sectors = ("lower", "upper")

    def __init__(self, symbol: str, lower: Iterable[AIndex], upper: Iterable[AIndex]):
        indices = tuple(lower) + tuple(upper)
        super().__init__(symbol, indices)
        self.lower = lower
        self.upper = upper

    @property
    def perms(self):
        if self._perms is not None:
            for perm in self._perms:
                yield perm
        else:
            nlower = len(self.lower)
            spins = tuple(i.spin for i in self.indices)
            if not self.is_restricted:
                for lower_perm, lower_sign in permutations_with_signs(range(nlower)):
                    for upper_perm, upper_sign in permutations_with_signs(range(nlower, self.rank)):
                        perm = tuple(lower_perm) + tuple(upper_perm)
                        sign = lower_sign * upper_sign
                        yield (perm, sign)
            else:
                # Only swap particles that originate from same spin...?
                # Special cases:
                if self.symbol.startswith("t2") or self.symbol.startswith("l2"):
                    yield ((0, 1, 2, 3), 1)
                    yield ((1, 0, 3, 2), 1)
                elif self.symbol.startswith("t3") or self.symbol.startswith("l3"):
                    yield ((0, 1, 2, 3, 4, 5), 1)
                    yield ((2, 1, 0, 5, 4, 3), 1)
                else:
                    yield (tuple(range(self.rank)), 1)

    def check_sanity(self):
        ATensor.check_sanity(self)

    def expand_spin_orbitals(self):
        """
        Expand any spin orbitals (those with unspecified spin) into sums
        over their alpha and beta components. Also expands the antisymmetry.
        """

        if qccg.spin == "ghf":
            return (self,), (1,)

        fixed_spins = {
                i: index.spin for i, index in enumerate(self.indices)
                if index.spin is not None
        }
        tensors = defaultdict(int)

        # FIXME how to generalise?
        spins_list = []
        if len(self.lower) == len(self.upper):
            for lower in itertools.product(range(2), repeat=len(self.lower)):
                for upper in set(itertools.permutations(lower)):
                    spins_list.append(tuple(lower) + tuple(upper))
        elif {len(self.lower), len(self.upper)} == {0, 1}:
            spins_list.append((0,))
            spins_list.append((1,))
        elif {len(self.lower), len(self.upper)} == {1, 2}:
            spins_list.append((0, 0, 0))
            spins_list.append((0, 1, 0))
            spins_list.append((1, 0, 1))
            spins_list.append((1, 1, 1))
        else:
            raise NotImplementedError

        for spins in spins_list:
            indices = tuple(index.copy(spin=spin) for index, spin in zip(self.indices, spins))

            # If we have fixed spins, skip this one if it's not valid:
            if any(indices[i].spin != spin for i, spin in fixed_spins.items()):
                continue

            tensor = self.copy(indices=indices)
            tensors[tensor] += 1

        tensors, factors = list(tensors.keys()), list(tensors.values())

        if qccg.spin == "uhf":
            # Expand the antisymmetry where spin allows
            for i, tensor in enumerate(tensors):
                spins = tuple(index.spin for index in tensor.indices)

                if not all(s in (0, 1) for s in spins):
                    continue

                new_tensors = []
                new_factors = []
                if len(self.upper) >= len(self.lower):
                    for perm, sign in permutations_with_signs(range(len(self.upper))):
                        indices = tensor.lower + tuple(tensor.upper[p] for p in perm)
                        spins_perm = tuple(index.spin for index in indices)

                        if spins == spins_perm:
                            full_perm = tuple(range(len(self.lower))) + \
                                    tuple(p+len(self.lower) for p in perm)
                            new_tensors.append(tensor.permute_indices(full_perm))
                            new_factors.append(factors[i] * sign)
                else:
                    for perm, sign in permutations_with_signs(range(len(self.lower))):
                        indices = tuple(tensor.lower[p] for p in perm) + tensor.upper
                        spins_perm = tuple(index.spin for index in indices)

                        if spins == spins_perm:
                            full_perm = tuple(perm) + tuple(p+len(self.lower) for p in range(len(self.upper)))
                            new_tensors.append(tensor.permute_indices(full_perm))
                            new_factors.append(factors[i] * sign)

                tensors[i] = new_tensors
                factors[i] = new_factors

        tensors = flatten(tensors)
        factors = flatten(factors)

        return tuple(tensors), tuple(factors)

    def canonicalise(self):
        """
        Canonicalise the indices according to the minimum equivalent
        representation according to the permutations. Returns a new
        tensor along with the phase of the canonicalisation.

        Also handles spin permutations for amplitudes.
        """

        tensors = [self]
        factors = [1]

        if qccg.spin == "rhf":
            # Spin flip if needed
            for i, tensor in enumerate(tensors):
                spins = tuple(index.spin for index in tensor.indices)

                if any(spin is None for spin in spins):
                    # Should this not be happening if we haven't expanded spin
                    # orbitals? i.e. if spinned indices are explicitly used
                    continue

                if sum(s == 1 for s in spins) > sum(s == 0 for s in spins):
                    indices = tuple(
                            index.copy(spin=(index.spin + 1) % 2)
                            for index in tensor.indices
                    )
                    tensors[i] = tensor.copy(indices=indices)

            # Expand same-spin contributions in linear combinations of
            # mixed spin components
            for i, tensor in enumerate(tensors):
                if tensor.rank > 2:
                    if all(index.spin == 0 for index in tensor.indices):
                        spins = []
                        for k in range(tensor.rank // 2):
                            # FIXME for R?
                            if len(tensor.lower) <= len(tensor.upper):
                                spin = [(1 if j % 2 else 0) for j in range(len(tensor.lower))]
                                spin += [(1 if k == j else 0) for j in range(len(tensor.upper))]
                            else:
                                spin = [(1 if k == j else 0) for j in range(len(tensor.lower))]
                                spin += [(1 if j % 2 else 0) for j in range(len(tensor.upper))]
                            spins.append(spin)

                        new_tensors = []
                        for spin in spins:
                            indices = tuple(
                                    index.copy(spin=s)
                                    for index, s in zip(tensor.indices, spin)
                            )
                            new_tensors.append(tensor.copy(indices=indices))

                        tensors[i] = new_tensors
                        factors[i] = [1] * len(new_tensors)

        tensors = flatten(tensors)
        factors = flatten(factors)

        # Now perform the usual canonicalisation:
        for i, tensor in enumerate(tensors):
            tensors[i], factors[i] = ATensor.canonicalise(tensor)

        tensors = flatten(tensors)
        factors = flatten(factors)

        return tuple(tensors), tuple(factors)

    def _sort_key(self):
        """
        Return a key for sorting objects of this type. Adds a penalty
        for neighbouring indices with the same spin.
        """

        penalty = []

        if len(self.lower) == len(self.upper):
            # For T, L we want mixed-spin cases to have alternating
            # spin i.e. abab, abaaba, etc.
            pattern_a = tuple(([0, 1] * self.rank)[:self.rank // 2]) * 2
            pattern_b = tuple(([1, 0] * self.rank)[:self.rank // 2]) * 2
            penalty += [
                    int(tuple(index.spin for index in self.indices) != pattern_a),
                    int(tuple(index.spin for index in self.indices) != pattern_b),
                    *[int(i.spin != j.spin) for i, j in zip(self.lower, self.upper)],
            ]
        else:
            # For R we want mixed-spin cases to have alternating
            # spin i.e. aba, bab, ababa, babab, etc.
            pattern_a = tuple(i % 2 for i in range(self.rank))
            pattern_b = tuple(0 if i == 1 else 1 for i in pattern_a)
            penalty += [
                    int(tuple(index.spin for index in self.indices) != pattern_a),
                    int(tuple(index.spin for index in self.indices) != pattern_b),
            ]

        # Active indices at the end of indices list
        active_lower = tuple(index.occupancy == index.occupancy.upper() for index in self.lower)
        active_upper = tuple(index.occupancy == index.occupancy.upper() for index in self.upper)
        penalty += [
                int(active_lower != tuple(sorted(active_lower))),
                int(active_upper != tuple(sorted(active_upper))),
        ]

        # Otherwise, prefer all a indices before b
        if len(self.lower) > 1:
            penalty.append(int(any((i.spin, j.spin) == (1, 0) for i, j in zip(self.lower[:-1], self.lower[1:]))))
        if len(self.upper) > 1:
            penalty.append(int(any((i.spin, j.spin) == (1, 0) for i, j in zip(self.upper[:-1], self.upper[1:]))))

        penalty = int("".join([str(x) for x in penalty]), 2)

        return flatten((
                self.rank,
                self.symbol,
                penalty,
                tuple(zip(*tuple(index._sort_key() for index in self.indices))),
        ))


class BosonicAmplitude(ATensor):
    """
    Bosonic amplitude tensor.
    """

    def check_sanity(self):
        ATensor.check_sanity(self)

    @property
    def perms(self):
        """
        Bosonic amplitudes have permutational symmetry according to
        symmetric exchange of the indices.
        """

        if self._perms is not None:
            for perm in self._perms:
                yield perm
        else:
            for perm, _ in permutations_with_signs(range(self.rank)):
                yield (tuple(perm), 1)


class MixedAmplitude(ATensor):
    """
    Mixed fermionic and bosonic coupling amplitude tensor.

    Requires specification of bosonic, upper and lower indices.
    """

    _sectors = ("bosonic", "lower", "upper")

    def __init__(
            self,
            symbol: str,
            bosonic: Iterable[AIndex],
            lower: Iterable[AIndex],
            upper: Iterable[AIndex],
    ):
        indices = tuple(bosonic) + tuple(lower) + tuple(upper)
        super().__init__(symbol, indices)
        self.bosonic = bosonic
        self.lower = lower
        self.upper = upper

    def check_sanity(self):
        ATensor.check_sanity(self)

    @property
    def perms(self):
        """
        Mixed amplitudes have permutational symmetry according to
        antisymmetric exchange of the indices within upper and lower
        sectors and symmetric exchange of the indices in the bosonic
        sector.
        """

        if self._perms is not None:
            for perm in self._perms:
                yield perm
        else:
            nboson = len(self.bosonic)
            nlower = len(self.lower)
            for bosonic_perm, _ in permutations_with_signs(range(nboson)):
                for lower_perm, lower_sign in permutations_with_signs(range(nlower)):
                    for upper_perm, upper_sign in permutations_with_signs(range(nlower, self.rank)):
                        sign = lower_sign * upper_sign
                        yield (tuple(bosonic_perm) + tuple(lower_perm) + tuple(upper_perm), sign)


del ASSUME_REAL
