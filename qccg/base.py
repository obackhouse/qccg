"""
Base class for algebraic objects
"""


class AlgebraicBase:
    """
    Base class for algebraic objects, implements:

        a) Boolean logic based on the _sort_key method
        b) Hash function based on the _sort_key method
    """

    def _sort_key(self):
        raise NotImplementedError


    # Logic:

    def __eq__(self, other):
        return self._sort_key() == other._sort_key()

    def __ne__(self, other):
        return self._sort_key() != other._sort_key()

    def __lt__(self, other):
        return self._sort_key() < other._sort_key()

    def __le__(self, other):
        return self._sort_key() <= other._sort_key()

    def __gt__(self, other):
        return self._sort_key() > other._sort_key()

    def __ge__(self, other):
        return self._sort_key() >= other._sort_key()


    # Hashing:

    def __hash__(self):
        return hash(self._sort_key())


    # Algebra:

    def __add__(self, other):
        raise NotImplementedError

    #def __radd__(self, other):
    #    return other + self

    def __sub__(self, other):
        return self + (-1 * other)

    #def __rsub__(self, other):
    #    return other + (-1 * self)

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        # Since we only deal with indexed tensors (real or complex
        # scalars), they will always commute.
        return self * other

    __matmul__ = __mul__
    __rmatmul__ = __rmul__

    def __div__(self, other):
        raise NotImplementedError("Inversion not supported")

    def __rdiv__(self, other):
        raise NotImplementedError("Inversion not supported")

    def __pow__(self, other):
        if isinstance(other, int):
            out = self.copy()
            for i in range(other-1):
                out *= self
            return out
        else:
            raise NotImplementedError("Non-integer powers not supported")
