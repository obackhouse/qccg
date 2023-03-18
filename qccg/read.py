"""
Import expressions
"""

import re
import itertools
from collections import defaultdict

from qccg.index import ExternalIndex, DummyIndex
from qccg.tensor import Fock, ERI, FermionicAmplitude, RDM1, RDM2, Delta, ATensor
from qccg.contraction import Contraction, Expression

pdaggerq_characters = {
        "o": list("ijklmnot") + ["o%d" % x for x in range(20)],
        "v": list("abcdefgh") + ["v%d" % x for x in range(20)],
}

default_characters = {
        "o": list("ijklmnot") + ["o%d" % x for x in range(20)],
        "v": list("abcdefgh") + ["v%d" % x for x in range(20)],
        "O": list("IJKLMNOT") + ["O%d" % x for x in range(20)],
        "V": list("ABCDEFGH") + ["V%d" % x for x in range(20)],
        "b": list("wxyz") + ["b%d" % x for x in range(20)],
        "x": "PQRSUV",
}

def from_pdaggerq(
        terms: list,
        characters: dict = default_characters,
        index_spins: dict = {},
) -> Expression:
    """
    Convert the output of `fully_contracted_strings` from `pdaggerq`
    into an `Expression`.
    """

    # Check if this is a UHF expression
    has_spin = any(any("_" in part for part in term) for term in terms)  # OK?

    # Dissolve permutation operators
    i = 0
    while i < len(terms):
        for j in range(len(terms[i])):
            if terms[i][j].startswith("P"):
                swap = terms[i][j].strip("P(").strip(")").split(",")
                terms[i] = [x for k, x in enumerate(terms[i]) if j != k]

                new_term = terms[i].copy()
                for k, x in enumerate(new_term):
                    if "(" in x:
                        check = x.index("(")
                    elif "<" in x:
                        check = 0
                    else:
                        float(x)  # Has to be a float, this checks
                        continue

                    part = x[check:]
                    part = part.replace(swap[0], r"%TEMP")
                    part = part.replace(swap[1], swap[0])
                    part = part.replace(r"%TEMP", swap[1])

                    new_term[k] = x[:check] + part

                # Permutation has a phase of -1
                new_term[0] = {"+": "-", "-": "+"}[new_term[0][0]] + new_term[0][1:]

                terms.insert(i+1, new_term)
                break
        else:
            i += 1

    contractions = []
    for term in terms:
        # First element is the factor
        factor = float(term[0].lstrip("+"))

        # Get tensors
        tensor_parts = term[1:]
        assert not any(el.startswith("P") for el in tensor_parts)

        # Work out the external indices
        indices = defaultdict(int)
        for term in tensor_parts:
            if term.endswith(")"):
                term_indices = term[term.index("(")+1:term.index(")")].split(",")
            elif term.startswith("<"):
                term_indices = term[term.index("<")+1:term.index(">")].replace("||", ",").split(",")
            else:
                raise ValueError

            for index in term_indices:
                indices[index] += 1

        externals = set(key for key, val in indices.items() if val == 1)

        # Build index objects
        index_map = {}
        for index in indices.keys():
            if index in pdaggerq_characters["o"]:
                occupancy = "o"
            elif index in pdaggerq_characters["v"]:
                occupancy = "v"
            else:
                raise ValueError(index)

            spin = index_spins.get(index, None)

            if index in externals:
                index_map[index] = ExternalIndex(index, occupancy, spin)
            else:
                index_map[index] = DummyIndex(index, occupancy, spin)

        # Build the contraction
        tensors = []
        for part in tensor_parts:
            if part.startswith("f"):
                index_chars = part[part.index("(")+1:part.index(")")].split(",")
                indices = tuple(index_map[index] for index in index_chars)
                indices = indices[::-1]  # why?
                if has_spin:
                    spins = tuple("ab".index(s) for s in part[2:4])
                    indices = tuple(index.copy(spin=spin) for index, spin in zip(indices, spins))
                tensor = Fock(indices)

            elif any(part.startswith(x) for x in ("t", "l", "r")):
                index_chars = part[part.index("(")+1:part.index(")")].split(",")
                indices = tuple(index_map[index] for index in index_chars)
                symbol = part.split("(")[0]
                if has_spin:
                    idx = part.index("_") + 1
                    spins = tuple("ab".index(s) for s in part[idx:idx+order*2])
                else:
                    spins = tuple(index.spin for index in indices)
                lower = tuple(index.copy(spin=spin) for index, spin in zip(indices, spins) if index.occupancy.lower() == "o")
                upper = tuple(index.copy(spin=spin) for index, spin in zip(indices, spins) if index.occupancy.lower() == "v")
                if part.startswith("l"):
                    lower, upper = upper, lower
                tensor = FermionicAmplitude(symbol, lower, upper)

            elif part.startswith("<"):
                index_chars = part[part.index("<")+1:part.index(">")].replace("||", ",").split(",")
                indices = tuple(index_map[index] for index in index_chars)
                indices = indices[2:] + indices[:2]  # why?
                if has_spin:
                    spins = tuple("ab".index(s) for s in part[-4:])
                    indices = tuple(index.copy(spin=spin) for index, spin in zip(indices, spins))
                    indices = (indices[0], indices[2], indices[1], indices[3])
                tensor = ERI(indices)

            elif part.startswith("denom"):
                # FIXME
                class TempTensor(FermionicAmplitude):
                    @property
                    def perms(self):
                        for p1 in itertools.permutations([0,1,2]):
                            for p2 in itertools.permutations([3,4,5]):
                                perm = p1 + p2
                                if all(self.indices[i].spin == self.indices[p].spin for i, p in enumerate(perm)):
                                    yield (perm, 1)
                symbol = part.split("(")[0]
                index_chars = part[part.index("(")+1:part.index(")")].split(",")
                indices = tuple(index_map[index] for index in index_chars)
                n = len(indices) // 2
                tensor = TempTensor(symbol, indices[:n], indices[n:])

            elif part.startswith("d"):
                index_chars = part[part.index("(")+1:part.index(")")].split(",")
                indices = tuple(index_map[index] for index in index_chars)
                indices = indices[::-1]  # why?
                if has_spin:
                    spins = tuple("ab".index(s) for s in part[2:4])
                    indices = tuple(index.copy(spin=spin) for index, spin in zip(indices, spins))
                tensor = Delta(indices)

            else:
                raise NotImplementedError(part)

            tensors.append(tensor)

        # Append the contractions
        if not has_spin:
            # If we don't have spin, just process the tensors raw
            contraction = Contraction((factor, *tensors))
            contractions.append(contraction)
        else:
            # Otherwise, we want to remove the antisymmetry
            eri_indices = tuple(i for i, tensor in enumerate(tensors) if isinstance(tensor, ERI))
            antisymmetric_indices = tuple(
                    i for i, tensor in enumerate(tensors)
                    if isinstance(tensor, ERI)
                    and len(set(index.spin for index in tensor.indices)) == 1
            )
            for perm in itertools.product(range(2), repeat=len(antisymmetric_indices)):
                new_tensors = tensors.copy()
                for i, anti in zip(antisymmetric_indices, perm):
                    if anti:
                        new_tensors[i] = new_tensors[i].permute_indices((0, 3, 2, 1))
                new_factor = factor * (-1)**sum(perm)

                valid_spins = {
                        (0, 0, 0, 0),
                        (0, 1, 0, 1),
                        (1, 0, 1, 0),
                        (1, 1, 1, 1),
                }
                if all(
                        tuple(index.spin for index in new_tensors[i].indices) in valid_spins
                        for i in eri_indices
                ):
                    contraction = Contraction((new_factor, *new_tensors))
                    contractions.append(contraction)

    # Build the expression
    expression = Expression(contractions)

    return expression


def from_wicked(
        terms: list,
        characters: dict = default_characters,
        index_spins: dict = {},
) -> Expression:
    """
    Convert the output of `Equation.compile("ambit")` from `wicked`
    into an `Expression`.
    """

    contractions = []
    for term in terms:
        # Parse the term
        term = term.compile("ambit")
        term = term.rstrip(";")
        term = term.replace("*", "")
        term = term.split()

        # Get the elements
        output = term.pop(0)
        op = term.pop(0)
        if all(x in "1234567890-." for x in term[0]):
            factor = term.pop(0)
        tensors = term

        # Convert the factor
        if factor == "-":
            factor = "-1"
        factor = float(factor)
        if op == "-=":
            factor *= -1

        # Work out the external indices
        if "[" in output:
            externals = output.split("[")[1].rstrip("]").split(",")
            externals = tuple(characters[c[0]][int(c[1:])] for c in externals)
            externals = set(externals)
        else:
            externals = set()

        tensor_parts = []
        all_indices = set()
        for i, tensor in enumerate(tensors):
            # Get the symbol and index characters
            symbol = tensor.split("[")[0]
            indices = tensor.split("[")[1].rstrip("]").split(",")
            indices = tuple(characters[c[0]][int(c[1:])] for c in indices)
            tensor_parts.append((symbol, indices))
            all_indices = all_indices.union(set(indices))

        # Build index objects
        index_map = {}
        for index in all_indices:
            if index in characters["o"]:
                occupancy = "o"
            elif index in characters["v"]:
                occupancy = "v"
            elif index in characters["O"]:
                occupancy = "O"
            elif index in characters["V"]:
                occupancy = "V"
            else:
                raise ValueError

            if index in externals:
                index_map[index] = ExternalIndex(index, occupancy, None)
            else:
                index_map[index] = DummyIndex(index, occupancy, None)

        # Build the contraction
        tensors = []
        for symbol, indices in tensor_parts:
            indices = tuple(index_map[index] for index in indices)
            if symbol == "f":
                tensor = Fock(indices)
            elif symbol in ("t", "l"):
                raise NotImplementedError("FIXME")
                order = len(indices) // 2
                lower = indices[:order]
                upper = indices[order:]
                tensor = FermionicAmplitude(symbol, lower, upper)
            elif symbol == "v":
                tensor = ERI(indices)
            else:
                raise NotImplementedError(part)

            tensors.append(tensor)

        # Append the contractions
        contraction = Contraction((factor, *tensors))
        contractions.append(contraction)

    # Build the expression
    expression = Expression(contractions)

    return expression


def from_wick(
        expr: str,
        characters: dict = default_characters,
        index_spins: dict = {},
) -> Expression:
    """
    Convert the string form of a `wick.AExpression` to an `Expression`.

    Just turns it into a pdaggerq format.
    """

    expr = expr.replace("}", "} ")
    expr = re.sub(r"\\sum\_{.*?}", "", expr)
    expr = re.sub(r"([-+]?\d+(?:\.\d+)?)(\D+)", r"\1 \2", expr)
    expr = expr.replace("+ ", "+")
    expr = expr.replace("- ", "-")

    expr = re.sub(
            r"f\_{([a-zA-Z]+[0-9]*)([a-zA-Z]+[0-9]*)}",
            r"f(\1,\2)",
            expr,
    )
    expr = re.sub(
            r"v\_{([a-zA-Z]+[0-9]*)([a-zA-Z]+[0-9]*)([a-zA-Z]+[0-9]*)([a-zA-Z]+[0-9]*)}",
            r"<\1,\2||\3,\4>",
            expr,
    )

    for i in range(10, 0, -1):
        for symbol in ["t", "l", "c", "r"]:
            pattern = r"%s\_{%s}" % (symbol, "".join(["([a-zA-Z]+[0-9]*)"] * i))
            res = r"%s%d(%s)" % (symbol, (i+1)//2, r",".join([r"\%d" % (i+1) for i in range(i)]))
            expr = re.sub(pattern, res, expr)

    terms = [e.split() for e in expr.split("\n")]

    return from_pdaggerq(terms, characters=characters, index_spins=index_spins)


def from_latex(
        string: str,
        characters: dict = pdaggerq_characters,
        index_spins: dict = {},
) -> Expression:
    """
    Convert a LaTeX expression to an `Expression`.
    """

    # Convert " - " -> " + -1.0 "
    string = string.replace(" - ", " + -1.0 ")

    # Convert \frac to factorised expression
    i = 0
    while i < (len(string) - 7):
        if string[i:i+7] == r"\\frac{" or string[i:i+6] == r"\frac{":
            # Find the closing bracket
            size = 7 if string[i:i+7] == r"\\frac{" else 6
            nopen = 0
            j = i + size
            while j < len(string):
                if string[j] == "{":
                    nopen += 1
                elif string[j] == "}":
                    if nopen == 0:
                        break
                    nopen -= 1
                j += 1
            assert string[j] == "}"

            # Find the factor and the other closing bracket
            first_close = j
            second_close = j + 2 + string[j+2:].index("}")
            factor_string = str(1 / float(string[first_close+2:second_close]))

            string = string[:i] + factor_string + " " + string[i+size:first_close] + string[second_close+1:]

        i += 1

    # Split on delimeter " + "
    parts = string.split(" + ")

    # Build the contractions
    contractions = []
    for part in parts:
        terms = part.split()
        tensors = []
        factors = []
        for term in terms:
            if all(t in "1234567890-." for t in term):
                factors.append(-1 if term == "-" else float(term))
            else:
                tensors.append(term)

        # Concatenate the factors:
        factor = 1
        for f in factors:
            factor *= f

        # Parse the tensors
        parts = []
        for i, tensor in enumerate(tensors):
            tensor = tensor.replace("{", "").replace("}", "")
            if "^" in tensor:
                if tensor.index("^") < tensor.index("_"):
                    symbol, part = tensor.split("^")
                    upper, lower = part.split("_")
                else:
                    symbol, part = tensor.split("_")
                    lower, upper = part.split("^")
                upper, lower = tuple(upper), tuple(lower)
                parts.append((symbol, (lower, upper)))
            else:
                symbol, indices = tensor.index("_")
                indices = tuple(indices)
                parts.append((symbol, indices))

        # Work out the external indices
        occupancies = {}
        for key, val in characters.items():
            for v in val:
                occupancies[v] = key
        index_counts = defaultdict(int)
        for symbol, indices in parts:
            if len(indices) == 2 and isinstance(indices[0], tuple):
                indices = indices[0] + indices[1]
            for index in indices:
                index_counts[index] += 1
        index_map = {
            index: (
                ExternalIndex(index, occupancies[index], index_spins.get(index, None))
                if count == 1 else
                DummyIndex(index, occupancies[index], index_spins.get(index, None))
            )
            for index, count in index_counts.items()
        }

        # Convert the tensors
        tensors = []
        for symbol, indices in parts:
            if len(indices) == 2 and isinstance(indices[0], tuple):
                if symbol in ("f", "v"):
                    indices = (tuple(index_map[index] for index in indices[0]+indices[1]),)
                else:
                    indices = (
                            tuple(index_map[index] for index in indices[0]),
                            tuple(index_map[index] for index in indices[1]),
                    )
            else:
                indices = (tuple(index_map[index] for index in indices),)

            if symbol == "f":
                tensor = Fock(*indices)
            elif symbol in ("t", "l", "c"):
                rank = len(indices[0])
                tensor = FermionicAmplitude(symbol+str(rank), *indices)
            elif symbol == "v":
                tensor = ERI(*indices)
            else:
                raise NotImplementedError(term)

            tensors.append(tensor)

        # Append the contraction
        contraction = Contraction((factor, *tensors))
        contractions.append(contraction)

    # Build the expression
    expression = Expression(contractions)

    return expression


