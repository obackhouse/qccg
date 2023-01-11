"""
Import expressions
"""

from collections import defaultdict

from qccg.index import ExternalIndex, DummyIndex
from qccg.tensor import Fock, ERI, FermionicAmplitude
from qccg.contraction import Contraction, Expression

pdaggerq_characters = {
        "o": "ijklmnot",
        "v": "abcdefgh",
}

def from_pdaggerq(terms: list, characters: dict = pdaggerq_characters) -> Expression:
    """
    Convert the output of `fully_contracted_strings` from `pdaggerq`
    into an `Expression`.
    """

    contractions = []
    for term in terms:
        # First element is the factor
        factor = float(term[0].lstrip("+"))

        # Find permuted indices
        permutation_operators = [el for el in term[1:] if el.startswith("P")]

        # Get tensors
        tensor_parts = [el for el in term[1:] if not el.startswith("P")]

        # Work out the external indices
        indices = defaultdict(int)
        for term in tensor_parts:
            if term.endswith(")"):
                term_indices = term[term.index("(")+1:term.index(")")].split(",")
            elif term.endswith(">"):
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
                raise ValueError

            if index in externals:
                index_map[index] = ExternalIndex(index, occupancy, None)
            else:
                index_map[index] = DummyIndex(index, occupancy, None)

        # Build the contraction
        tensors = []
        for part in tensor_parts:
            if part.startswith("f"):
                index_chars = part[part.index("(")+1:part.index(")")].split(",")
                indices = tuple(index_map[index] for index in index_chars)
                tensor = Fock(indices)
            elif part.startswith("t"):
                symbol = part[0]
                index_chars = part[part.index("(")+1:part.index(")")].split(",")
                indices = tuple(index_map[index] for index in index_chars)
                upper = indices[:len(indices)//2]
                lower = indices[len(indices)//2:]
                tensor = FermionicAmplitude(symbol, lower, upper)
            elif part.startswith("<"):
                index_chars = part[part.index("<")+1:part.index(">")].replace("||", ",").split(",")
                indices = tuple(index_map[index] for index in index_chars)
                tensor = ERI(indices)
            else:
                raise NotImplementedError(part)

            tensors.append(tensor)

        contraction = Contraction((factor, *tensors))
        contractions.append(contraction)

    # Build the expression
    expression = Expression(contractions)

    return expression
