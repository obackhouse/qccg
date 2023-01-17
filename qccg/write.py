"""
Code generation
"""

from collections import defaultdict

default_characters = {
        "o": "ijklmnop",
        "v": "abcdefgh",
        "b": "wxyz",
}

def write_einsum(
        expression: "Expression",
        output: "Tensor",
        einsum_function: str = "np.einsum",
        characters: dict = default_characters,
        add_occupancies: set = {"f", "v"},
        reorder_axes: set = {"v": (0, 2, 1, 3)},
        indent: int = 0,
) -> str:
    """
    Writes an `Expression` in the form of an einsum.
    """

    ## Build a map of indices to characters
    #counters = defaultdict(int)
    #index_map = {}
    #for indices in [
    #        output.indices,
    #        *(contraction.indices for contraction in expression.contractions),
    #]:
    #    for index in indices:
    #        if index not in index_map:
    #            index_map[index.character] = \
    #                    characters[index.occupancy][counters[index.occupancy]]
    #            counters[index.occupancy] += 1

    # Write the terms
    subscript_out = "".join([index.character for index in output.indices])
    terms = []
    for contraction in expression.contractions:
        tensors = []
        subscripts_in = []
        for tensor in contraction.tensors:
            symbol = tensor.symbol
            indices = tensor.indices

            if symbol in reorder_axes:
                indices = tuple(indices[i] for i in reorder_axes[symbol])

            if any(index.spin in (0, 1) for index in indices):
                if not all(index.spin in (0, 1) for index in indices):
                    raise NotImplementedError
                symbol += "." + "".join(["ab"[index.spin] for index in indices])

            if tensor.symbol in add_occupancies:
                symbol += "." + "".join([index.occupancy for index in indices])

            tensors.append(symbol)
            subscripts_in.append("".join([index.character for index in indices]))

        tensors = ", ".join(tensors)
        subscripts_in = ",".join(subscripts_in)

        term = "{res} += {fn}(\"{inp}->{out}\", {tens}){op}{fac}".format(
                res=output.symbol,
                fn=einsum_function,
                inp=subscripts_in,
                out=subscript_out,
                tens=tensors,
                op="" if contraction.factor == 1.0 else " * ",
                fac="" if contraction.factor == 1.0 else contraction.factor,
        )
        terms.append(term)

    return (indent * " ") + ("\n%s" % (indent * " ")).join(terms)
