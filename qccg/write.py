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
) -> str:
    """
    Writes an `Expression` in the form of an einsum.
    """

    # Build a map of indices to characters
    counters = defaultdict(int)
    index_map = {}
    for indices in [
            output.indices,
            *(contraction.indices for contraction in expression.contractions),
    ]:
        for index in indices:
            if index not in index_map:
                index_map[index] = characters[counters[index.occupancy, index.spin]]
                counters[index.occupancy, index.spin] += 1

    # Write the terms
    subscript_out = "".join([index_map[index] for index in output.indices])
    terms = []
    for contraction in expression.contractions:
        subscripts_in = ",".join([
                "".join([index_map[index] for index in tensor.indices])
                for tensor in contractions.tensors
        ])
        term = "{res} += {fn}({inp}->{out}){op}{fac}".format(
                res=output.symbol,
                fn=einsum_function,
                inp=subscript_in,
                out=subscript_out,
                op="" if contraction.factor == 1.0 else " * ",
                fac="" if contraction.factor == 1.0 else contraction.factor,
        )
        terms.append(term)

    return "\n".join(terms)
