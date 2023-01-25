"""
Code generation
"""

from collections import defaultdict

import qccg
from qccg.misc import All

default_characters = {
        "o": "ijklmnop",
        "v": "abcdefgh",
        "b": "wxyz",
}


# TODO delete intermediates
def write_einsum(
        expression: "Expression",
        output: "Tensor",
        einsum_function: str = "np.einsum",
        zeros_function: str = "np.zeros",
        characters: dict = default_characters,
        add_occupancies: set = {"f", "v"},
        add_spins: set = All,
        reorder_axes: set = {},
        indent: int = 0,
        index_sizes: dict = {"o": "nocc", "v": "nvir", "b": "nbos"},
        garbage_collection: bool = True,
) -> str:
    """
    Writes an `Expression` in the form of an einsum.
    """

    terms = []

    # If index_sizes is passed, write initialisation
    if index_sizes:
        # Update with dummies
        sizes = []
        for index in output.indices:
            # FIXME
            key = None
            for key, val in characters.items():
                if index.character in val:
                    break
            if index.spin in (None, 2):
                sizes.append(index_sizes[key])
            else:
                sizes.append(index_sizes[key] + "[%d]" % index.spin)
        if len(sizes) == 0:
            terms.append("%s = 0" % output.symbol)
        else:
            shape = ", ".join(sizes)
            if output.rank == 1:
                shape += ","
            terms.append("{res} = {zeros}(({shape}), dtype=np.float64)".format(
                res=output.symbol,
                zeros=zeros_function,
                shape=shape,
            ))

    # Write the terms
    subscript_out = "".join([index.character for index in output.indices])
    for contraction in expression.contractions:
        tensors = []
        subscripts_in = []
        for tensor in contraction.tensors:
            symbol = tensor.symbol
            indices = tensor.indices

            if symbol in reorder_axes:
                indices = tuple(indices[i] for i in reorder_axes[symbol])

            if tensor.symbol in add_spins:
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


def write_opt_einsums(
        expressions: list,
        outputs: list,
        final_outputs: "Tensor",
        **kwargs,
) -> str:
    """
    Write a list of expressions into ordered einsums. Calls are
    ordered according to their need, and arrays are deleted when
    no longer needed.
    """

    einsums = [
            write_einsum(expression, output, **kwargs)
            for expression, output in zip(expressions, outputs)
    ]
    lines = "\n".join(einsums).split("\n")
    out = []

    # First, insert all lines that construct the outputted tensors
    i = 0
    while i < len(lines):
        if any(lines[i].startswith(tensor.symbol) for tensor in final_outputs):
            out.append(lines.pop(i))
        else:
            i += 1

    # Now, insert the intermediate definitions where they are needed,
    # until they have all been used up
    defined = set()
    while len(lines):
        for i in range(len(out)):
            reset = False
            if kwargs.get("einsum_function", "einsum") in out[i]:
                constituents = out[i].split(", ")[1:]
                constituents[-1] = constituents[-1].split(")")[0]
                k = i
                if out[k-1].startswith(out[k].split()[0]) and \
                        kwargs.get("zeros_function", "zeros") in out[k-1] and k:
                    k -= 1
                #while out[k].startswith(out[i].split()[0]) and k:
                #    k -= 1
                for constituent in constituents:
                    if constituent not in defined:
                        # TODO O(1)
                        for j in range(len(lines)-1, -1, -1):
                            if lines[j].startswith(constituent):
                                out.insert(k, lines.pop(j))
                                reset = True
                        defined.add(constituent)
            if reset:
                break

    assert len(lines) == 0

    # Find the first and last appearence of each tensor
    final_line = {x: 0 for x in defined if x.startswith("x")}
    for i, line in enumerate(out):
        constituents = line.split(", ")[1:]
        constituents[-1] = constituents[-1].split(")")[0]
        for constituent in constituents:
            if constituent.startswith("x"):
                final_line[constituent] = i

    # Add delete statements after tensors are used the final time
    del_lists = defaultdict(list)
    for constituent, line in final_line.items():
        del_lists[line].append(constituent)
    for line, del_list in sorted(list(del_lists.items()))[::-1]:
        out.insert(line+1, "del %s" % ", ".join(del_list))

    out = "\n".join(out)

    return out
