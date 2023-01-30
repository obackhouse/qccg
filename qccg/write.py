"""
Code generation
"""

from collections import defaultdict

import qccg
from qccg.misc import All, flatten

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
        final_outputs: list,
        **kwargs,
) -> str:
    """
    Write a list of expressions into ordered einsums. Calls are
    ordered according to their need, and arrays are deleted when
    no longer needed.
    """

    # FIXME this requires that the intermediate labels are already
    # in a sensible order according to their need

    # Split the expressions into single contractions
    outputs = flatten([
            [output] * len(expression.contractions)
            for expression, output in zip(expressions, outputs)
    ])
    expressions = flatten([
            [
                expression.__class__((contraction,), simplify=False)
                for contraction in expression.contractions
            ]
            for expression in expressions
    ])

    assert len(expressions) == len(outputs)

    new_expressions = []
    new_outputs = []

    # Push any expression resulting in one of the final outputs to the
    # end of the list
    for i in range(len(expressions)-1, -1, -1):
        if outputs[i] in final_outputs:
             new_expressions.insert(0, expressions.pop(i))
             new_outputs.insert(0, outputs.pop(i))

    # Sort the output expressions according to the RHS intermediates
    def score_expression(expression):
        scores = [
                [
                    int(tensor.symbol.lstrip("x")) if tensor.symbol.startswith("x") else -1
                    for tensor in contraction.tensors
                ]
                for contraction in expression.contractions
        ]
        return max(flatten(scores))
    key = lambda tup: score_expression(tup[0])
    new_expressions, new_outputs = zip(*sorted(zip(new_expressions, new_outputs), key=key))
    new_expressions = list(new_expressions)
    new_outputs = list(new_outputs)

    if len(expressions):
        # Sort the remaining expressions according to the LHS intermediate
        def score_output(output):
            score = int(output.symbol.lstrip("x")) if output.symbol.startswith("x") else -1
            return score
        key = lambda tup: score_output(tup[1])
        expressions, outputs = zip(*sorted(zip(expressions, outputs), key=key))
        expressions = list(expressions)
        outputs = list(outputs)

        # Insert the remaining expressions
        i = 0
        while len(expressions):
            current_score = score_expression(new_expressions[i])
            if current_score != -1:
                while len(expressions) and score_output(outputs[0]) <= current_score:
                    new_expressions.insert(i, expressions.pop(0))
                    new_outputs.insert(i, outputs.pop(0))
                    i += 1
            i += 1

    # Build einsums
    einsums = flatten([
            write_einsum(expression, output, **kwargs).split("\n")
            for expression, output in zip(new_expressions, new_outputs)
    ])

    # Since we split the contractions we need to remove duplicate
    # initialisation statements
    seen = set()
    delete = set()
    for i, line in enumerate(einsums):
        if kwargs.get("zeros_function", "zeros") in line or " = 0" in line:
            if line in seen:
                delete.add(i)
            seen.add(line)
    einsums = [
            line
            for i, line in enumerate(einsums)
            if i not in delete
    ]

    # Find the last appearence of each tensor
    final_line = defaultdict(int)
    for i, line in enumerate(einsums):
        if not (kwargs.get("zeros_function", "zeros") in line or " = 0" in line):
            tensors = line.split(", ")[1:]
            tensors[-1] = tensors[-1].split(")")[0]
            for tensor in tensors:
                if tensor.startswith("x"):
                    final_line[tensor] = i

    # Add delete statements after tensors are used the final time
    del_lists = defaultdict(list)
    for tensor, line in final_line.items():
        del_lists[line].append(tensor)
    for line, del_list in sorted(list(del_lists.items()))[::-1]:
        einsums.insert(line+1, "%sdel %s" % (kwargs.get("indent", 0) * " ", ", ".join(del_list)))

    return "\n".join(einsums)
