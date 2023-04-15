"""
Code generation
"""

from collections import defaultdict
import itertools
import re

import qccg
from qccg.misc import All, flatten
from qccg.read import default_characters
from ebcc.util import permutations_with_signs

# TODO remove code duplication
# TODO rewrite C loops as DGEMM?
# TODO can we get the permutational symmetry of intermediates from gristmill?

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
        index_sizes: dict = {"o": "nocc", "v": "nvir", "O": "naocc", "V": "navir", "b": "nbos"},
        force_optimize_kwarg: bool = False,
        string_subscript: bool = False,
) -> str:
    """
    Writes an `Expression` in the form of an einsum.
    """

    terms = []
    characters_inv = {}
    for key, val in characters.items():
        characters_inv.update({v: key for v in val})

    # Get the output
    res = output.symbol

    if output.symbol in reorder_axes:
        raise NotImplementedError

    if output.symbol in add_spins and not (output.symbol.startswith("x") and output.symbol[1:].isnumeric()):
        if any(index.spin in (0, 1) for index in output.indices):
            if not all(index.spin in (0, 1) for index in output.indices):
                raise NotImplementedError
            res += "_" + "".join(["ab"[index.spin] for index in output.indices])

    if output.symbol in add_occupancies:
        res += "_" + "".join([index.occupancy for index in output.indices])

    # If index_sizes is passed, write initialisation
    if index_sizes:
        # Update with dummies
        sizes = []
        for index in output.indices:
            key = characters_inv[index.character]
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
                res=res,
                zeros=zeros_function,
                shape=shape,
            ))

    # Write the terms
    for contraction in expression.contractions:
        tensors = []
        subscripts_in = []
        index_map = {}
        index_count = 0
        for tensor in contraction.tensors:
            symbol = tensor.symbol
            indices = tensor.indices

            if symbol in reorder_axes:
                indices = tuple(indices[i] for i in reorder_axes[symbol])

            if tensor.symbol in add_spins and not (tensor.symbol.startswith("x") and tensor.symbol[1:].isnumeric()):
                if any(index.spin in (0, 1) for index in indices):
                    if not all(index.spin in (0, 1) for index in indices):
                        raise NotImplementedError
                    symbol += "." + "".join(["ab"[index.spin] for index in indices])

            if tensor.symbol in add_occupancies:
                symbol += "." + "".join([index.occupancy for index in indices])

            tensors.append(symbol)
            subscripts_in_entry = []
            for index in indices:
                if index not in index_map:
                    index_map[index] = index_count
                    index_count += 1
                subscripts_in_entry.append(index_map[index])
            subscripts_in.append(tuple(subscripts_in_entry))

        subscripts_out = tuple(index_map[index] for index in output.indices)

        if string_subscript:
            index_map_inv = {val: key for key, val in index_map.items()}
            string = ""
            for subscript in subscripts_in:
                for s in subscript:
                    string += index_map_inv[s].character
                string += ","
            string = string[:-1] + "->"
            for s in subscripts_out:
                string += index_map_inv[s].character
            operands = "\"" + string + "\", " + ", ".join(tensors)
        else:
            operands = []
            for tensor, subscript in zip(tensors, subscripts_in):
                operands.append(tensor)
                operands.append(subscript)
            operands.append(subscripts_out)
            operands = ", ".join([str(op) for op in operands])

        term = "{res} += {fn}({operands}{opt}){op}{fac}".format(
                res=res,
                fn=einsum_function,
                operands=operands,
                op="" if contraction.factor == 1.0 else " * ",
                fac="" if contraction.factor == 1.0 else contraction.factor,
                opt=", optimize=True" if force_optimize_kwarg else "",
        )
        terms.append(term)

    return (indent * " ") + ("\n%s" % (indent * " ")).join(terms)


def write_c_loop(
        expression: "Expression",
        output: "Tensor",
        characters: dict = default_characters,
        add_occupancies: set = {"f", "v"},
        add_spins: set = All,
        reorder_axes: set = {},
        indent: int = 0,
        index_sizes: dict = {"o": "nocc", "v": "nvir", "O": "naocc", "V": "navir", "b": "nbos"},
        omp_collapse: int = None,
) -> str:
    """
    Writes an `Expression` in the form of a C for loop.
    """

    if any(len(contraction.tensors) > 2 for contraction in expression.contractions):
        raise ValueError("write_c_loop shouldn't be used for non-optimised expressions.")

    terms = []
    characters_inv = {}
    for key, val in characters.items():
        characters_inv.update({v: key for v in val})

    def make_index(indices):
        indices = [index if isinstance(index, str) else index.character for index in indices]
        string = ""
        for i in range(len(indices)):
            if i == (len(indices) - 1):
                string += indices[i]
            else:
                stride = "*".join([index_sizes[characters_inv[index]] for index in indices[i+1:]])
                string += "%s*%s+" % (indices[i], stride)
        return string

    def _find_symmetric_index_groups(tensor, perms):
        # Find symmetric index groups for a single tensor
        pots = {(sign, occ): [set() for i in tensor] for sign in (1, -1) for occ in characters.keys()}
        for perm, sign in perms:
            ptensor = [tensor[i] for i in perm]
            for i, p in enumerate(ptensor):
                pots[sign, characters_inv[p]][i].add(p)

        groups_dict = defaultdict(list)
        for occ in characters.keys():
            for i, pot in zip(tensor, pots[1, occ]):
                if len(pot):
                    groups_dict[tuple(pot)].append(i)
        groups_symm = tuple(tuple(x) for x in groups_dict.values() if len(x) > 1)

        groups_dict = defaultdict(list)
        for occ in characters.keys():
            for i, pot in zip(tensor, pots[-1, occ]):
                if len(pot):
                    groups_dict[tuple(pot)].append(i)
        groups_anti = tuple(tuple(x) for x in groups_dict.values() if len(x) > 1)

        return groups_symm, groups_anti

    # Get the output
    res = output.symbol

    if output.symbol in reorder_axes:
        raise NotImplementedError

    if output.symbol in add_spins and not (output.symbol.startswith("x") and output.symbol[1:].isnumeric()):
        if any(index.spin in (0, 1) for index in output.indices):
            if not all(index.spin in (0, 1) for index in output.indices):
                raise NotImplementedError
            res += "_" + "".join(["ab"[index.spin] for index in output.indices])

    if output.symbol in add_occupancies:
        res += "_" + "".join([index.occupancy for index in output.indices])

    # If index_sizes is passed, write initialisation
    if index_sizes:
        # Update with dummies
        sizes = []
        for index in output.indices:
            key = characters_inv[index.character]
            if index.spin in (None, 2):
                sizes.append(index_sizes[key])
            else:
                sizes.append(index_sizes[key] + "[%d]" % index.spin)
        if len(sizes) == 0:
            terms.append("double %s = 0.0;" % output.symbol)
        else:
            shape = "*".join(sizes)
            terms.append("double *{res} = (double*)calloc({shape}, sizeof(double));".format(res=res, shape=shape))

    # Write the terms
    for contraction in expression.contractions:
        term = ""
        i = 0

        # NOTE: output.perms gives the full symmetry of the output tensor,
        # which may be less than the symmetry of each contraction.
        groups_symm, groups_anti = _find_symmetric_index_groups([index.character for index in sorted(output.indices)], output.perms)

        if len(output.indices):
            term += "#pragma omp parallel for collapse(%d)\n" % (min(omp_collapse, len(output.indices)) if omp_collapse is not None else len(output.indices))
            for index in sorted(output.indices):
                start = 0
                for group in (groups_symm + groups_anti):
                    for j, ind in enumerate(group):
                        if index.character == ind:
                            start = 0 if j == 0 else group[j-1]

                key = characters_inv[index.character]
                term += "{indent}for (size_t {char} = {start}; {char} < {end}; {char}++) {{\n".format(
                        indent=" "*i,
                        char=index.character,
                        start=start,
                        end=index_sizes[key],
                )
                i += 1

            index_out = make_index(output.indices)
            term += "{indent}size_t iout_ = {index_out};\n".format(
                    indent=" "*i,
                    index_out=index_out,
            )

        for index in sorted(contraction.dummies):
            key = characters_inv[index.character]
            term += "{indent}for (size_t {char} = {start}; {char} < {end}; {char}++) {{\n".format(
                    indent=" "*i,
                    char=index.character,
                    start=0,
                    end=index_sizes[key],
            )
            i += 1

        if len(output.indices):
            term += "{indent}{res}[iout_] += ".format(
                    indent=" "*i,
                    res=res,
            )
        else:
            term += "{indent}{res} += ".format(
                    indent=" "*i,
                    res=res,
            )

        for tensor in contraction.tensors:
            symbol = tensor.symbol
            indices = tensor.indices

            if symbol in reorder_axes:
                indices = tuple(indices[i] for i in reorder_axes[symbol])

            if tensor.symbol in add_spins and not (tensor.symbol.startswith("x") and tensor.symbol[1:].isnumeric()):
                if any(index.spin in (0, 1) for index in indices):
                    if not all(index.spin in (0, 1) for index in indices):
                        raise NotImplementedError
                    symbol += "." + "".join(["ab"[index.spin] for index in indices])

            if tensor.symbol in add_occupancies:
                symbol += "." + "".join([index.occupancy for index in indices])

            index = make_index(tensor.indices)
            term += "{symbol}[{index}] * ".format(
                    symbol=symbol,
                    index=index,
            )

        term += str(contraction.factor)
        term += ";\n"

        if len(output.indices):
            if len(groups_symm) or len(groups_anti):
                term += " " * len(output.indices)
                term += "}" * (i - len(output.indices))
                term += "\n"

            for groups, sign in [(groups_symm, ""), (groups_anti, "-")]:
                for groups_perm in itertools.product(*[list(zip(*permutations_with_signs(range(len(group)))))[0] for group in groups]):
                    str_to_index = {}
                    for group, perm in zip(groups, groups_perm):
                        str_to_index.update({group[j]: group[k] for j, k in enumerate(perm)})
                    indices = tuple(str_to_index.get(ind.character, ind.character) for ind in output.indices)
                    index_perm = make_index(indices)
                    if index_perm == index_out:
                        continue
                    term += "{indent}{res}[{index_perm}] = {sign}{res}[iout_];  // {comment}\n".format(
                            indent=" "*len(output.indices),
                            res=res,
                            sign=sign,
                            index_perm=index_perm,
                            comment="".join(list(indices))+"->"+sign+"".join([ind.character for ind in output.indices]),
                    )

        if len(groups_symm) or len(groups_anti):
            term += "}" * len(output.indices)
        else:
            term += "}" * i
        terms.append(term)

    terms = "\n".join(terms)

    return "\n".join([(" " * indent) + term for term in terms.split("\n")])


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
            if not kwargs.get("string_subscript", False):
                # The line is i.e. 'x0 += einsum(x, (0, 1, 2, 3), y, (4, 1, 5, 3), (4, 0, 5, 2))'
                # and we want to find ['x', 'y']
                tensors = line[line.index("("):-(line[::-1].index(")")+1)]
                tensors = tensors.replace("(", " ").replace(")", " ").replace(",", " ")
                tensors = [x for x in tensors.split() if not x.isdigit()]
            else:
                raise NotImplementedError

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


def write_opt_c_loops(
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

    # Build C loops
    c_loops = flatten([
            write_c_loop(expression, output, **kwargs).split("\n")
            for expression, output in zip(new_expressions, new_outputs)
    ])

    # Since we split the contractions we need to remove duplicate
    # initialisation statements
    seen = set()
    delete = set()
    for i, line in enumerate(c_loops):
        if " = 0.0;" in line or "calloc" in line:
            if line in seen:
                delete.add(i)
            seen.add(line)
    c_loops = [
            line
            for i, line in enumerate(c_loops)
            if i not in delete
    ]

    # Find the last appearence of each tensor
    final_line = defaultdict(int)
    for i, line in enumerate(c_loops):
        if not (" = 0.0;" in line or "calloc" in line or any(line.lstrip().startswith(x) for x in ("for", "}", "#pragma"))):
            tensors = re.findall(r"(?:^|\s)(\w+(?:\.\w+)*)\[", line)
            for tensor in tensors:
                if tensor.startswith("x"):
                    final_line[tensor] = i

    # Add delete statements after tensors are used the final time
    del_lists = defaultdict(list)
    for tensor, line in final_line.items():
        del_lists[line].append(tensor)
    for line, del_list in sorted(list(del_lists.items()))[::-1]:
        for arr in del_list:
            inc = 1
            while c_loops[line+inc].startswith(" " * (kwargs.get("indent", 0)+1)):
                inc += 1
            c_loops.insert(line+inc+1, "%sfree(%s);" % (kwargs.get("indent", 0) * " ", arr))

    return "\n".join(c_loops)
