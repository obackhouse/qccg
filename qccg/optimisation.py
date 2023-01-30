"""
Optimisation

    See:
     - https://doi.org/10.1016/j.procs.2012.04.044
     - https://doi.org/10.1007/11758501_39

    OpMin:
     - http://hpcrl.cse.ohio-state.edu/wiki/index.php/TCE-OpMin#Input_Files
"""

import os
from collections import defaultdict

from qccg.index import ExternalIndex
from qccg.tensor import ATensor, Scalar, Fock, ERI, FermionicAmplitude
from qccg.contraction import Contraction, Expression


DEFAULT_SIZES = {
        "o": 10,
        "v": 50,
}

OPMIN_COMMAND = "python2 /home/olli/git/opmin/src/opmin.py"


def optimise_expression(expressions, outputs, sizes=DEFAULT_SIZES, opmin=OPMIN_COMMAND):
    """
    Optimise an expression using OpMin.
    """

    if not isinstance(expressions, (list, tuple)):
        expressions = [expressions]
        outputs = [outputs]

    assert len(expressions) == len(outputs)

    # Write the input file
    inp = open("opmin_inp.eq", "w")
    inp.write("{\n")

    for key, size in sizes.items():
        inp.write("  range %s = %d;\n" % (key, size))

    inp.write("\n")

    indices = defaultdict(set)
    index_map = {}
    for expression in expressions:
        for contraction in expression.contractions:
            for index in contraction.indices:
                indices[index.occupancy].add(index.character)
                index_map[index.character] = index
    for sector, indices in indices.items():
        inp.write("  index %s = %s;\n" % (", ".join(sorted(indices)), sector))

    inp.write("\n")

    tensors = set()
    for expression in expressions:
        for contraction in expression.contractions:
            for tensor in contraction.tensors:
                occupancies = tuple(index.occupancy for index in tensor.indices)
                tensors.add((tensor.symbol, occupancies))
    for symbol, occupancies in tensors:
        inp.write("  array %s%s%s([%s][]);\n" % (
                symbol,
                "_" if len(occupancies) else "",
                "".join(occupancies),
                ",".join(occupancies)
            )
        )

    for output in outputs:
        symbol = output.symbol
        occupancies = tuple(index.occupancy for index in output.indices)
        indices = tuple(index.character for index in output.indices)
        inp.write("  array %s%s%s([%s][]);\n" % (
                symbol,
                "_" if len(occupancies) else "",
                "".join(occupancies),
                ",".join(occupancies),
            )
        )

    inp.write("\n")

    for output, expression in zip(outputs, expressions):
        symbol = output.symbol
        occupancies = tuple(index.occupancy for index in output.indices)
        indices = tuple(index.character for index in output.indices)

        if len(indices) == 0:
            inp.write("  %s =\n" % symbol)
        else:
            inp.write("  %s_%s[%s] =\n" % (symbol, "".join(occupancies), ",".join(indices)))

        for i, contraction in enumerate(expression.contractions):
            inp.write("    ")

            if i > 0:
                sign = "+" if contraction.factor >= 0 else "-"
                inp.write(sign + " ")
                inp.write(str(abs(round(contraction.factor, 10))))
            else:
                inp.write(str(round(contraction.factor, 10)))

            for tensor in contraction.tensors:
                occupancies = tuple(index.occupancy for index in tensor.indices)
                indices = tuple(index.character for index in tensor.indices)
                inp.write(
                        " * %s%s%s[%s]" % (
                            tensor.symbol,
                            "_" if len(indices) else "",
                            "".join(occupancies),
                            ",".join(indices)
                        )
                )

            if (i + 1) == len(expression.contractions):
                inp.write(";")

            inp.write("\n")

    inp.write("}")

    inp.close()


    # Dispatch OpMin
    os.system("%s -c -f opmin_inp.eq" % opmin)


    # Get the output
    with open("opmin_inp.eq.out", "r") as f:
        inp = f.readlines()
        inp = [x.strip() for x in inp]

    
    # Parse the output
    opt_expressions = []
    opt_outputs = []
    intermediate_counter = 0
    intermediate_map = {}
    for line in inp:
        if not any(line.startswith(name) for name in ("range", "index", "array", "{", "}")) \
                and len(line.split()):
            lhs, rhs = line.split(" = ")

            # Get output symbol and indices
            if "[" in lhs:
                symbol = lhs.split("[")[0]
                indices = lhs.split("[")[1].strip("]").split(",")
                indices = tuple(index_map[index] for index in indices)
            else:
                symbol = lhs
                indices = tuple()

            # Remove occupancy and spin
            if "_" in symbol and not symbol.startswith("_a"):
                while not any(symbol == output.symbol for output in outputs):
                    symbol = "_".join(symbol.split("_")[:-1])

            # Get new intermediate name
            if symbol.startswith("_a"):
                if symbol not in intermediate_map:
                    intermediate_map[symbol] = "x%d" % intermediate_counter
                    intermediate_counter += 1
                symbol = intermediate_map[symbol]

            # Get the tensor
            # FIXME will be tricky with excited amplitudes - implement
            # support for covariant/contravariant indices throughout
            if len(indices) == 0:
                output = Scalar(symbol)
            elif symbol.startswith("x"):
                indices = tuple(
                        ExternalIndex(index.character, index.occupancy, index.spin)
                        for index in indices
                )
                output = ATensor(symbol, indices)
            elif symbol.startswith("t") or symbol.startswith("l"):
                upper = indices[:len(indices)//2]
                lower = indices[len(indices)//2:]
                output = FermionicAmplitude(symbol, upper, lower)
            else:
                raise ValueError(symbol)

            rhs = rhs.lstrip("(").rstrip(");")
            rhs_parts = rhs.split(" + ")
            rhs_parts = [x.lstrip("(").rstrip(")") for x in rhs_parts]

            contractions = []
            for part in rhs_parts:
                # Get tensors and factors
                tensors = part.split(" * ")
                if all(x in "1234567890.-" for x in tensors[0]):
                    factor, tensors = float(tensors[0]), tensors[1:]
                else:
                    factor = 1.0

                for i, tensor in enumerate(tensors):
                    # Update sign if tensors have minus signs
                    if tensor.startswith("-"):
                        factor *= -1
                        tensor = tensor.lstrip("-")

                    # Get symbol and indices
                    if "[" in tensor:
                        symbol = tensor.split("[")[0]
                        indices = tensor.split("[")[1].rstrip("]").split(",")
                        indices = tuple(index_map[index] for index in indices)
                    else:
                        symbol = tensor
                        indices = tuple()

                    # Remove occupancy and spin
                    if "_" in symbol and not symbol.startswith("_a"):
                        symbol = symbol.split("_")[0]

                    # Get new intermediate name
                    if symbol.startswith("_a"):
                        if symbol not in intermediate_map:
                            intermediate_map[symbol] = "x%d" % intermediate_counter
                            intermediate_counter += 1
                        symbol = intermediate_map[symbol]

                    # Get the tensor
                    # FIXME will be tricky with excited amplitudes - implement
                    # support for covariant/contravariant indices throughout
                    if symbol.startswith("x"):
                        indices = tuple(
                                ExternalIndex(index.character, index.occupancy, index.spin)
                                for index in indices
                        )
                        tensors[i] = ATensor(symbol, indices)
                    elif symbol.startswith("f"):
                        tensors[i] = Fock(indices)
                    elif symbol.startswith("v"):
                        tensors[i] = ERI(indices)
                    elif symbol.startswith("t") or symbol.startswith("l"):
                        upper = indices[:len(indices)//2]
                        lower = indices[len(indices)//2:]
                        tensors[i] = FermionicAmplitude(symbol, upper, lower)
                    else:
                        raise ValueError(symbol)

                contractions.append(Contraction((factor, *tensors)))

            opt_outputs.append(output)
            opt_expressions.append(Expression(contractions))

    # Clean up
    #os.system("rm -f opmin_inp.eq")
    #os.system("rm -f opmin_inp.eq.out")
    #os.system("rm -f opmin_tester.py")

    return opt_expressions, opt_outputs


del DEFAULT_SIZES, OPMIN_COMMAND
