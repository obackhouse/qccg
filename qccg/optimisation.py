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

import qccg
from qccg.index import ExternalIndex, DummyIndex
from qccg.tensor import ATensor, Scalar, Fock, ERI, CDERI, FermionicAmplitude, RDM1, RDM2, Delta
from qccg.contraction import Contraction, Expression


DEFAULT_SIZES = {
        "o": 10,
        "v": 50,
        "O": 1,
        "V": 1,
        "x": 200,
        "b": 2,
}

OPMIN_COMMAND = "python2 /home/olli/git/opmin/src/opmin.py"

default_characters = {
        "o": "ijklmnot",
        "v": "abcdefgh",
        "O": "IJKLMNOT",
        "V": "ABCDEFGH",
        "b": "wxyz",
        "x": "PQRSUV",
}


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
            elif symbol.startswith("t") or symbol.startswith("l") or symbol.startswith("r"):
                lower = tuple(index for index in indices if index.occupancy.lower() == "o")
                upper = tuple(index for index in indices if index.occupancy.lower() == "v")
                if symbol.startswith("l"):
                    lower, upper = upper, lower
                output = FermionicAmplitude(symbol, lower, upper)
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
                        if len(indices) == 4:
                            tensors[i] = ERI(indices)
                        else:
                            tensors[i] = CDERI(indices)
                    elif symbol.startswith("t") or symbol.startswith("l") or symbol.startswith("r"):
                        lower = tuple(index for index in indices if index.occupancy.lower() == "o")
                        upper = tuple(index for index in indices if index.occupancy.lower() == "v")
                        if symbol.startswith("l"):
                            lower, upper = upper, lower
                        tensors[i] = FermionicAmplitude(symbol, lower, upper)
                    else:
                        raise ValueError(symbol)

                contractions.append(Contraction((factor, *tensors)))

            opt_outputs.append(output)
            opt_expressions.append(Expression(contractions))

    # Clean up
    os.system("rm -f opmin_inp.eq")
    os.system("rm -f opmin_inp.eq.out")
    os.system("rm -f opmin_tester.py")

    return opt_expressions, opt_outputs


def optimise_expression_gristmill(expressions, outputs, sizes=DEFAULT_SIZES, strat="trav", **gristmill_kwargs):
    """
    Optimise an expression using gristmill.
    """

    from dummy_spark import SparkContext
    import drudge
    import gristmill
    import sympy

    ctx = SparkContext()
    dr = drudge.Drudge(ctx)

    # Check the spin
    indices = set()
    for output in outputs:
        for index in output.indices:
            if index.occupancy != "x":
                indices.add(index)
    for expression in expressions:
        for contraction in expression.contractions:
            for tensor in contraction.tensors:
                for index in tensor.indices:
                    if index.occupancy != "x":
                        indices.add(index)
    spins = set(index.spin for index in indices)
    ghf = spins == {None}
    uhf = spins in ({0}, {1}, {0, 1})
    rhf = spins == {2}
    assert sum([ghf, uhf, rhf]) == 1

    # Register the dummies
    substs = {}
    ranges = {}
    if "o" in sizes and "O" not in sizes:
        sizes["O"] = sizes["o"]
    if "v" in sizes and "V" not in sizes:
        sizes["V"] = sizes["v"]
    for sector in sizes.keys():
        for spin in ["", "a", "b"]:
            size = sympy.Symbol("N"+sector+spin)
            rng = drudge.Range(sector+spin, 0, size)
            inds = [sympy.Symbol("%s%d%s" % (sector, i, spin)) for i in range(50)]
            dr.set_dumms(rng, inds)
            substs[size] = sizes[sector]
            ranges[sector+spin] = rng
    dr.add_resolver_for_dumms()

    # Get the drudge einsums
    terms = []
    perms = {}
    index_counter = defaultdict(int)
    index_map = {}
    for expression, output in zip(expressions, outputs):
        if output.rank != 0:
            base = sympy.IndexedBase(output.symbol + "_" + "".join([i.occupancy for i in output.indices]))
            inds = []
            for index in output.indices:
                if index not in index_map:
                    sector = index.occupancy
                    spin = {0: "a", 1: "b"}.get(index.spin, "")
                    inds.append("%s%d%s" % (sector, index_counter[sector], spin))
                    index_map[index] = inds[-1]
                    index_counter[sector] += 1
                else:
                    inds.append(index_map[index])
            out = base
            index_to_range = lambda index: ranges["".join([x for x in index if not x.isdigit()])]
            lhs_ranges = [(sympy.Symbol(index), index_to_range(index)) for index in inds]
        else:
            base = sympy.Symbol(output.symbol + "_")
            out = base
            lhs_ranges = []

        expr = 0
        for contraction in expression.contractions:
            einst = contraction.factor
            for tensor in contraction.tensors:
                symbol = tensor.symbol + "_" + "".join([i.occupancy for i in tensor.indices])
                base = sympy.IndexedBase(symbol)
                inds = []
                for index in tensor.indices:
                    if index not in index_map:
                        sector = index.occupancy
                        spin = {0: "a", 1: "b"}.get(index.spin, "")
                        inds.append("%s%d%s" % (sector, index_counter[sector], spin))
                        index_map[index] = inds[-1]
                        index_counter[sector] += 1
                    else:
                        inds.append(index_map[index])
                inds = tuple(sympy.Symbol(x) for x in inds)
                einst *= base[inds]
                if symbol not in perms:
                    perms[symbol] = list(tensor.perms)
                else:
                    perms[symbol] = [p for p in perms[symbol] if p in tensor.perms]
            expr += einst

        rhs = dr.einst(expr)
        terms.append(dr.define(out, *lhs_ranges, rhs))

    # Set the symmetry permutations
    for symbol, perm in perms.items():
        base = sympy.IndexedBase(symbol)
        perm_list = [
                drudge.Perm(list(p), {-1: drudge.NEG, 1: drudge.IDENT}[a])
                for p, a in perm
                if tuple(p) != tuple(range(len(p)))
        ]
        if not len(perm_list):
            perm_list = [None]
        dr.set_symm(base, *perm_list)

    # Optimise expression
    terms = gristmill.optimize(
            terms,
            substs=substs,
            contr_strat=getattr(gristmill.ContrStrat, strat.upper()),
            interm_fmt="x{}",
            **gristmill_kwargs,
    )

    # Convert expressions back
    expressions = []
    outputs = []
    for term in terms:
        externals = set()
        index_map = {}
        index_counter = defaultdict(int)
        if isinstance(term.lhs, sympy.Symbol):
            symbol = term.lhs.name
            inds = []
        else:
            symbol = term.lhs.base.name
            inds = []
            for index in term.lhs.indices:
                sector = index.name[0]
                n = int(index.name[1:].strip("a").strip("b"))
                if index in index_map:
                    char = index_map[index]
                else:
                    char = default_characters[sector][index_counter[sector.lower()]]
                    index_map[index] = char
                    index_counter[sector.lower()] += 1
                externals.add(char)
                spin = None if ghf else (2 if rhf else {"a": 0, "b": 1}[repr(index)[-1]])
                inds.append(ExternalIndex(char, sector, spin))

        # Remove occupancy tags
        if not (symbol.startswith("x") and all(x.isdigit() for x in symbol[1:])):
            while symbol[-1] != "_":
                symbol = symbol[:-1]
            symbol = symbol[:-1]

        if len(inds) == 0:
            output = Scalar(symbol)
        elif symbol.startswith("x"):
            output = ATensor(symbol, inds)
        elif symbol.startswith("delta"):
            output = Delta(inds)
        elif symbol.startswith("γ") or symbol.startswith("Γ") or symbol.startswith("rdm"):
            if len(inds) == 2:
                output = RDM1(inds)
            else:
                output = RDM2(inds)
        elif symbol.startswith("t") or symbol.startswith("l") or symbol.startswith("r"):
            lower = tuple(index for index in inds if index.occupancy.lower() == "o")
            upper = tuple(index for index in inds if index.occupancy.lower() == "v")
            if symbol.startswith("l"):
                lower, upper = upper, lower
            output = FermionicAmplitude(symbol, lower, upper)
        else:
            raise ValueError(symbol)

        contractions = []
        for amp in [t.amp for t in term.rhs_terms]:
            factor = float(sympy.prod(amp.atoms(sympy.Number)))
            contr = factor

            for tensor in amp.atoms(sympy.Indexed):
                symbol = tensor.base.name
                inds = []
                for index in tensor.indices:
                    sector = index.name[0]
                    n = int(index.name[1:].strip("a").strip("b"))
                    if index in index_map:
                        char = index_map[index]
                    else:
                        char = default_characters[sector][index_counter[sector.lower()]]
                        index_map[index] = char
                        index_counter[sector.lower()] += 1
                    spin = None if ghf else (2 if rhf else {"a": 0, "b": 1}[repr(index)[-1]])
                    if char in externals:
                        inds.append(ExternalIndex(char, sector, spin))
                    else:
                        inds.append(DummyIndex(char, sector, spin))

                # Remove occupancy tags
                if not (symbol.startswith("x") and all(x.isdigit() for x in symbol[1:])):
                    while symbol[-1] != "_":
                        symbol = symbol[:-1]
                    symbol = symbol[:-1]

                if len(inds) == 0:
                    tensor = Scalar(symbol)
                elif symbol.startswith("x") or symbol.startswith("denom"):
                    tensor = ATensor(symbol, inds)
                elif symbol.startswith("f"):
                    tensor = Fock(inds)
                elif symbol.startswith("v"):
                    if len(inds) == 4:
                        tensor = ERI(inds)
                    else:
                        tensor = CDERI(inds)
                elif symbol.startswith("delta"):
                    tensor = Delta(inds)
                elif symbol.startswith("γ") or symbol.startswith("Γ") or symbol.startswith("rdm"):
                    if len(inds) == 2:
                        tensor = RDM1(inds)
                    else:
                        tensor = RDM2(inds)
                elif symbol.startswith("t") or symbol.startswith("l") or symbol.startswith("r") or symbol.startswith("c"):
                    lower = tuple(index for index in inds if index.occupancy.lower() == "o")
                    upper = tuple(index for index in inds if index.occupancy.lower() == "v")
                    if symbol.startswith("l"):
                        lower, upper = upper, lower
                    tensor = FermionicAmplitude(symbol, lower, upper)
                else:
                    raise ValueError(symbol)

                contr = contr * tensor

            for contr in contr.canonicalise(canonicalise_dummies=False):
                contractions.append(contr)

        # Determine the permutations of the intermediates
        if output.symbol.startswith("x"):
            all_perms = []
            for contr in contractions:
                perms = {(tuple(range(output.rank)), 1)}
                if len(contr.tensors) == 1:
                    # Just a transpose
                    trans = tuple(contr.tensors[0].indices.index(i) for i in output.indices)
                    for perm in contr.tensors[0].perms:
                        perms.add((tuple(perm[0][p] for p in trans), perm[1]))
                elif len(contr.tensors) == 2:
                    for perm1 in contr.tensors[0].perms:
                        for perm2 in contr.tensors[1].perms:
                            perm = []
                            for p in perm1[0]:
                                index = contr.tensors[0].indices[p]
                                if index in output.indices:
                                    perm.append(output.indices.index(index))
                            for p in perm2[0]:
                                index = contr.tensors[1].indices[p]
                                if index in output.indices:
                                    perm.append(output.indices.index(index))
                            perms.add((tuple(perm), perm1[1]*perm2[1]))
                all_perms.append(perms)
            output._perms = sorted(set.intersection(*all_perms))

        # FIXME - the dummies are now messed up. Simplifying this breaks things.
        expression = Expression(contractions, simplify=False)
        expressions.append(expression)
        outputs.append(output)

    return expressions, outputs


del DEFAULT_SIZES, OPMIN_COMMAND
