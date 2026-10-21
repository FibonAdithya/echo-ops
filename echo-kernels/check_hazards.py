#!/usr/bin/env python3
"""Usage: check_hazards.py file.ptx -- exits 1 and lists every instruction in ECHO's own
library PTX that could differ between GPUs or drivers: approximate maths, floating-point
mul/add/sub without an explicit rounding mode (the driver may fuse them), float atomics,
and hardware-dependent special registers. ECHO's library must contain none of these."""
import re
import sys

INSTR = re.compile(r"^\s*(?:@!?%\w+\s+)?([a-z][a-z0-9_.]*)\b(.*)$")
SPECIAL = re.compile(r"%(clock64|clock|globaltimer\w*|smid|nsmid|warpid|nwarpid)\b")
FLOAT_T = ("f16", "f16x2", "bf16", "bf16x2", "f32", "f64")
ROUND = ("rn", "rz", "rm", "rp")

def hazard(op):
    parts = op.split(".")
    base = parts[0]
    if "approx" in parts or (base == "div" and "full" in parts):
        return "approximate maths"
    if base in ("ex2", "lg2", "sin", "cos", "tanh", "rsqrt"):
        return "approximate maths"
    if base in ("mul", "add", "sub", "mad") and parts[-1] in FLOAT_T and not any(p in ROUND for p in parts):
        return "fusable float op without rounding mode"
    if base in ("atom", "red") and parts[-1] in FLOAT_T:
        return "float atomic"
    if base == "activemask":
        return "scheduling-dependent"
    return None

bad = []
for n, line in enumerate(open(sys.argv[1]), 1):
    m = INSTR.match(line)
    if m and not line.lstrip().startswith(("//", ".")):
        why = hazard(m.group(1))
        if why:
            bad.append(f"{n}: {line.strip()}  <- {why}")
    if SPECIAL.search(line):
        bad.append(f"{n}: {line.strip()}  <- hardware-dependent special register")
if bad:
    print(f"{len(bad)} hazards in {sys.argv[1]}:", *bad[:50], sep="\n", file=sys.stderr)
    sys.exit(1)
print(f"no hazards in {sys.argv[1]}")
