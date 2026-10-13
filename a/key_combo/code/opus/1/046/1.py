import sys
from pathlib import Path

import numpy as np

_BASE = Path(__file__).resolve().parents[1]
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from arc_codegen_local_common import make_generate_input, make_main


main = make_main(46, 1)
generate_input = make_generate_input(46, 1)
