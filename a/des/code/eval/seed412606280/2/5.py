from importlib.machinery import SourceFileLoader
from pathlib import Path

_helper = SourceFileLoader("_arc_fallback_common", str(Path(__file__).resolve().parents[2] / "_arc_fallback_common.py")).load_module()
main = _helper.main
generate_input = _helper.generate_input
