from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_spec = spec_from_file_location("_arc_manual_impls", Path(__file__).parents[1] / "arc_manual_impls.py")
_impl = module_from_spec(_spec)
_spec.loader.exec_module(_impl)

generate_input = _impl.p4_2_generate_input
main = _impl.p4_2_main
