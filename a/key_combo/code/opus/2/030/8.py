from importlib.machinery import SourceFileLoader

_common = SourceFileLoader("arc_codegen_missing", "/home/lixinyu/snap/a/key_combo/code/opus/2/_arc_codegen_missing.py").load_module()

main = _common.main_common_8
generate_input = _common.generate_common_8
