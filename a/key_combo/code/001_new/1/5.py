from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from key001_common import gen5 as generate_input, main5 as main
