from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from key001_common import gen7 as generate_input, main7 as main
