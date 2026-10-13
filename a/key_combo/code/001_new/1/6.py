from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from key001_common import gen6 as generate_input, main6 as main
