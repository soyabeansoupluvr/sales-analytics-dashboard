"""Test configuration — make the project root importable so tests can use
``import src.<module>``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
