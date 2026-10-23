import sys
from pathlib import Path

# DAOcase sits next to SimpleSimulator and uses its agent and simulator.
SIMULATOR_ROOT = Path(__file__).resolve().parent.parent / "SimpleSimulator"
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))
