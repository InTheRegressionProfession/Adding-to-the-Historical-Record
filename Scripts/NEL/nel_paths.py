"""
NEL Utilities Script
These are shared helper functions and script components to ensure consistency in variables among different script run files.

Code developed with the assistance of Claude Opus 5/5.5

"""

import os
import sys
from pathlib import Path

NEL_DIR = Path(__file__).resolve().parent

_candidates = ([Path(os.environ["NER_FOLDER"])] if os.environ.get("NER_FOLDER") else []) \
    + [NEL_DIR.parent / "NER", NEL_DIR]
for _cand in _candidates:
    if (_cand / "ner_utils.py").exists() and (_cand / "compare_broad.py").exists():
        NER_FOLDER = _cand.resolve()
        break
else:
    sys.exit("Can't find the NER folder (the one with ner_utils.py and compare_broad.py).\n"
             f"Looked in: {', '.join(str(c) for c in _candidates)}.\n"
             "Put the NEL folder next to NER, or set the NER_FOLDER environment variable to its path.")

if str(NER_FOLDER) not in sys.path:
    sys.path.insert(1, str(NER_FOLDER))
