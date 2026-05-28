#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))

if __name__ == '__main__':
    args = [sys.argv[0]]
    for arg in sys.argv[1:]:
        args.append('--cfg' if arg == '--config' else arg)
    sys.argv = args
    import eval
    eval.main()
