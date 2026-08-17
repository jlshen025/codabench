"""Make the shipped modules (udiva_data, sdl, cv, predictors, pack) importable from dev/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
