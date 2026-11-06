"""TAGE-style online token-history predictor for self-speculation (CPU simulator).

Zero new weights: all state is built online from committed tokens.
See docs/TAGE.md. Evaluation here is an oracle-verifier simulation, NOT a North result.
"""
from .predictor import TagePredictor, TageConfig
from .baselines import PromptLookup, SuffixLast, SuffixFreq, SuffixCal
from .sim import simulate, SimResult
