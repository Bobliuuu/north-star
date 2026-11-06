import pathlib, sys
src = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(src)); sys.path.insert(0, str(src / "weightedit" / "tests"))  # tiny.py
