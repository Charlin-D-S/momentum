"""Point d'entree : python -m challenger_benchmark challenger.yaml"""
import sys
from .pipeline import run

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python -m challenger_benchmark <challenger.yaml>")
        sys.exit(1)
    run(sys.argv[1])
