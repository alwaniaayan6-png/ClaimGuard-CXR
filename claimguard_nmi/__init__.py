"""ClaimGuard Path B — image-grounded claim verification.

See ARCHITECTURE_PATH_B.md for the design document. This package is a
clean break from the v1-v4 text-only pipeline (which lives under
``verifact.inference`` and ``verifact.scripts``). Do NOT mix imports
between the two packages — they encode different ground-truth
assumptions.
"""

__version__ = "0.1.0-alpha"
