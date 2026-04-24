"""
conftest.py
===========
Aggiunge la root del progetto al sys.path in modo che pytest
trovi i moduli (materials, elements, ecc.) indipendentemente
da dove viene lanciato.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
