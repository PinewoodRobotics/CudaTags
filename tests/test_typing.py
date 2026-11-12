"""Run without native libraries: python -m unittest discover -s tests."""
import ast
import importlib
import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / 'src/main/python'
sys.path.insert(0, str(PACKAGE))

class TypingTests(unittest.TestCase):
    def test_import_without_backend(self):
        package = importlib.import_module('cuda_tags')
        self.assertNotIn('cuda_tags.cuda_tags', sys.modules)
        with self.assertRaises(AttributeError):
            getattr(package, 'not_a_symbol')

    def test_stub_covers_exports(self):
        package = importlib.import_module('cuda_tags')
        tree = ast.parse((PACKAGE / 'cuda_tags/__init__.pyi').read_text())
        names = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        names |= {n.target.id for n in tree.body if isinstance(n, ast.AnnAssign)}
        names |= {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(set(package.__all__), names)
        self.assertTrue((PACKAGE / 'cuda_tags/py.typed').exists())
