"""SL-02 import boundary: features/storelab/core/ imports only the standard library and
numpy, scipy, pandas, pyarrow, pydantic and cvxpy (tickets.md SL-02 and section 3.1).
"""
from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path

from features.storelab import core

CORE_DIR = Path(core.__file__).resolve().parent
CORE_PACKAGE = "features.storelab.core"
ALLOWED = {"numpy", "scipy", "pandas", "pyarrow", "pydantic", "cvxpy"}
STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def allowed(module: str) -> bool:
    top = module.split(".")[0]
    return top in STDLIB or top in ALLOWED or module == CORE_PACKAGE or module.startswith(CORE_PACKAGE + ".")


def violations(source: str, module: str, is_package: bool = False) -> list[str]:
    """Imports in ``source`` (the module ``module``) that leave the allowed set."""
    package = module.split(".") if is_package else module.split(".")[:-1]
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += ["import %s" % a.name for a in node.names if not allowed(a.name)]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1]
                target = ".".join(base + ([node.module] if node.module else []))
                if not target.startswith(CORE_PACKAGE):
                    found.append("from %s%s import (resolves to %s)" % ("." * node.level, node.module or "", target))
            elif not allowed(node.module):
                found.append("from %s import" % node.module)
            elif node.module == "importlib" and any(a.name == "import_module" for a in node.names):
                found.append("from importlib import import_module")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in ("import_module", "__import__"):
                found.append("dynamic import %s()" % name)
    return found


def core_modules():
    for path in sorted(CORE_DIR.rglob("*.py")):
        parts = list(path.relative_to(CORE_DIR).with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        yield path, ".".join([CORE_PACKAGE] + (parts[:-1] if is_package else parts)), is_package


class ImportBoundaryTest(unittest.TestCase):
    """AC-5: core/ imports only the standard library and the allow-list."""

    def test_core_imports_only_the_standard_library_and_the_allow_list(self):
        modules = list(core_modules())
        self.assertGreaterEqual(len(modules), 5)
        for path, module, is_package in modules:
            with self.subTest(module=module):
                self.assertEqual(violations(path.read_text(encoding="utf-8"), module, is_package), [])

    def test_the_checker_flags_imports_outside_the_boundary(self):
        module = CORE_PACKAGE + ".example"
        flagged = ["import yaml", "import requests as r", "from features.storelab import cli",
                   "from features.storelab.config_files import load_config", "from ..config_files import x",
                   "from .. import cli", "import importlib\nimportlib.import_module('yaml')", "__import__('yaml')",
                   "from importlib import import_module", "import features.storelab.cli"]
        accepted = ["from __future__ import annotations", "import numpy.random", "from scipy import stats",
                    "import pandas as pd, pyarrow", "from pydantic import BaseModel", "import cvxpy",
                    "from . import config", "from .config import RunConfig", "from importlib import metadata",
                    "from features.storelab.core.config import RunConfig", "import json, subprocess"]
        for source in flagged:
            with self.subTest(flagged=source):
                self.assertNotEqual(violations(source, module), [])
        for source in accepted:
            with self.subTest(accepted=source):
                self.assertEqual(violations(source, module), [])

    def test_importing_core_loads_no_service_module_and_no_yaml(self):
        code = ("import sys, features.storelab.core as c, features.storelab.core.config, "
                "features.storelab.core.manifest, features.storelab.core.versioning, "
                "features.storelab.core.model_registry\n"
                "print('\\n'.join(sorted(m for m in sys.modules if m == 'yaml' or "
                "(m.startswith('features') and not m.startswith('features.storelab.core')))))")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=CORE_DIR.parents[2],
                                capture_output=True, text=True, check=True)
        self.assertEqual(sorted(result.stdout.split()), ["features", "features.storelab"])


if __name__ == "__main__":
    unittest.main()
