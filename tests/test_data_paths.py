from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from hr.core.data_paths import load_data_paths, resolve_data_path, resolve_path_spec


class DataPathRegistryTests(unittest.TestCase):
    def test_resolve_dp_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "data_paths.yaml"
            cfg.write_text(
                textwrap.dedent(
                    """
                    harm_kl:
                      preferences_dir: "harm_kl/data/preferences"
                    """
                ).strip()
            )
            registry = load_data_paths(cfg)
            self.assertEqual(
                resolve_path_spec("dp://harm_kl.preferences_dir", registry),
                Path("harm_kl/data/preferences"),
            )

    def test_changing_registry_redirects_resolved_path(self) -> None:
        reg1 = {"harm_kl": {"preferences_dir": "harm_kl/data/preferences"}}
        reg2 = {"harm_kl": {"preferences_dir": "tmp/alt/preferences"}}
        self.assertEqual(resolve_data_path(reg1, "harm_kl.preferences_dir"), Path("harm_kl/data/preferences"))
        self.assertEqual(resolve_data_path(reg2, "harm_kl.preferences_dir"), Path("tmp/alt/preferences"))


if __name__ == "__main__":
    unittest.main()
