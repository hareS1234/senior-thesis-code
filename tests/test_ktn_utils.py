from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from generate_basin_keep_lists import read_min_list
from ktn_utils import compute_mfpt_from_Q, leading_relaxation_times
from quantitative_keeplist_checks import build_Qeff_for_deltaE, main as quantitative_main
from config import MarkovFilePaths
from run_all_build import model_is_complete


class KTNUtilityTests(unittest.TestCase):
    def test_two_state_column_generator(self):
        Q = sp.csr_matrix([[-2.0, 3.0], [2.0, -3.0]])

        self.assertAlmostEqual(compute_mfpt_from_Q(Q, [0], [1]), 0.5)
        self.assertAlmostEqual(compute_mfpt_from_Q(Q, [1], [0]), 1.0 / 3.0)
        np.testing.assert_allclose(leading_relaxation_times(Q, 1), [0.2])

    def test_row_generator_option(self):
        Q_row = sp.csr_matrix([[-2.0, 2.0], [3.0, -3.0]])
        value = compute_mfpt_from_Q(
            Q_row,
            [0],
            [1],
            column_generator=False,
        )
        self.assertAlmostEqual(value, 0.5)

    def test_three_state_passage_time(self):
        Q = sp.csr_matrix(
            [
                [-2.0, 0.0, 0.0],
                [2.0, -4.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        )
        self.assertAlmostEqual(compute_mfpt_from_Q(Q, [0], [2]), 0.75)
        self.assertAlmostEqual(compute_mfpt_from_Q(Q, [1], [2]), 0.25)

    def test_generator_orientation_is_checked(self):
        Q_row = sp.csr_matrix([[-2.0, 2.0], [3.0, -3.0]])
        with self.assertRaisesRegex(ValueError, "columns"):
            compute_mfpt_from_Q(Q_row, [0], [1])

    def test_endpoint_file_with_leading_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            endpoint_file = Path(tmp) / "min.A"
            endpoint_file.write_text("2\n4\n9\n", encoding="utf-8")
            self.assertEqual(read_min_list(endpoint_file), [4, 9])


class BasinReductionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dps_dir = Path(self.temp_dir.name)
        self.dps_dir.joinpath("min.data").write_text(
            "0.0\n1.0\n2.0\n3.0\n",
            encoding="utf-8",
        )
        self.dps_dir.joinpath("ts.data").write_text(
            "0.5 0 0 1 2\n2.5 0 0 3 4\n",
            encoding="utf-8",
        )
        self.dps_dir.joinpath("min.A").write_text("1\n", encoding="utf-8")
        self.dps_dir.joinpath("min.B").write_text("4\n", encoding="utf-8")

        markov_dir = self.dps_dir / "markov_T300K"
        markov_dir.mkdir()
        B = sp.csr_matrix(
            [
                [0.0, 0.5, 0.0, 0.0],
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0, 1.0],
                [0.0, 0.0, 0.5, 0.0],
            ]
        )
        sp.save_npz(markov_dir / "B_T300K.npz", B)
        np.save(markov_dir / "tau_T300K.npy", np.array([1.0, 0.5, 0.5, 1.0]))
        np.save(markov_dir / "original_min_ids_T300K.npy", np.arange(1, 5))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_keep_set_builds_generator_without_pygt(self):
        Q_eff, kept_order, requested = build_Qeff_for_deltaE(
            self.dps_dir,
            deltaE_cut=-1.0,
            E_window=0.0,
            temperature=300.0,
        )

        np.testing.assert_array_equal(kept_order, [0, 1, 2, 3])
        np.testing.assert_array_equal(requested, [0, 1, 2, 3])
        np.testing.assert_allclose(np.asarray(Q_eff.sum(axis=0)).ravel(), 0.0)

    @unittest.skipUnless(importlib.util.find_spec("PyGT"), "PyGT is not installed")
    def test_pygt_reduction_uses_original_minimum_ids(self):
        Q_eff, kept_order, requested = build_Qeff_for_deltaE(
            self.dps_dir,
            deltaE_cut=0.75,
            E_window=0.0,
            temperature=300.0,
            block=2,
        )

        np.testing.assert_array_equal(kept_order, [0, 2, 3])
        np.testing.assert_array_equal(requested, [0, 2, 3])
        self.assertEqual(Q_eff.shape, (3, 3))
        np.testing.assert_allclose(
            np.asarray(Q_eff.sum(axis=0)).ravel(),
            0.0,
            atol=1e-12,
        )
        self.assertGreater(compute_mfpt_from_Q(Q_eff, [0], [2]), 0.0)

    @unittest.skipUnless(importlib.util.find_spec("PyGT"), "PyGT is not installed")
    def test_quantitative_cli_writes_complete_csv(self):
        argv = [
            "quantitative_keeplist_checks.py",
            "--data-dir",
            str(self.dps_dir),
            "--deltaE-grid",
            "0.75",
            "--E-window",
            "0",
            "--n-relax",
            "2",
        ]
        with patch.object(sys, "argv", argv), patch("sys.stdout", new=io.StringIO()):
            quantitative_main()

        output = self.dps_dir / "robustness_vs_deltaE.csv"
        lines = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("N_requested", lines[0])
        self.assertIn("N_dropped_outside_micro", lines[0])
        self.assertIn("MFPT_A_to_B", lines[0])


class BuildRunnerTests(unittest.TestCase):
    def test_complete_model_requires_analysis_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dps_dir = Path(tmp)
            paths = MarkovFilePaths(dps_dir, 300.0)
            paths.markov_dir.mkdir()
            required = (
                paths.B_path,
                paths.K_path,
                paths.Q_path,
                paths.tau_path,
                paths.pi_path,
                paths.energies_path,
                paths.entropies_path,
                paths.retained_mask_path,
                paths.orig_ids_path,
            )
            for required_path in required:
                required_path.touch()

            self.assertTrue(model_is_complete(dps_dir, 300.0))
            paths.orig_ids_path.unlink()
            self.assertFalse(model_is_complete(dps_dir, 300.0))


if __name__ == "__main__":
    unittest.main()
