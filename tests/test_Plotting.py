import unittest
import warnings

import pytest

# matplotlib is an optional dependency of moments, and is not installed in every
# test environment (e.g. cibuildwheel's wheel tests), so skip if it's missing.
matplotlib = pytest.importorskip("matplotlib")

# Use a non-interactive backend so tests run headless and never block on show().
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import moments


class Plot2DSFSTestCase(unittest.TestCase):
    """
    Tests for moments.Plotting.plot_single_2d_sfs. Regression tests for issue #94,
    where plotting modified the globally registered "hsv" colormap in place.
    """

    def setUp(self):
        self.fs = moments.Demographics2D.snm([10, 10])
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def tearDown(self):
        plt.close(self.fig)

    def _plotted_cmap(self):
        # The pcolor mesh is the only collection drawn onto the axes.
        return self.ax.collections[0].get_cmap()

    def test_no_deprecation_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            moments.Plotting.plot_single_2d_sfs(self.fs, ax=self.ax, show=False)

    def test_global_colormap_not_modified(self):
        hsv = matplotlib.colormaps["hsv"]
        under_before = hsv.get_under()
        moments.Plotting.plot_single_2d_sfs(self.fs, ax=self.ax, show=False)
        self.assertTrue(np.array_equal(hsv.get_under(), under_before))
        self.assertTrue(np.array_equal(plt.cm.hsv.get_under(), plt.cm.hsv(0.0)))

    def test_user_colormap_not_modified(self):
        cmap = matplotlib.colormaps["viridis"]
        under_before = cmap.get_under()
        moments.Plotting.plot_single_2d_sfs(self.fs, ax=self.ax, cmap=cmap, show=False)
        self.assertTrue(np.array_equal(cmap.get_under(), under_before))
        self.assertEqual(self._plotted_cmap().name, "viridis")

    def test_under_color_is_white(self):
        moments.Plotting.plot_single_2d_sfs(self.fs, ax=self.ax, show=False)
        self.assertTrue(
            np.array_equal(self._plotted_cmap().get_under(), (1.0, 1.0, 1.0, 1.0))
        )

    def test_colormap_by_name(self):
        moments.Plotting.plot_single_2d_sfs(
            self.fs, ax=self.ax, cmap="magma", show=False
        )
        cmap = self._plotted_cmap()
        self.assertEqual(cmap.name, "magma")
        self.assertTrue(np.array_equal(cmap.get_under(), (1.0, 1.0, 1.0, 1.0)))

    def test_2d_comparison_plot(self):
        # The comparison plots call plot_single_2d_sfs twice for the model and data
        # panels, and should also leave the global colormap untouched.
        model = self.fs
        data = moments.Spectrum(
            np.random.default_rng(0).poisson(1000 * model.data), pop_ids=["A", "B"]
        )
        plt.close(self.fig)
        moments.Plotting.plot_2d_comp_Poisson(model, data, show=False)
        self.assertTrue(np.array_equal(plt.cm.hsv.get_under(), plt.cm.hsv(0.0)))
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
