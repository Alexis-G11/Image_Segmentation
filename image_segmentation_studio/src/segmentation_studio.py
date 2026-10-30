"""ImageSegmentationStudio - unified entry point (Member 4).

Combines K-means, Otsu thresholding and watershed behind one class:
  - runs every method on an image and times it
  - side-by-side matplotlib visualisation (all outputs labelled)
  - performance comparison table (console + CSV + Markdown)
  - batch processing of a folder
  - uses utils.mock_* automatically if a teammate's module is missing
    or not finished yet, so the pipeline always runs.
"""
import csv
import importlib
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:  # works as package (src.xxx) or as plain scripts
    from . import utils
except ImportError:
    import utils


# (display name, module, function names to try in order, mock fallback)
METHOD_SPECS = [
    ("K-means (K=3)", "kmeans_segmentation",
     ["segment_kmeans", "kmeans_segmentation", "segment"], utils.mock_kmeans),
    ("Otsu threshold", "thresholding_segmentation",
     ["segment_otsu", "segment_threshold", "otsu_threshold", "segment"],
     utils.mock_threshold),
    ("Watershed", "watershed_segmentation",
     ["segment_watershed", "watershed_segmentation", "segment"],
     utils.mock_watershed),
]


def _load_method(module_name, func_names, mock):
    """Return (callable, is_mock). Falls back to the mock if import fails."""
    pkg = __package__ or ""
    candidates = [f"{pkg}.{module_name}"] if pkg else []
    candidates.append(module_name)
    for name in candidates:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for fn in func_names:
            if hasattr(mod, fn):
                return getattr(mod, fn), False
    return mock, True


class ImageSegmentationStudio:
    """Run, time, display and compare all segmentation methods."""

    def __init__(self, output_dir="results", max_side=800, repeats=3):
        self.output_dir = utils.ensure_dir(output_dir)
        self.max_side = max_side      # downscale big photos for speed
        self.repeats = repeats        # timing = mean over N runs
        self.methods = {}             # name -> (callable, is_mock)
        for name, module, funcs, mock in METHOD_SPECS:
            self.methods[name] = _load_method(module, funcs, mock)

    # ------------------------------------------------------------ status
    def status(self):
        return {n: ("MOCK" if m else "REAL") for n, (_, m) in self.methods.items()}

    def register(self, name, func):
        """Add or replace a method at runtime (e.g. a Gaussian-blur Otsu)."""
        self.methods[name] = (func, False)

    # --------------------------------------------------------------- run
    def segment(self, image):
        """Run all methods on one BGR image -> dict name -> result dict."""
        results = {}
        for name, (func, is_mock) in self.methods.items():
            try:
                raw, secs = utils.time_call(func, image.copy(),
                                            repeats=self.repeats)
                out = utils.to_bgr(raw)
                results[name] = {"image": out, "seconds": secs,
                                 "regions": utils.count_regions(raw),
                                 "mock": is_mock, "error": None}
            except Exception as exc:  # one failing method must not stop the studio
                results[name] = {"image": np.zeros_like(image), "seconds": float("nan"),
                                 "regions": 0, "mock": is_mock, "error": str(exc)}
        return results

    def process_file(self, path, save=True, show=False):
        """Load, segment, visualise and save outputs for one image."""
        image = utils.load_image(path, self.max_side)
        results = self.segment(image)
        stem = os.path.splitext(os.path.basename(path))[0]
        if save:
            for name, r in results.items():
                tag = name.split()[0].lower().replace("-", "")
                utils.save_image(os.path.join(self.output_dir, stem, f"{stem}_{tag}.png"),
                                 r["image"])
        fig_path = os.path.join(self.output_dir, stem, f"{stem}_comparison.png")
        self.visualize(image, results, title=stem,
                       save_path=fig_path if save else None, show=show)
        return image, results

    def process_folder(self, folder, save=True):
        """Process every image of a folder; returns {filename: results}."""
        paths = utils.list_images(folder)
        if not paths:
            raise FileNotFoundError(f"No images found in {folder}")
        all_results = {}
        for p in paths:
            _, res = self.process_file(p, save=save)
            all_results[os.path.basename(p)] = res
        if save:
            self.save_table(all_results)
        return all_results

    # ----------------------------------------------------- visualisation
    def visualize(self, image, results, title="", save_path=None, show=True):
        """Original | method 1 | method 2 | method 3, all labelled."""
        if not show:
            matplotlib.use("Agg", force=True)
        import cv2
        n = len(results) + 1
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original", fontsize=11)
        for ax, (name, r) in zip(axes[1:], results.items()):
            ax.imshow(cv2.cvtColor(r["image"], cv2.COLOR_BGR2RGB))
            label = f"{name}\n{r['seconds'] * 1000:.1f} ms | {r['regions']} regions"
            if r["mock"]:
                label += "\n[MOCK result]"
            if r["error"]:
                label += f"\nERROR: {r['error'][:40]}"
            ax.set_title(label, fontsize=10)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Image Segmentation Studio - {title}", fontsize=13, y=1.0)
        fig.tight_layout()
        if save_path:
            utils.ensure_dir(os.path.dirname(save_path))
            fig.savefig(save_path, dpi=130, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------ performance tables
    @staticmethod
    def _rows(all_results):
        rows = []
        for fname, res in all_results.items():
            for name, r in res.items():
                rows.append([fname, name, f"{r['seconds'] * 1000:.2f}",
                             r["regions"], "mock" if r["mock"] else "real"])
        return rows

    def table_text(self, all_results):
        """Markdown table ready to paste into the report."""
        head = ["Image", "Method", "Time (ms)", "Regions", "Source"]
        lines = ["| " + " | ".join(head) + " |",
                 "|" + "|".join(["---"] * len(head)) + "|"]
        lines += ["| " + " | ".join(str(c) for c in row) + " |"
                  for row in self._rows(all_results)]
        # average per method
        lines += ["", "**Average time per method**", "",
                  "| Method | Mean time (ms) |", "|---|---|"]
        per = {}
        for res in all_results.values():
            for name, r in res.items():
                per.setdefault(name, []).append(r["seconds"] * 1000)
        for name, vals in per.items():
            lines.append(f"| {name} | {np.nanmean(vals):.2f} |")
        return "\n".join(lines)

    def save_table(self, all_results, stem="performance"):
        """Write performance.csv and performance.md in the output folder."""
        csv_path = os.path.join(self.output_dir, f"{stem}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "method", "time_ms", "regions", "source"])
            w.writerows(self._rows(all_results))
        md_path = os.path.join(self.output_dir, f"{stem}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.table_text(all_results))
        return csv_path, md_path