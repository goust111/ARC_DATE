from __future__ import annotations

from pathlib import Path
import random
import sys
from typing import Iterable, Tuple

import numpy as np

# Ensure `arc_utils` can be imported when this file is loaded by SourceFileLoader.
_UTIL_DIR: str | None = None
_FILE_DIR = Path(__file__).resolve().parent
for _candidate in (_FILE_DIR, *_FILE_DIR.parents):
    candidate_path = _candidate / "new" / "solvers" / "arc_utils.py"
    if candidate_path.exists():
        _UTIL_DIR = str(_candidate / "new" / "solvers")
        break
if _UTIL_DIR is None:
    raise RuntimeError("Cannot locate new/solvers/arc_utils.py for this task")
if _UTIL_DIR not in sys.path:
    sys.path.append(_UTIL_DIR)

from arc_utils import Color  # noqa: E402

BG = int(Color.BLACK)
RED = int(Color.RED)
YELLOW = int(Color.YELLOW)
GRAY = int(Color.GRAY)


def _periodic_mask(h: int, w: int, period_r: int, period_c: int, offset_r: int, offset_c: int) -> np.ndarray:
    rows = np.arange(h)
    cols = np.arange(w)
    row_mask = ((rows - offset_r) % period_r == 0)[:, None]
    col_mask = ((cols - offset_c) % period_c == 0)[None, :]
    return row_mask & col_mask


def _infer_full_obstacles(grid: np.ndarray, gray_coords: np.ndarray) -> Tuple[int, int, int, int, np.ndarray]:
    h, w = grid.shape
    rows = gray_coords[:, 0]
    cols = gray_coords[:, 1]
    # Outside the veil (yellow) and outside visible red/gray pixels, no hidden obstacle
    # is allowed, otherwise these cells would have been visible.
    forbidden = (grid != YELLOW) & (grid != GRAY) & (grid != RED)

    row_idxs = np.arange(h)
    col_idxs = np.arange(w)

    best: tuple[int, int, int, int, np.ndarray] | None = None
    best_area = h * w + 1

    for period_r in range(1, h + 1):
        anchor_r = int(rows[0] % period_r)
        if not np.all(rows % period_r == anchor_r):
            continue
        row_mask = (row_idxs - anchor_r) % period_r == 0

        for period_c in range(1, w + 1):
            anchor_c = int(cols[0] % period_c)
            if not np.all(cols % period_c == anchor_c):
                continue

            col_mask = (col_idxs - anchor_c) % period_c == 0
            obstacle = row_mask[:, None] & col_mask[None, :]

            if np.any(obstacle & forbidden):
                continue

            area = period_r * period_c
            if area < best_area or (
                area == best_area
                and best is not None
                and (period_r < best[0] or (period_r == best[0] and period_c < best[1]))
            ):
                best = (period_r, period_c, anchor_r, anchor_c, obstacle)
                best_area = area

    if best is None:
        raise AssertionError("visible gray pixels do not uniquely determine a valid periodic lattice")
    return best


def _drop_markers(grid: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.array(grid, copy=True)

    marker_mask = out == RED
    if not np.any(marker_mask):
        return out

    marker_cols = np.flatnonzero(np.any(marker_mask, axis=0))
    assert np.all(np.bincount(marker_cols, minlength=w) <= 1), "each marker column must contain at most one marker"

    out[marker_mask] = BG

    for c in marker_cols.tolist():
        c = int(c)
        obs_rows = np.flatnonzero(obstacle[:, c])
        if obs_rows.size:
            dest_r = int(obs_rows[0] - 1)
            assert dest_r >= 0, "marker must be above all obstacle cells in its column"
        else:
            dest_r = h - 1

        out[dest_r, c] = RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Infer periodic hidden obstacles under yellow veil, then drop each top marker to the first obstacle stop point.

    Input structure:
    - Black background, gray obstacle pixels form a 2D-regular lattice, with yellow rectangular veil cells hiding some obstacles.
    - Some top-row red marker pixels, at most one per column, each lying above any gray pixels in that column.
    - No other colors carry structural meaning.

    Rule:
    - Infer the smallest-period rectangular lattice of obstacle points that includes all visible gray cells and has no gray/marker
      outside veil conflicts with non-veiled cells.
    - Each marker falls downward in its column, stopping on the row above the first obstacle in that inferred lattice; if that
      column has no obstacle, it falls to the bottom row.
    - The drop destination is written red, all original markers are cleared, and every existing yellow, visible gray, and black
      pixel remains otherwise unchanged.

    Key insight:
    - The veil hides obstacles, so only cells outside yellow that are not gray/marker are guaranteed empty; this constraint makes
      the full lattice period inferable and the stop row unambiguous.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    unique_colors = set(map(int, np.unique(g)))
    assert unique_colors <= {BG, RED, YELLOW, GRAY}, "unexpected colors in input"

    gray_coords = np.argwhere(g == GRAY)
    assert gray_coords.size >= 1, "expected at least one visible gray pixel"

    _, _, _, _, obstacle = _infer_full_obstacles(g, gray_coords)
    return _drop_markers(g, obstacle)


def _simulate_output(inp: np.ndarray, full_obstacle: np.ndarray) -> np.ndarray:
    return _drop_markers(inp, full_obstacle)


def _build_one_candidate() -> np.ndarray | None:
    H = random.randint(14, 28)
    W = random.randint(14, 28)

    period_r = random.randint(3, 6)
    period_c = random.randint(2, 6)
    base_r = random.randint(2, period_r - 1)
    base_c = random.randint(0, period_c - 1)

    full_obstacle = _periodic_mask(H, W, period_r, period_c, base_r, base_c)
    obstacle_cols = np.flatnonzero(full_obstacle.any(axis=0))
    empty_cols = np.flatnonzero(~full_obstacle.any(axis=0))
    if obstacle_cols.size == 0 or empty_cols.size == 0:
        return None

    # Place red markers: mostly on obstacle columns, and occasionally in obstacle-free columns.
    obstacle_marker_cols = sorted(np.random.choice(obstacle_cols, size=random.randint(1, min(5, obstacle_cols.size)), replace=False).tolist())
    if random.random() < 0.75 and empty_cols.size:
        picked = random.sample(range(empty_cols.size), random.randint(1, min(3, empty_cols.size)))
        empty_marker_cols = sorted(empty_cols[np.array(picked, dtype=int)].tolist())
    else:
        empty_marker_cols = []

    grid = np.full((H, W), BG, dtype=int)
    grid[full_obstacle] = GRAY

    # Hide a subset of gray pixels with veil rectangles; keep top area mostly visible so period inference has stable anchors.
    veil = np.zeros((H, W), dtype=bool)
    top_safe_row = min(H - 4, base_r + 2 * period_r)
    top_safe_row = max(top_safe_row, 2)

    for _ in range(random.randint(1, 3)):
        if H - top_safe_row <= 0:
            continue
        for _ in range(60):
            h = random.randint(1, min(6, H - top_safe_row))
            w = random.randint(2, min(8, W))
            r0 = random.randint(top_safe_row, H - h)
            c0 = random.randint(0, W - w)
            veil[r0 : r0 + h, c0 : c0 + w] = True
            break

    # Add a forced hidden point in case random rectangles miss everything.
    # Keep this target near a lattice obstacle so hidden-veil branch is visible in generation.
    if np.sum(full_obstacle & veil) == 0:
        rs, cs = np.where(full_obstacle)
        idx = random.randrange(len(rs))
        rr = int(rs[idx])
        if rr + 1 < H:
            cc = int(cs[idx])
            veil[rr + 1, cc] = True

    grid[veil] = YELLOW

    # Guarantee some visible reference obstacles for a stable period guess.
    if np.sum(full_obstacle & (grid == GRAY)) < 4:
        return None

    # Force one landing point to fall through yellow sometimes, and keep some black-landings too.
    if random.random() < 0.85:
        candidates: list[tuple[int, int]] = []
        for c in obstacle_marker_cols:
            first = np.flatnonzero(full_obstacle[:, c])[0]
            landing = int(first - 1)
            if landing >= 0 and grid[landing, c] == BG:
                candidates.append((c, landing))
        if candidates:
            c, landing = random.choice(candidates)
            grid[landing, c] = YELLOW

    marker_cols = sorted(set(obstacle_marker_cols) | set(empty_marker_cols))
    if not marker_cols:
        return None

    inp = np.array(grid, copy=True)
    for c in marker_cols:
        inp[0, c] = RED

    # Keep the generator aligned with the solver assumptions.
    expected = _simulate_output(inp, full_obstacle)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    # Branch coverage guard: include columns with and without obstacles in at least one attempt.
    if not any(full_obstacle[:, c].any() for c in marker_cols):
        return None

    # Keep a chance that there is at least one obstacle-free marker column in the same sample.
    if random.random() < 0.55 and not any(not full_obstacle[:, c].any() for c in marker_cols):
        return None

    if random.random() < 0.55 and not any(full_obstacle[:, c].any() and inp[(np.flatnonzero(full_obstacle[:, c])[0] - 1), c] == YELLOW for c in marker_cols if np.flatnonzero(full_obstacle[:, c]).size):
        return None

    return inp


def generate_input() -> np.ndarray:
    """Generate random ARC-style inputs that satisfy hidden periodic lattice + yellow veil + top markers."""
    for _ in range(250):
        inp = _build_one_candidate()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for task 016/1")
