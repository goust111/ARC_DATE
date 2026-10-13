"""Solve legend/hole-conservation rows by restoring missing dots and recoloring keyed gray objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import binary_fill_holes, label

# Ensure arc_utils can be imported when this file is loaded directly.
UTILS_DIR = Path(__file__).resolve().parents[6] / "new" / "solvers"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, find_components, get_bg, paste, random_free_location, STRUCT4

MAX_HOLES = 4
ROW_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


@dataclass(frozen=True)
class _RowSpec:
    color: int
    count: int
    has_points: bool


def _count_holes(mask: np.ndarray) -> int:
    if not bool(mask.any()):
        return 0
    filled = binary_fill_holes(mask)
    holes = filled & (~mask)
    _, n = label(holes, structure=STRUCT4)
    return int(n)


def _holes_from_coords(coords: np.ndarray) -> int:
    if coords.size == 0:
        return 0
    r0, c0, r1, c1 = bbox_coords(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return _count_holes(local)


def _find_legend_bbox(grid: np.ndarray, bg: int, frame_color: int = Color.BLACK) -> tuple[int, int, int, int]:
    components = [
        comp
        for comp in find_components(grid, bg=bg, colors=[frame_color], connectivity=8, monochromatic=True)
    ]
    candidates: list[tuple[int, int, int, int]] = []

    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        coords = comp["coords"]
        inside = (coords[:, 0] > r0) & (coords[:, 0] < r1) & (coords[:, 1] > c0) & (coords[:, 1] < c1)
        if bool(np.any(inside)):
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(patch[0, :] == frame_color)
            and np.all(patch[-1, :] == frame_color)
            and np.all(patch[:, 0] == frame_color)
            and np.all(patch[:, -1] == frame_color)
        ):
            continue

        inner = patch[1:-1, 1:-1]
        if inner.size == 0 or not np.any(inner != bg):
            continue

        candidates.append((r0, c0, r1, c1))

    assert candidates, "expected one black legend frame"
    return max(candidates, key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1))


def _extract_legend_rows(grid: np.ndarray, legend_bbox: tuple[int, int, int, int], bg: int) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = legend_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]

    rows: list[tuple[int, int, int, int]] = []
    for dr, row in enumerate(inner):
        nz = np.where(row != bg)[0]
        if nz.size == 0:
            continue
        key_c = int(nz.min())
        key_color = int(row[key_c])
        if key_color in (Color.BLACK, Color.GREY):
            continue

        point_count = int(np.sum(row[key_c + 1 :] == key_color))
        rows.append((r0 + 1 + dr, c0 + 1 + key_c, key_color, point_count))

    assert rows, "expected at least one key row"
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """Map row dots and outer holes through a bidirectional conservation law.

    Input structure:
    - A black legend frame, with key rows inside; each row has one key-color pixel at its
      leftmost non-background cell.
    - Some key rows already contain spaced dots of that key color.
    - Outside the legend, gray objects with holes and key-color objects with holes may appear.

    Rule:
    - For a row with point count n, recolor the entire gray object outside the legend that has
      exactly n enclosed 4-connected holes to the row's key color.
    - For a row with no points, count holes in the key color object outside the legend, then write
      that many points to the right of the key pixel with one-cell spacing.
    - Preserve the legend frame and all other background/no-hole objects.

    Key insight:
    - The invariant is point-count == hole-count; rows and outer objects complete each other
      in both directions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg != Color.BLACK, "expected background != 0 to separate legend frame"

    out = g.copy()
    legend = _find_legend_bbox(g, bg, frame_color=Color.BLACK)
    r0, c0, r1, c1 = legend
    rows = _extract_legend_rows(g, legend, bg)

    outside = np.ones_like(g, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False

    gray_components = [
        (idx, coords, _holes_from_coords(coords))
        for idx, coords in enumerate(components_mask((g == Color.GREY) & outside, connectivity=4))
        if coords.size > 0
    ]
    used_gray = set[int]()

    for row, key_c, key_color, point_count in rows:
        if point_count > 0:
            candidates = [
                (idx, coords)
                for idx, coords, hole_count in gray_components
                if idx not in used_gray and hole_count == point_count
            ]
            assert candidates, "no gray component matches point count"
            idx, coords = candidates[0]
            out[coords[:, 0], coords[:, 1]] = key_color
            used_gray.add(idx)
            continue

        key_objects = [
            coords for coords in components_mask((g == key_color) & outside, connectivity=4) if coords.size > 0
        ]
        assert len(key_objects) == 1, "expected one key-color object for each no-point row"
        key_holes = _holes_from_coords(key_objects[0])
        assert key_holes >= 0, "hole count must be defined"

        for t in range(key_holes):
            cc = int(key_c + 2 * (t + 1))
            assert cc < c1, "legend row is too narrow for generated points"
            out[row, cc] = key_color

    assert not np.array_equal(out, g), "main should produce a change"
    return out


def _make_holey_mask(height: int, width: int, hole_count: int) -> np.ndarray | None:
    if hole_count < 0:
        return None
    if hole_count == 0:
        return np.ones((height, width), dtype=bool)
    if height < 3 or width < 3:
        return None
    if hole_count > (height - 2) * (width - 2):
        return None

    candidates = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]

    for _ in range(300):
        random.shuffle(candidates)
        holes: list[tuple[int, int]] = []
        for r, c in candidates:
            if len(holes) == hole_count:
                break
            if any(abs(r - hr) + abs(c - hc) == 1 for hr, hc in holes):
                continue
            holes.append((r, c))

        if len(holes) != hole_count:
            continue

        mask = np.ones((height, width), dtype=bool)
        for r, c in holes:
            mask[r, c] = False

        _, comp_cnt = label(mask, structure=STRUCT4)
        if comp_cnt != 1:
            continue
        if _count_holes(mask) != hole_count:
            continue

        return mask

    return None


def _make_object_patch(color: int, bg: int, hole_count: int) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(4, 9)
        w = random.randint(4, 9)
        if hole_count > (h - 2) * (w - 2):
            continue

        mask = _make_holey_mask(h, w, hole_count)
        if mask is None:
            continue

        patch = np.full((h, w), bg, dtype=int)
        patch[mask] = color
        return patch
    return None


def _place_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    bg: int,
    forbidden: np.ndarray,
) -> tuple[int, int]:
    working = canvas.copy()
    block_color = 10 if bg != 10 else 11
    working[forbidden] = block_color

    r, c = random_free_location(working, patch, transparent=bg, padding=1, padding_connectivity=4)
    paste(canvas, patch, r, c, transparent=bg)
    return r, c


def _sample_specs() -> list[_RowSpec]:
    n_rows = random.randint(2, 4)
    key_colors = random.sample([c for c in ROW_COLOR_POOL if c != Color.GREY], k=n_rows)

    branches = [True, False]
    if n_rows > 2:
        for _ in range(n_rows - 2):
            branches.append(random.choice([True, False]))
    random.shuffle(branches)

    point_counts = random.sample(range(1, MAX_HOLES + 1), k=branches.count(True))
    p_iter = iter(point_counts)

    specs: list[_RowSpec] = []
    for row_idx in range(n_rows):
        if branches[row_idx]:
            specs.append(_RowSpec(color=key_colors[row_idx], count=next(p_iter), has_points=True))
        else:
            specs.append(_RowSpec(color=key_colors[row_idx], count=random.randint(1, MAX_HOLES), has_points=False))

    return specs


def _render_legend(specs: list[_RowSpec], bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    key_rows = len(specs)
    max_count = max(spec.count for spec in specs)
    inner_w = max(4, 2 * max_count + 2)
    inner_h = key_rows
    patch = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
    patch[0, :] = Color.BLACK
    patch[-1, :] = Color.BLACK
    patch[:, 0] = Color.BLACK
    patch[:, -1] = Color.BLACK

    for idx, spec in enumerate(specs):
        r = 1 + idx
        key_c = 1
        patch[r, key_c] = spec.color
        if spec.has_points:
            for t in range(spec.count):
                cc = key_c + 2 * (t + 1)
                if cc < patch.shape[1] - 1:
                    patch[r, cc] = spec.color

    legend_bbox = (0, 0, patch.shape[0] - 1, patch.shape[1] - 1)
    return patch, legend_bbox


def _is_clean_result(grid: np.ndarray, specs: list[_RowSpec]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False

    bg = get_bg(grid)
    legend = _find_legend_bbox(grid, bg, frame_color=Color.BLACK)
    rows = _extract_legend_rows(grid, legend, bg)
    if len(rows) != len(specs):
        return False

    return True


def generate_input() -> np.ndarray:
    """Generate randomized valid inputs with both branch patterns represented."""
    for _ in range(400):
        bg = random.choice([c for c in ROW_COLOR_POOL if c != Color.GREY])
        specs = _sample_specs()

        legend, _ = _render_legend(specs, bg)
        H, W = legend.shape
        if H + 6 > 30 or W + 8 > 30:
            continue

        min_h = max(14, legend.shape[0] + 6)
        min_w = max(14, legend.shape[1] + 10)
        H = random.randint(min_h, 30)
        W = random.randint(min_w, 30)

        grid = np.full((H, W), fill_value=bg, dtype=int)

        # Ensure at least one-cell margin around the legend.
        lr = random.randint(1, H - legend.shape[0] - 1)
        lc = random.randint(1, W - legend.shape[1] - 1)
        paste(grid, legend, lr, lc)

        forbidden = np.zeros_like(grid, dtype=bool)
        forbidden[lr : lr + legend.shape[0], lc : lc + legend.shape[1]] = True

        patch_specs = list(specs)
        random.shuffle(patch_specs)
        ok = True

        for spec in patch_specs:
            patch = _make_object_patch(spec.color if not spec.has_points else Color.GREY, bg, spec.count)
            if patch is None:
                ok = False
                break
            try:
                _place_patch(grid, patch, bg, forbidden)
            except Exception:
                ok = False
                break

        if not ok:
            continue

        # Add gray distractors with no holes (optional).
        if random.random() < 0.8:
            for _ in range(random.randint(0, 2)):
                h = random.randint(2, 4)
                w = random.randint(2, 4)
                patch = np.full((h, w), Color.GREY, dtype=int)
                if h > 1 and w > 1:
                    patch[0, 0] = bg
                try:
                    _place_patch(grid, patch, bg, forbidden)
                except Exception:
                    pass

        if not _is_clean_result(grid, specs):
            continue

        return grid

    raise RuntimeError("failed to generate a valid 7.py input")
