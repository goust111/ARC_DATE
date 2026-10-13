from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import crop_bbox
except Exception:  # pragma: no cover - only used outside the local ARC workspace
    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, box)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


BBox = tuple[int, int, int, int]


def _is_canvas_boundary(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _is_corner(r: int, c: int, h: int, w: int) -> bool:
    return (r in (0, h - 1)) and (c in (0, w - 1))


def _find_generator_color_and_box(g: np.ndarray) -> tuple[int, BBox]:
    h, w = map(int, g.shape)
    candidates: list[tuple[int, BBox]] = []

    for raw_color in np.unique(g):
        color = int(raw_color)
        coords = np.argwhere(g == color)
        if int(coords.shape[0]) != 4:
            continue

        pts = [(int(r), int(c)) for r, c in coords]
        if any(not _is_canvas_boundary(r, c, h, w) for r, c in pts):
            continue
        if any(_is_corner(r, c, h, w) for r, c in pts):
            continue

        horizontal = [(r, c) for r, c in pts if r in (0, h - 1)]
        vertical = [(r, c) for r, c in pts if c in (0, w - 1)]
        if len(horizontal) != 2 or len(vertical) != 2:
            continue

        rows = sorted({r for r, _c in vertical})
        cols = sorted({c for _r, c in horizontal})
        if len(rows) != 2 or len(cols) != 2:
            continue
        if rows[0] >= rows[1] or cols[0] >= cols[1]:
            continue

        candidates.append((color, (rows[0], cols[0], rows[1], cols[1])))

    assert len(candidates) == 1, "expected exactly one boundary-only four-pixel generator color"
    return candidates[0]


def _inside_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _find_control_color(g: np.ndarray, generator_color: int, box: BBox) -> int:
    inside = _inside_mask(g.shape, box)
    candidates: list[int] = []

    for raw_color in np.unique(g):
        color = int(raw_color)
        if color == int(generator_color):
            continue
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        if bool(np.all(inside[coords[:, 0], coords[:, 1]])):
            candidates.append(color)

    assert len(candidates) == 1, "expected exactly one window-private control color"
    return candidates[0]


def _unique_majority_color(crop: np.ndarray) -> int:
    counts = Counter(map(int, crop.ravel()))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty crop"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique window majority color"
    return int(ranked[0][0])


def _expanded_output(crop_shape: tuple[int, int], base: int, control: int, markers: np.ndarray) -> np.ndarray:
    out = np.full(crop_shape, int(base), dtype=int)
    rows = sorted({int(r) for r, _c in markers})
    cols = sorted({int(c) for _r, c in markers})
    out[rows, :] = int(control)
    out[:, cols] = int(control)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the implicit boundary-defined window and expand its private marker pixels into full rows and columns.

    Input structure:
    - A noisy canvas contains one frame-generator color that appears exactly four times, all on the canvas boundary.
    - Two generator pixels on the top or bottom boundary give the window columns, and two on the left or right boundary give the window rows.
    - Exactly one control color appears only inside that implicit window; every other non-generator color appears both inside and outside it.

    Rule:
    - Crop to the inclusive window defined by the two generator rows and two generator columns.
    - Use the unique most common color inside that crop as the output base color.
    - For every control-color pixel in the crop, paint its entire output row and column with the control color.

    Key insight:
    - The rectangle is never drawn directly; the boundary-only generator color defines the crop, and the window-only color defines row and column controls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "expected an ARC-sized grid"

    generator_color, box = _find_generator_color_and_box(g)
    control_color = _find_control_color(g, generator_color, box)
    crop = crop_bbox(g, box)
    base_color = _unique_majority_color(crop)
    assert base_color != control_color, "control color must be sparse, not the window base"

    marker_coords = np.argwhere(crop == control_color)
    assert marker_coords.size > 0, "expected at least one control marker"
    return _expanded_output(crop.shape, base_color, control_color, marker_coords)


# --------------- generator (implicit boundary window) ---------------

_GEN_COLORS = np.arange(10, dtype=int)
_GEN_RETRIES = 500
_GEN_GRID_MIN, _GEN_GRID_MAX = 13, 30
_GEN_WINDOW_MIN, _GEN_WINDOW_MAX = 5, 13
_GEN_DISTRACTOR_MIN, _GEN_DISTRACTOR_MAX = 2, 4
_GEN_MARKER_MIN, _GEN_MARKER_MAX = 2, 4
_GEN_INSIDE_NOISE_FRAC = (0.16, 0.30)
_GEN_OUTSIDE_NOISE_FRAC = (0.10, 0.22)


def _choice_int(rng: np.random.Generator, values) -> int:
    return int(values[int(rng.integers(0, len(values)))])


def _shuffled_cells(rng: np.random.Generator, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    order = rng.permutation(len(cells))
    return [cells[int(i)] for i in order]


def _sample_marker_coords(rng: np.random.Generator, h: int, w: int) -> list[tuple[int, int]] | None:
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if len(interior) < _GEN_MARKER_MIN:
        return None

    max_markers = min(_GEN_MARKER_MAX, len(interior))
    for _ in range(80):
        n_markers = int(rng.integers(_GEN_MARKER_MIN, max_markers + 1))
        cells = _shuffled_cells(rng, interior)[:n_markers]
        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue

        line_area = len(rows) * w + len(cols) * h - len(rows) * len(cols)
        if line_area >= h * w:
            continue
        if line_area > int(0.68 * h * w):
            continue
        return cells
    return None


def _paint_crop(
    rng: np.random.Generator,
    shape: tuple[int, int],
    base_color: int,
    control_color: int,
    distractors: list[int],
    marker_cells: list[tuple[int, int]],
) -> np.ndarray | None:
    h, w = map(int, shape)
    marker_set = set(marker_cells)
    available = [(r, c) for r in range(h) for c in range(w) if (r, c) not in marker_set]
    min_noise = len(distractors)
    max_noise = min(len(available), max(min_noise, int(round(h * w * _GEN_INSIDE_NOISE_FRAC[1]))))
    if max_noise < min_noise:
        return None

    noise_count = int(rng.integers(min_noise, max_noise + 1))
    crop = np.full(shape, int(base_color), dtype=int)
    noise_cells = _shuffled_cells(rng, available)[:noise_count]
    shuffled_distractors = [int(c) for c in rng.permutation(distractors)]

    for (r, c), color in zip(noise_cells[: len(shuffled_distractors)], shuffled_distractors):
        crop[r, c] = int(color)
    for r, c in noise_cells[len(shuffled_distractors) :]:
        crop[r, c] = _choice_int(rng, distractors)
    for r, c in marker_cells:
        crop[r, c] = int(control_color)

    counts = Counter(map(int, crop.ravel()))
    if counts[int(base_color)] <= max(v for k, v in counts.items() if k != int(base_color)):
        return None
    return crop


def _outside_cells(shape: tuple[int, int], box: BBox, reserved: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, box)
    return [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not (r0 <= r <= r1 and c0 <= c <= c1) and (r, c) not in reserved
    ]


def _build_expected(shape: tuple[int, int], base: int, control: int, markers: list[tuple[int, int]]) -> np.ndarray:
    marker_arr = np.array(markers, dtype=int)
    return _expanded_output(shape, int(base), int(control), marker_arr)


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    generator_color: int,
    control_color: int,
    box: BBox,
    base_color: int,
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    try:
        found_generator, found_box = _find_generator_color_and_box(grid)
        found_control = _find_control_color(grid, found_generator, found_box)
        crop = crop_bbox(grid, found_box)
        found_base = _unique_majority_color(crop)
        actual = main(grid)
    except Exception:
        return False

    if found_generator != int(generator_color) or found_control != int(control_color):
        return False
    if found_box != tuple(map(int, box)) or found_base != int(base_color):
        return False
    if not np.array_equal(actual, expected):
        return False

    inside = _inside_mask(grid.shape, box)
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color in (int(generator_color), int(control_color)):
            continue
        color_mask = grid == color
        if not bool(np.any(color_mask & inside)):
            return False
        if not bool(np.any(color_mask & ~inside)):
            return False

    if np.array_equal(crop, expected):
        return False
    control_count = int(np.count_nonzero(expected == int(control_color)))
    base_count = int(np.count_nonzero(expected == int(base_color)))
    return control_count > 0 and base_count > 0 and control_count < expected.size


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(_GEN_GRID_MIN, _GEN_GRID_MAX + 1))
    grid_w = int(rng.integers(_GEN_GRID_MIN, _GEN_GRID_MAX + 1))

    max_win_h = min(_GEN_WINDOW_MAX, grid_h - 2)
    max_win_w = min(_GEN_WINDOW_MAX, grid_w - 2)
    if max_win_h < _GEN_WINDOW_MIN or max_win_w < _GEN_WINDOW_MIN:
        return None
    win_h = int(rng.integers(_GEN_WINDOW_MIN, max_win_h + 1))
    win_w = int(rng.integers(_GEN_WINDOW_MIN, max_win_w + 1))

    r0 = int(rng.integers(1, grid_h - win_h))
    c0 = int(rng.integers(1, grid_w - win_w))
    r1 = r0 + win_h - 1
    c1 = c0 + win_w - 1
    box: BBox = (r0, c0, r1, c1)

    palette = [int(c) for c in rng.permutation(_GEN_COLORS)]
    generator_color, control_color, base_color = palette[:3]
    n_distractors = int(rng.integers(_GEN_DISTRACTOR_MIN, _GEN_DISTRACTOR_MAX + 1))
    distractors = palette[3 : 3 + n_distractors]
    field_colors = [int(base_color), *map(int, distractors)]

    marker_cells = _sample_marker_coords(rng, win_h, win_w)
    if marker_cells is None:
        return None

    crop = _paint_crop(rng, (win_h, win_w), base_color, control_color, distractors, marker_cells)
    if crop is None:
        return None

    outside_bg = _choice_int(rng, field_colors if rng.random() < 0.35 else distractors)
    grid = np.full((grid_h, grid_w), int(outside_bg), dtype=int)

    horizontal_sides = [0 if rng.random() < 0.5 else grid_h - 1 for _ in range(2)]
    vertical_sides = [0 if rng.random() < 0.5 else grid_w - 1 for _ in range(2)]
    generator_cells = {
        (horizontal_sides[0], c0),
        (horizontal_sides[1], c1),
        (r0, vertical_sides[0]),
        (r1, vertical_sides[1]),
    }
    if len(generator_cells) != 4:
        return None

    out_cells = _outside_cells(grid.shape, box, generator_cells)
    if len(out_cells) < len(field_colors):
        return None
    out_cells = _shuffled_cells(rng, out_cells)

    min_noise = len(field_colors)
    max_noise = min(len(out_cells), max(min_noise, int(round(len(out_cells) * _GEN_OUTSIDE_NOISE_FRAC[1]))))
    noise_count = int(rng.integers(min_noise, max_noise + 1))
    noise_cells = out_cells[:noise_count]
    shuffled_fields = [int(c) for c in rng.permutation(field_colors)]
    for (r, c), color in zip(noise_cells[: len(shuffled_fields)], shuffled_fields):
        grid[r, c] = int(color)
    for r, c in noise_cells[len(shuffled_fields) :]:
        grid[r, c] = _choice_int(rng, field_colors)

    grid[r0 : r1 + 1, c0 : c1 + 1] = crop
    for r, c in generator_cells:
        grid[r, c] = int(generator_color)

    expected = _build_expected((win_h, win_w), base_color, control_color, marker_cells)
    if not _validate_generated(grid, expected, generator_color, control_color, box, base_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid implicit boundary window input")
