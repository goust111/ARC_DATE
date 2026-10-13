from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Prefer the shared ARC toolbox helpers, but keep a local fallback so this file
# still works when loaded directly via SourceFileLoader.
_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import crop_bbox as _arc_crop_bbox
    from arc_utils import find_components as _arc_find_components
    from arc_utils import get_bg as _arc_get_bg
except Exception:  # pragma: no cover - fallback path only
    _arc_crop_bbox = None
    _arc_find_components = None
    _arc_get_bg = None


def _get_bg(grid: np.ndarray) -> int:
    if _arc_get_bg is not None:
        return int(_arc_get_bg(grid))
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    if _arc_crop_bbox is not None:
        return _arc_crop_bbox(grid, box)
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _fallback_find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    monochromatic: bool = False,
) -> list[dict]:
    assert connectivity == 4
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    def neighbors(r: int, c: int):
        if r > 0:
            yield r - 1, c
        if r + 1 < h:
            yield r + 1, c
        if c > 0:
            yield r, c - 1
        if c + 1 < w:
            yield r, c + 1

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            seed_color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in neighbors(cr, cc):
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    if monochromatic and int(grid[nr, nc]) != seed_color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append(
                {
                    "color": seed_color if monochromatic else None,
                    "coords": np.array(coords, dtype=int),
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return components


def _find_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    if _arc_find_components is not None:
        return _arc_find_components(grid, bg=bg, connectivity=4, monochromatic=False)
    return _fallback_find_components(grid, bg=bg, connectivity=4, monochromatic=False)


def _infer_palette(grid: np.ndarray, bg: int) -> tuple[int, int]:
    fg = grid[grid != bg]
    colors, counts = np.unique(fg, return_counts=True)
    assert len(colors) == 2, "expected one fill color and one marker color"
    order = np.argsort(counts)
    assert int(counts[order[0]]) < int(counts[order[1]]), "marker color must be globally rarer"
    marker = int(colors[order[0]])
    fill = int(colors[order[1]])
    return fill, marker


def _rectangle_records(grid: np.ndarray) -> tuple[int, int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg = int(g[0, 0])
    fill, marker = _infer_palette(g, bg)
    rectangles = _find_rectangles(g, bg)
    assert len(rectangles) >= 3, "expected at least three rectangles"

    records: list[dict] = []
    for rect in rectangles:
        box = tuple(map(int, rect["bbox"]))
        crop = _crop_bbox(g, box)
        colors = {int(v) for v in np.unique(crop)}
        assert colors.issubset({fill, marker}), "rectangle crop must contain only fill and marker"
        assert np.all(crop != bg), "each rectangle must fill its bbox"
        marker_count = int(np.count_nonzero(crop == marker))
        assert marker_count > 0, "each rectangle must contain at least one marker pixel"
        records.append(
            {
                "bbox": box,
                "crop": crop,
                "area": int(crop.size),
                "marker_count": marker_count,
            }
        )
    return bg, fill, marker, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique solid rectangle whose marker count equals the total number of rectangles.

    Input structure
    - The upper-left cell is black background.
    - The foreground consists of separated 4-connected solid rectangles that all use one common fill color.
    - Inside those rectangles, a second rarer color appears as sparse marker pixels.

    Rule
    - Count the total number of foreground rectangles, call it N.
    - Count the marker pixels inside each rectangle.
    - Return the exact bounding-box crop of the unique rectangle whose marker count is exactly N.

    Key insight
    - The selection key is self-referential: the needed marker count is not written explicitly, it is the number of rectangles present in the same input.
    """
    _bg, _fill, _marker, records = _rectangle_records(grid)
    rect_count = len(records)
    winners = [record for record in records if record["marker_count"] == rect_count]
    assert len(winners) == 1, "expected exactly one self-count rectangle"
    return winners[0]["crop"].copy()


BG_COLOR = 0
RECT_COUNT_RANGE = (3, 6)
GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
RECT_H_RANGE = (3, 8)
RECT_W_RANGE = (3, 9)
RECT_GAP = 2
MARKER_AREA_FACTOR = 3
MARKER_EXTRA_POOL = 7
MAX_ATTEMPTS = 400
COLOR_CHOICES = tuple(range(1, 10))


def _sample_marker_counts(rect_count: int) -> list[int]:
    below_count = random.randint(1, rect_count - 2)
    above_count = rect_count - 1 - below_count
    below = random.sample(list(range(1, rect_count)), below_count)
    above_pool = list(range(rect_count + 1, rect_count + 1 + MARKER_EXTRA_POOL))
    above = random.sample(above_pool, above_count)
    counts = below + [rect_count] + above
    random.shuffle(counts)
    return counts


def _size_options(marker_count: int) -> list[tuple[int, int]]:
    options = []
    min_area = max(12, MARKER_AREA_FACTOR * marker_count + 2)
    for h in range(*RECT_H_RANGE):
        for w in range(*RECT_W_RANGE):
            area = h * w
            if area >= min_area and marker_count <= area // 2:
                options.append((h, w))
    return options


def _feasible_locations(occupied: np.ndarray, rect_h: int, rect_w: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    locations = []
    for r in range(h - rect_h + 1):
        for c in range(w - rect_w + 1):
            if r == 0 and c == 0:
                continue
            r0 = max(0, r - RECT_GAP)
            c0 = max(0, c - RECT_GAP)
            r1 = min(h, r + rect_h + RECT_GAP)
            c1 = min(w, c + rect_w + RECT_GAP)
            if not occupied[r0:r1, c0:c1].any():
                locations.append((r, c))
    return locations


def _place_rectangles(shape: tuple[int, int], sizes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    occupied = np.zeros(shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for idx in order:
        rect_h, rect_w = sizes[idx]
        locations = _feasible_locations(occupied, rect_h, rect_w)
        if not locations:
            return None
        pos = random.choice(locations)
        r, c = pos
        occupied[r : r + rect_h, c : c + rect_w] = True
        placements[idx] = pos
    return [pos for pos in placements if pos is not None]


def _render_rectangle(rect_h: int, rect_w: int, fill: int, marker: int, marker_count: int) -> np.ndarray:
    patch = np.full((rect_h, rect_w), fill, dtype=int)
    marker_cells = random.sample(range(rect_h * rect_w), marker_count)
    patch.reshape(-1)[marker_cells] = marker
    return patch


def _expected_crop(grid: np.ndarray) -> np.ndarray:
    _bg, _fill, _marker, records = _rectangle_records(grid)
    rect_count = len(records)
    winner = [record for record in records if record["marker_count"] == rect_count]
    assert len(winner) == 1
    return winner[0]["crop"]


def _validate_scene(grid: np.ndarray) -> bool:
    try:
        bg, fill, marker, records = _rectangle_records(grid)
    except Exception:
        return False

    if bg != BG_COLOR or fill == marker:
        return False

    rect_count = len(records)
    counts = [record["marker_count"] for record in records]
    if len(set(counts)) != rect_count:
        return False
    if counts.count(rect_count) != 1:
        return False
    if not (min(counts) < rect_count < max(counts)):
        return False
    if any(record["marker_count"] >= record["area"] for record in records):
        return False
    if np.count_nonzero(grid == marker) >= np.count_nonzero(grid == fill):
        return False

    expected = _expected_crop(grid)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.randint(*RECT_COUNT_RANGE)
    marker_counts = _sample_marker_counts(rect_count)
    sizes = []
    for marker_count in marker_counts:
        options = _size_options(marker_count)
        if not options:
            return None
        sizes.append(random.choice(options))

    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    placements = _place_rectangles((grid_h, grid_w), sizes)
    if placements is None:
        return None

    fill_color, marker_color = random.sample(COLOR_CHOICES, 2)
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    for (rect_h, rect_w), (r, c), marker_count in zip(sizes, placements, marker_counts):
        patch = _render_rectangle(rect_h, rect_w, fill_color, marker_color, marker_count)
        grid[r : r + rect_h, c : c + rect_w] = patch

    if not _validate_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
