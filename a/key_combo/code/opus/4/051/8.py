from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, in_bounds


BG = Color.BLACK
BAND_COLORS = (Color.BLUE, Color.GREEN, Color.TEAL)
MARKER = Color.RED
TRAIL = Color.PINK
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {step: (-step[0], -step[1]) for step in DIRS.values()}
GRID_SIDE_RANGE = (14, 19)
BAND_HEIGHT_OPTIONS = (2, 3)
TRAIL_LENGTH_OPTIONS = (2, 3)
MARKER_COUNT_OPTIONS = (2, 3)
MAX_ATTEMPTS = 500


def _top_band_height(grid: np.ndarray) -> int:
    band_mask = (grid != BG) & (grid != MARKER) & (grid != TRAIL)
    top_h = 0
    for r in range(grid.shape[0]):
        if band_mask[r].any():
            if r != top_h:
                raise AssertionError("band rows must be contiguous from the top edge")
            top_h += 1
        elif top_h > 0:
            break
    assert top_h > 0, "expected a non-empty top band"
    return int(top_h)


def _trail_info(grid: np.ndarray, r: int, c: int) -> tuple[tuple[int, int], int]:
    matches: list[tuple[tuple[int, int], int]] = []
    for step in DIRS.values():
        dr, dc = step
        rr, cc = r + dr, c + dc
        length = 0
        while in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == TRAIL:
            length += 1
            rr += dr
            cc += dc
        if length > 0:
            matches.append((step, int(length)))
    assert len(matches) == 1, "each marker must have exactly one straight trail"
    return matches[0]


def _first_obstacle(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> tuple[int, int] | None:
    dr, dc = step
    rr, cc = start[0] + dr, start[1] + dc
    while in_bounds(grid.shape, rr, cc):
        if int(grid[rr, cc]) != BG:
            return int(rr), int(cc)
        rr += dr
        cc += dc
    return None


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = step
    rr, cc = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while in_bounds(shape, rr, cc):
        cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return cells


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    top_h = _top_band_height(g)
    out = g.copy()
    out[-top_h:, :] = g[:top_h, :][::-1, :]

    red_coords = [tuple(map(int, coord)) for coord in np.argwhere(g == MARKER)]
    assert red_coords, "expected at least one red marker"

    for r, c in red_coords:
        source_dir, _trail_len = _trail_info(g, r, c)
        ray_dir = OPPOSITE[source_dir]
        hit = _first_obstacle(out, (r, c), ray_dir)
        if hit is not None:
            out[hit] = BG

    out[g == TRAIL] = BG
    for r, c in red_coords:
        out[r, c] = MARKER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the sparse top color band to the bottom, then erase the first obstacle hit by each marker's reverse ray.

    Input structure:
    - Black background with a 2-3 row sparse color band touching the top edge.
    - The same number of bottom rows are initially empty and serve as the mirror destination.
    - The middle area contains red marker pixels, each with one adjacent straight pink trail of length 2-3.

    Rule:
    - Copy the top band to the bottom rows in vertically reversed row order, preserving the original top band.
    - For each red marker, read the pink trail as the source direction and cast a ray from the marker in the opposite direction.
    - Erase the first non-black pixel hit by that ray, leave markers red, and finally erase every pink trail pixel.

    Key insight:
    - Some ray targets do not exist in the input at all; they only appear after the top band is mirrored into the bottom rows.
    """
    return _transform(grid)


def _sample_marker_types() -> list[str]:
    count = random.choice(MARKER_COUNT_OPTIONS)
    marker_types = ["up_hit", "down_hit"]
    if count == 3:
        marker_types.append(random.choices(
            ["up_hit", "down_hit", "up_none", "down_none"],
            weights=[2, 2, 3, 3],
            k=1,
        )[0])
    random.shuffle(marker_types)
    return marker_types


def _sample_hit_pattern(band_h: int, palette: tuple[int, ...]) -> list[int]:
    deepest = random.randrange(band_h)
    rows = [deepest]
    for r in range(deepest):
        if random.random() < 0.45:
            rows.append(r)
    if len(rows) == 1 and deepest > 0 and random.random() < 0.5:
        rows.append(random.randrange(deepest))

    pattern = [BG] * band_h
    for r in rows:
        pattern[r] = random.choice(palette)
    return pattern


def _build_top_band(width: int, band_h: int, column_modes: dict[int, str]) -> np.ndarray:
    palette = tuple(random.sample(BAND_COLORS, k=random.choice((2, 3))))
    band = np.zeros((band_h, width), dtype=int)

    for c in range(width):
        mode = column_modes.get(c, "random")
        if mode == "none":
            continue
        if mode == "hit":
            band[:, c] = _sample_hit_pattern(band_h, palette)
            continue

        if random.random() < 0.15:
            continue
        band[:, c] = _sample_hit_pattern(band_h, palette)

    for r in range(band_h):
        if not np.any(band[r] != BG):
            candidates = [c for c in range(width) if column_modes.get(c) != "none"]
            fill_col = random.choice(candidates)
            band[r, fill_col] = random.choice(palette)
        if np.all(band[r] != BG):
            candidates = [c for c in range(width) if column_modes.get(c) == "random"]
            if candidates:
                band[r, random.choice(candidates)] = BG

    used_colors = sorted(set(map(int, band[band != BG])))
    if len(used_colors) < 2:
        filled = list(zip(*np.where(band != BG)))
        if filled:
            rr, cc = random.choice(filled)
            alternatives = [color for color in palette if color != int(band[rr, cc])]
            if alternatives:
                band[rr, cc] = random.choice(alternatives)

    assert len(sorted(set(map(int, band[band != BG])))) >= 2, "need at least two band colors"
    return band


def _row_bounds(height: int, band_h: int, trail_len: int, marker_type: str) -> tuple[int, int]:
    if marker_type.startswith("up_"):
        lo = band_h + 2
        hi = height - band_h - trail_len - 2
    else:
        lo = band_h + trail_len + 2
        hi = height - band_h - 3
    return int(lo), int(hi)


def _place_marker_rows(height: int, band_h: int, marker_types: list[str], trail_lens: list[int]) -> list[int] | None:
    rows: list[int] = []
    for marker_type, trail_len in zip(marker_types, trail_lens):
        lo, hi = _row_bounds(height, band_h, trail_len, marker_type)
        choices = [r for r in range(lo, hi + 1) if all(abs(r - prev) >= 2 for prev in rows)]
        if not choices:
            return None
        rows.append(random.choice(choices))
    return rows


def _paint_marker(grid: np.ndarray, row: int, col: int, marker_type: str, trail_len: int) -> None:
    if marker_type.startswith("up_"):
        step = DIRS["down"]
    else:
        step = DIRS["up"]

    grid[row, col] = MARKER
    rr, cc = row, col
    for _ in range(trail_len):
        rr += step[0]
        cc += step[1]
        grid[rr, cc] = TRAIL


def _expected_target_region(height: int, band_h: int, marker_type: str) -> tuple[int, int]:
    if marker_type == "up_hit":
        return 0, band_h - 1
    if marker_type == "down_hit":
        return height - band_h, height - 1
    raise AssertionError("only hit markers have target regions")


def _validate_instance(grid: np.ndarray, marker_specs: list[dict]) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    band_h = _top_band_height(grid)
    if band_h not in BAND_HEIGHT_OPTIONS:
        return False
    if np.any(grid[-band_h:, :] != BG):
        return False

    red_coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == MARKER)]
    if len(red_coords) != len(marker_specs):
        return False

    mirrored = grid.copy()
    mirrored[-band_h:, :] = grid[:band_h, :][::-1, :]

    seen_paths: set[tuple[int, int]] = set()
    hit_targets: list[tuple[int, int]] = []
    saw_up_hit = False
    saw_down_hit = False

    for spec in marker_specs:
        row, col = int(spec["row"]), int(spec["col"])
        if int(grid[row, col]) != MARKER:
            return False

        try:
            source_dir, trail_len = _trail_info(grid, row, col)
        except AssertionError:
            return False
        if trail_len not in TRAIL_LENGTH_OPTIONS:
            return False
        ray_dir = OPPOSITE[source_dir]
        ray_cells = _ray_cells(grid.shape, (row, col), ray_dir)
        if any(cell in seen_paths for cell in ray_cells):
            return False
        seen_paths.update(ray_cells)

        hit = _first_obstacle(mirrored, (row, col), ray_dir)
        marker_type = str(spec["type"])
        if marker_type.endswith("_hit"):
            if hit is None:
                return False
            r0, r1 = _expected_target_region(grid.shape[0], band_h, marker_type)
            if not (r0 <= hit[0] <= r1):
                return False
            hit_targets.append(hit)
            if marker_type == "up_hit":
                saw_up_hit = True
            if marker_type == "down_hit":
                saw_down_hit = True
        else:
            if hit is not None:
                return False

    if not saw_up_hit or not saw_down_hit:
        return False
    if len(hit_targets) != len(set(hit_targets)):
        return False

    actual = main(grid)
    expected = _transform(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    band_h = random.choice(BAND_HEIGHT_OPTIONS)
    if height < band_h * 2 + 7 or width < 8:
        return None

    marker_types = _sample_marker_types()
    trail_lens = [random.choice(TRAIL_LENGTH_OPTIONS) for _ in marker_types]
    rows = _place_marker_rows(height, band_h, marker_types, trail_lens)
    if rows is None:
        return None

    col_pool = list(range(2, width - 2))
    marker_cols: list[int] = []
    for _ in marker_types:
        choices = [col for col in col_pool if all(abs(col - prev) >= 2 for prev in marker_cols)]
        if not choices:
            return None
        marker_cols.append(random.choice(choices))
    column_modes = {
        int(col): ("none" if marker_type.endswith("_none") else "hit")
        for col, marker_type in zip(marker_cols, marker_types)
    }
    band = _build_top_band(width, band_h, column_modes)

    for col, marker_type in zip(marker_cols, marker_types):
        if marker_type.endswith("_hit") and not np.any(band[:, col] != BG):
            return None
        if marker_type.endswith("_none") and np.any(band[:, col] != BG):
            return None

    grid = np.zeros((height, width), dtype=int)
    grid[:band_h, :] = band

    specs: list[dict] = []
    for row, col, marker_type, trail_len in zip(rows, marker_cols, marker_types, trail_lens):
        _paint_marker(grid, row, col, marker_type, trail_len)
        specs.append({
            "type": marker_type,
            "row": int(row),
            "col": int(col),
            "trail_len": int(trail_len),
        })

    if not _validate_instance(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
