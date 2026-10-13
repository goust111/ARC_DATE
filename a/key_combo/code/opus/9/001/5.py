import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("arc_utils_shared", _ARC_UTILS_PATH)
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color

PALETTE = (Color.GREEN, Color.YELLOW, Color.RED, Color.ORANGE)
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == Color.GREY)]
    assert len(rows) == 1, "expected one full grey separator row"
    row = rows[0]
    assert 0 < row < grid.shape[0] - 1
    return row


def _parse(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    sep = _separator_row(g)

    ref_region = g[:sep]
    src_region = g[sep + 1 :]
    assert ref_region.size > 0 and src_region.size > 0

    ref_bg = _dominant_color(ref_region)
    src_bg = _dominant_color(src_region)
    assert ref_bg == Color.BLUE
    assert src_bg == Color.MAROON
    assert np.all(g[: sep - 1] == ref_bg)

    ref_row = sep - 1
    ref_cols = [c for c in range(w) if int(g[ref_row, c]) != ref_bg]
    assert 3 <= len(ref_cols) <= len(PALETTE)
    assert all(int(g[ref_row, c]) in PALETTE for c in ref_cols)
    assert len(set(int(g[ref_row, c]) for c in ref_cols)) == len(ref_cols)

    anchors = np.argwhere(g == Color.PINK)
    assert anchors.shape == (1, 2), "expected one magenta anchor"
    anchor = tuple(map(int, anchors[0]))
    assert sep < anchor[0] < h - 1
    assert 0 < anchor[1] < w - 1

    src_mask = np.zeros_like(g, dtype=bool)
    src_mask[sep + 1 :] = True
    src_pixels = np.argwhere(src_mask & (g != src_bg) & (g != Color.PINK))
    assert src_pixels.shape[0] == len(ref_cols)
    src_by_col: dict[int, tuple[int, int]] = {}
    for r, c in src_pixels:
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        assert rr > sep
        assert color in PALETTE
        assert cc in ref_cols
        assert cc not in src_by_col
        src_by_col[cc] = (rr, color)

    assert sorted(src_by_col) == ref_cols
    assert set(color for _, color in src_by_col.values()) == set(int(g[ref_row, c]) for c in ref_cols)
    assert all(anchor[0] != r for r, _ in src_by_col.values())
    assert all(anchor[1] != c for c in src_by_col)

    return {
        "grid": g,
        "shape": (h, w),
        "sep": sep,
        "ref_bg": ref_bg,
        "src_bg": src_bg,
        "ref_row": ref_row,
        "ref_cols": ref_cols,
        "ref_colors": {c: int(g[ref_row, c]) for c in ref_cols},
        "anchor": anchor,
        "src_by_col": src_by_col,
    }


def _solve(grid: np.ndarray, *, use_distance: bool = True, use_color: bool = True) -> np.ndarray:
    info = _parse(grid)
    g = info["grid"]
    h, _w = info["shape"]
    sep = info["sep"]
    src_bg = info["src_bg"]
    anchor_r, anchor_c = info["anchor"]

    out = g.copy()
    for c in info["ref_cols"]:
        src_r, src_color = info["src_by_col"][c]
        ref_color = info["ref_colors"][c]
        moving_up = (src_color == ref_color) if use_color else False
        target_row = sep + 1 if moving_up else h - 1
        wanted = abs(target_row - src_r)
        distance = abs(src_r - anchor_r) + abs(c - anchor_c)
        step = min(wanted, distance) if use_distance else wanted
        new_r = src_r - step if moving_up else src_r + step
        out[src_r, c] = src_bg
        out[new_r, c] = src_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each source pixel in its column, with direction from color alignment and distance capped by a Manhattan field.

    Input structure:
    - A full grey horizontal separator splits the grid into a blue reference side above and a maroon source side below.
    - The row just above the separator contains several distinct colored reference pixels, one per active column.
    - The source side contains one colored source pixel in each of those same columns and one interior magenta anchor pixel that shares no row or column with any source pixel.

    Rule:
    - Compare each source pixel with the reference pixel in the same column.
    - If the colors match, that source pixel moves upward toward the separator; otherwise it moves downward toward the source-side bottom edge.
    - Its actual displacement is capped by its Manhattan distance to the magenta anchor, so the moved amount is the minimum of the default slide distance and that anchor distance.
    - Erase the original source pixel locations, draw the slid pixels in column order, and keep the reference side, separator, and anchor unchanged.

    Key insight:
    - Two independent clues must be combined: same-column color comparison decides the slide direction, while the anchor's Manhattan-distance field limits how far each pixel may travel.
    """
    return _solve(grid, use_distance=True, use_color=True)


def _pick_spaced_columns(width: int, count: int) -> list[int] | None:
    cols = list(range(1, width - 1))
    random.shuffle(cols)
    chosen: list[int] = []
    for c in cols:
        if all(abs(c - prev) >= 2 for prev in chosen):
            chosen.append(c)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _build_case() -> np.ndarray | None:
    width = random.randint(10, 16)
    count = random.randint(3, 4)
    cols = _pick_spaced_columns(width, count)
    if cols is None:
        return None

    possible_anchor_cols = [c for c in range(1, width - 1) if c not in cols]
    random.shuffle(possible_anchor_cols)

    for anchor_c in possible_anchor_cols:
        dr = random.randint(1, 4)
        distances = [dr + abs(anchor_c - c) for c in cols]
        bottom_distance = max(distances)
        if len(set(distances)) < 2:
            continue
        if not any(d < bottom_distance for d in distances):
            continue
        if bottom_distance > 8:
            continue

        sep = random.randint(4, 7)
        source_height = bottom_distance + 1
        height = sep + 1 + source_height
        if height > 24:
            continue

        source_row = sep + 1
        anchor_r = source_row + dr
        if not (sep < anchor_r < height - 1):
            continue

        far_choices = [idx for idx, d in enumerate(distances) if d == bottom_distance]
        near_choices = [idx for idx, d in enumerate(distances) if d < bottom_distance]
        if not far_choices or not near_choices:
            continue
        far_idx = random.choice(far_choices)
        near_idx = random.choice(near_choices)

        ref_colors = random.sample(list(PALETTE), count)
        src_colors = ref_colors.copy()
        src_colors[far_idx], src_colors[near_idx] = src_colors[near_idx], src_colors[far_idx]

        grid = np.full((height, width), Color.MAROON, dtype=int)
        grid[:sep] = Color.BLUE
        grid[sep] = Color.GREY
        grid[anchor_r, anchor_c] = Color.PINK

        ref_row = sep - 1
        for idx, c in enumerate(cols):
            grid[ref_row, c] = ref_colors[idx]
            grid[source_row, c] = src_colors[idx]

        return grid
    return None


def _validate_case(grid: np.ndarray) -> bool:
    info = _parse(grid)
    sep = info["sep"]
    h, _w = info["shape"]
    source_row = sep + 1
    bottom_distance = h - 1 - source_row

    same_cols = []
    diff_cols = []
    distances = {}
    for c in info["ref_cols"]:
        src_r, src_color = info["src_by_col"][c]
        ref_color = info["ref_colors"][c]
        distance = abs(src_r - info["anchor"][0]) + abs(c - info["anchor"][1])
        distances[c] = distance
        if src_color == ref_color:
            same_cols.append(c)
        else:
            diff_cols.append(c)

    if not same_cols or len(diff_cols) < 2:
        return False
    if not any(distances[c] == bottom_distance for c in diff_cols):
        return False
    if not any(distances[c] < bottom_distance for c in diff_cols):
        return False

    out = main(grid)
    no_distance = _solve(grid, use_distance=False, use_color=True)
    no_color = _solve(grid, use_distance=True, use_color=False)

    if np.array_equal(out, no_distance):
        return False
    if np.array_equal(out, no_color):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is not None and _validate_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
