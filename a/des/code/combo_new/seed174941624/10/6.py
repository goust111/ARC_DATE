from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BORDER = 5
DOT = 8
FILL_COLORS = [1, 2, 3, 4, 6, 7, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400

Coord = tuple[int, int]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(int(x) for x in grid.flat).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, val in enumerate(mask_1d.tolist()):
        if val and start is None:
            start = int(idx)
        elif not val and start is not None:
            runs.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(mask_1d) - 1)))
    return runs


def _split_legend_and_canvas(grid: np.ndarray, bg: int) -> tuple[slice, slice]:
    active_cols = np.any(grid != int(bg), axis=0)
    col_runs = _runs(active_cols)
    assert len(col_runs) >= 2, "expected legend block on the left and canvas content on the right"
    legend_start, legend_end = col_runs[0]
    assert np.any(active_cols[legend_end + 1 :]), "expected non-empty canvas to the right of the legend"
    return slice(int(legend_start), int(legend_end + 1)), slice(int(legend_end + 1), grid.shape[1])


def _legend_mapping(legend: np.ndarray, bg: int) -> tuple[dict[int, int], int]:
    counts = Counter(int(x) for x in legend.flat if int(x) != int(bg))
    assert counts, "legend must contain non-background pixels"
    dot_color = int(counts.most_common(1)[0][0])

    mapping: dict[int, int] = {}
    for r0, r1 in _runs(np.any(legend != int(bg), axis=1)):
        entry = legend[r0 : r1 + 1]
        n_dots = int(np.count_nonzero(entry == int(dot_color)))
        key_colors = [int(color) for color in np.unique(entry) if int(color) not in (int(bg), int(dot_color))]
        assert len(key_colors) == 1, "each legend row must contain exactly one key color"
        assert n_dots > 0, "each legend row must contain at least one dot"
        mapping[int(n_dots)] = int(key_colors[0])

    assert mapping, "legend mapping must be non-empty"
    return mapping, int(dot_color)


def _fill_canvas(canvas: np.ndarray, mapping: dict[int, int], dot_color: int) -> np.ndarray:
    counts = Counter(int(x) for x in canvas.flat if int(x) != int(BG))
    counts.pop(int(dot_color), None)
    assert counts, "canvas must contain a border color"
    border_color = int(counts.most_common(1)[0][0])

    out = canvas.copy()
    for comp in _components(canvas != int(border_color)):
        if any(r == 0 or c == 0 or r == canvas.shape[0] - 1 or c == canvas.shape[1] - 1 for r, c in comp):
            continue
        dot_count = sum(1 for r, c in comp if int(canvas[r, c]) == int(dot_color))
        assert dot_count in mapping, "enclosed region dot count must appear in the legend"
        fill = int(mapping[int(dot_count)])
        for r, c in comp:
            out[r, c] = int(fill)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each enclosed gray-outlined region by looking up its dot count in the legend on the left.

    Input structure:
    - The background is black.
    - On the left, a vertical legend has one colored key pixel per row and 1 to 5 same-colored marker dots immediately to its right; blank rows separate the legend entries.
    - To the right, one-pixel-thick gray boundaries form several closed regions whose interiors contain black cells and some of the same marker dots.

    Rule:
    - Read the legend row by row and build a mapping from "number of marker dots" to the row's key color.
    - In the right-hand canvas, treat the gray boundary pixels as walls and find every 4-connected non-wall region that does not touch the canvas boundary.
    - Count the marker dots inside each enclosed region, look up the corresponding color from the legend, and repaint the whole enclosed region with that color.
    - Keep the gray boundaries and the legend unchanged; the marker dots inside the closed regions disappear into the fill color.

    Key insight:
    - The left block is a count-based legend: dot quantity, not dot position, chooses a color, and the same count-to-color code must then be applied to enclosed regions on the right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    legend_cols, canvas_cols = _split_legend_and_canvas(g, bg)
    mapping, dot_color = _legend_mapping(g[:, legend_cols], bg)

    out = g.copy()
    out[:, canvas_cols] = _fill_canvas(g[:, canvas_cols], mapping, dot_color)
    return out


def _core_mask(solid: np.ndarray) -> np.ndarray:
    core = np.zeros_like(solid, dtype=bool)
    if solid.shape[0] < 3 or solid.shape[1] < 3:
        return core
    core[1:-1, 1:-1] = (
        solid[1:-1, 1:-1]
        & solid[:-2, 1:-1]
        & solid[2:, 1:-1]
        & solid[1:-1, :-2]
        & solid[1:-1, 2:]
    )
    return core


def _valid_solid(solid: np.ndarray, min_core_area: int) -> bool:
    if int(np.count_nonzero(solid)) == 0:
        return False
    if len(_components(solid)) != 1:
        return False
    core = _core_mask(solid)
    if int(np.count_nonzero(core)) < int(min_core_area):
        return False
    if len(_components(core)) != 1:
        return False
    boundary = solid & (~core)
    return len(_components(boundary)) == 1


def _try_carve_notch(solid: np.ndarray, required_dots: int) -> np.ndarray | None:
    h, w = solid.shape
    side = random.choice(["top", "bottom", "left", "right"])
    test = solid.copy()

    if side in ("top", "bottom"):
        if w < 5 or h < 5:
            return None
        notch_w = random.randint(1, w - 4)
        start = random.randint(1, w - notch_w - 1)
        depth = random.randint(1, max(1, min(2, h - 3)))
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        test[rows, start : start + notch_w] = False
    else:
        if h < 5 or w < 5:
            return None
        notch_h = random.randint(1, h - 4)
        start = random.randint(1, h - notch_h - 1)
        depth = random.randint(1, max(1, min(2, w - 3)))
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        test[start : start + notch_h, cols] = False

    if _valid_solid(test, required_dots + 2):
        return test
    return None


def _make_region_patch(required_dots: int) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(5, 8)
        w = random.randint(5, 8)
        solid = np.ones((h, w), dtype=bool)
        if not _valid_solid(solid, required_dots + 2):
            continue

        for _ in range(random.randint(0, 3)):
            carved = _try_carve_notch(solid, required_dots)
            if carved is not None:
                solid = carved

        core = _core_mask(solid)
        boundary = solid & (~core)
        interior_cells = [(int(r), int(c)) for r, c in np.argwhere(core)]
        if len(interior_cells) < required_dots + 2:
            continue

        dots = set(random.sample(interior_cells, required_dots))
        patch = np.zeros((h, w), dtype=int)
        patch[boundary] = int(BORDER)
        for r, c in dots:
            patch[int(r), int(c)] = int(DOT)
        return patch
    return None


def _build_legend(entries: list[tuple[int, int]]) -> np.ndarray:
    gaps = [random.randint(1, 2) for _ in range(max(0, len(entries) - 1))]
    height = len(entries) + sum(gaps)
    width = 1 + max(count for count, _color in entries)
    legend = np.zeros((height, width), dtype=int)
    row = 0
    for idx, (count, color) in enumerate(entries):
        legend[int(row), 0] = int(color)
        legend[int(row), 1 : 1 + count] = int(DOT)
        row += 1
        if idx < len(gaps):
            row += gaps[idx]
    return legend


def _expanded_overlap(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rs = max(0, r0 - 1)
    re = min(blocked.shape[0], r0 + h + 1)
    cs = max(0, c0 - 1)
    ce = min(blocked.shape[1], c0 + w + 1)
    return bool(blocked[rs:re, cs:ce].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rs = max(0, r0 - 1)
    re = min(blocked.shape[0], r0 + h + 1)
    cs = max(0, c0 - 1)
    ce = min(blocked.shape[1], c0 + w + 1)
    blocked[rs:re, cs:ce] = True


def _paste_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, row_range: tuple[int, int], col_range: tuple[int, int]) -> bool:
    ph, pw = patch.shape
    r_min, r_max = row_range
    c_min, c_max = col_range
    if r_min > r_max or c_min > c_max:
        return False

    positions = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        if _expanded_overlap(blocked, r0, c0, ph, pw):
            continue
        mask = patch != int(BG)
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        if bool((sub[mask] != int(BG)).any()):
            continue
        sub[mask] = patch[mask]
        _mark_blocked(blocked, r0, c0, ph, pw)
        return True
    return False


def _sample_region_counts(legend_counts: list[int], n_regions: int) -> list[int]:
    counts = random.sample(legend_counts, min(len(legend_counts), n_regions))
    while len(counts) < n_regions:
        counts.append(random.choice(legend_counts))
    random.shuffle(counts)
    return [int(count) for count in counts]


def _gen_attempt() -> np.ndarray | None:
    n_legend = random.randint(2, 5)
    legend_counts = random.sample([1, 2, 3, 4, 5], n_legend)
    legend_colors = random.sample(FILL_COLORS, n_legend)
    legend_entries = list(zip(legend_counts, legend_colors))
    random.shuffle(legend_entries)
    legend = _build_legend(legend_entries)

    n_regions = random.randint(2, 4)
    region_counts = _sample_region_counts(legend_counts, n_regions)
    patches: list[np.ndarray] = []
    for count in region_counts:
        patch = _make_region_patch(int(count))
        if patch is None:
            return None
        patches.append(patch)

    max_patch_h = max(p.shape[0] for p in patches)
    max_patch_w = max(p.shape[1] for p in patches)
    left_margin = random.randint(1, 2)
    separator = random.randint(2, 4)
    min_h = max(legend.shape[0] + 2, max_patch_h + 4, 12)
    min_w = left_margin + legend.shape[1] + separator + max_patch_w + 4
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    legend_top = random.randint(1, grid_h - legend.shape[0] - 1)
    legend_left = left_margin
    canvas_left = legend_left + legend.shape[1] + separator

    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    grid[legend_top : legend_top + legend.shape[0], legend_left : legend_left + legend.shape[1]] = legend
    _mark_blocked(blocked, legend_top, legend_left, legend.shape[0], legend.shape[1])

    for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        ph, pw = patch.shape
        ok = _paste_patch(
            grid,
            blocked,
            patch,
            (1, grid_h - ph - 2),
            (canvas_left, grid_w - pw - 2),
        )
        if not ok:
            return None

    # Keep the legend as the leftmost active block and ensure a real separator before the canvas.
    active_cols = np.any(grid != int(BG), axis=0)
    col_runs = _runs(active_cols)
    if len(col_runs) < 2:
        return None
    if col_runs[0][0] != legend_left or col_runs[0][1] != legend_left + legend.shape[1] - 1:
        return None

    mapping, dot_color = _legend_mapping(grid[:, legend_left : legend_left + legend.shape[1]], BG)
    canvas = grid[:, legend_left + legend.shape[1] :]
    filled = _fill_canvas(canvas, mapping, dot_color)
    if np.array_equal(filled, canvas):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-region legend puzzle")
