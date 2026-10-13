from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_COLORS = tuple(range(10))
_GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    return sorted(coords)


def _legend_candidate(grid: np.ndarray, bg: int, row0: int) -> dict | None:
    h, w = grid.shape
    if row0 < 0 or row0 + 1 >= h:
        return None
    mask0 = grid[row0] != bg
    mask1 = grid[row0 + 1] != bg
    if not mask0.any() or not np.array_equal(mask0, mask1):
        return None
    cols = np.flatnonzero(mask0)
    if cols.size < 2:
        return None
    if int(cols[-1] - cols[0] + 1) != int(cols.size):
        return None
    top = [int(grid[row0, c]) for c in cols]
    bottom = [int(grid[row0 + 1, c]) for c in cols]
    if len(set(top)) != len(top):
        return None
    mapping = {src: dst for src, dst in zip(top, bottom)}
    if any(color == bg for color in mapping):
        return None
    return {"row0": int(row0), "cols": cols.astype(int), "mapping": mapping}


def _find_legend(grid: np.ndarray, bg: int) -> dict:
    candidates = []
    for row0 in (0, grid.shape[0] - 2):
        candidate = _legend_candidate(grid, bg, int(row0))
        if candidate is not None:
            candidates.append(candidate)
    assert candidates, "expected a 2-row legend on the top or bottom edge"
    candidates.sort(key=lambda cand: len(cand["mapping"]), reverse=True)
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore noisy hollow rectangles and fill the heavily occluded ones via an edge legend.

    Input structure:
    - The background is the most common color.
    - Several axis-aligned hollow rectangles are drawn in distinct border colors, with some border pixels replaced by one shared noise color.
    - A 2-row legend sits on the top or bottom edge: the first row lists rectangle border colors and the second row lists their fill colors.

    Rule:
    - Read the edge legend as a border-color → fill-color mapping.
    - For each legend color, recover its rectangle bbox from the remaining visible border pixels, then restore every perimeter cell; perimeter cells currently holding the shared noise color count as missing pixels for that rectangle.
    - If the missing-pixel ratio is at most one quarter of the full perimeter, only repair the border; otherwise repair the border and fill the rectangle interior with the mapped fill color.

    Key insight:
    The shared noise color is not a real object color: it temporarily hides parts of otherwise regular hollow rectangles, so the visible border fragments plus the legend are enough to reconstruct both shape and fill behavior.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend = _find_legend(g, bg)
    mapping = legend["mapping"]

    body_mask = np.ones(g.shape, dtype=bool)
    row0 = int(legend["row0"])
    body_mask[row0 : row0 + 2, :] = False

    border_colors = set(mapping)
    body_colors = {int(color) for color in np.unique(g[body_mask]) if int(color) != bg}
    extra_colors = sorted(body_colors - border_colors)
    noise = extra_colors[0] if len(extra_colors) == 1 else None

    out = g.copy()
    seen_noise: set[tuple[int, int]] = set()
    for border_color, fill_color in mapping.items():
        coords = np.argwhere((g == int(border_color)) & body_mask)
        assert coords.size, "each legend color must appear on a rectangle border"
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        perimeter = _perimeter_coords(r0, c0, r1, c1)
        perimeter_set = set(perimeter)
        assert all((int(r), int(c)) in perimeter_set for r, c in coords)

        missing = []
        if noise is not None:
            missing = [(r, c) for r, c in perimeter if body_mask[r, c] and int(g[r, c]) == int(noise)]
            seen_noise.update(missing)

        for r, c in perimeter:
            out[r, c] = int(border_color)

        if len(missing) * 4 > len(perimeter) and r1 - r0 > 1 and c1 - c0 > 1:
            out[r0 + 1 : r1, c0 + 1 : c1] = int(fill_color)

    if noise is not None:
        all_noise = {
            (int(r), int(c))
            for r, c in np.argwhere((g == int(noise)) & body_mask)
        }
        assert seen_noise == all_noise, "every noise pixel must belong to exactly one rectangle perimeter"

    return out


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(lo, hi + 1))


def _branch_flags(n_rects: int) -> list[bool]:
    if n_rects == 2:
        high_count = 1
    elif n_rects == 3:
        high_count = _randint(1, 2)
    else:
        high_count = 2
    flags = [True] * high_count + [False] * (n_rects - high_count)
    _RNG.shuffle(flags)
    return flags


def _legend_rows(height: int, on_top: bool) -> tuple[int, int]:
    return (0, 1) if on_top else (height - 2, height - 1)


def _place_rectangles(height: int, width: int, n_rects: int, legend_top: bool) -> list[dict] | None:
    usable_top = 3 if legend_top else 1
    usable_bottom = height - 2 if legend_top else height - 4
    usable_left = 1
    usable_right = width - 2
    h_max = min(8, usable_bottom - usable_top + 1)
    w_max = min(9, usable_right - usable_left + 1)
    if h_max < 4 or w_max < 4:
        return None

    occupied = np.zeros((height, width), dtype=bool)
    used_edge_rows: set[int] = set()
    used_edge_cols: set[int] = set()
    rectangles: list[dict] = []

    for _ in range(n_rects):
        placed = False
        for _attempt in range(250):
            rect_h = _randint(4, h_max)
            rect_w = _randint(4, w_max)
            r0 = _randint(usable_top, usable_bottom - rect_h + 1)
            c0 = _randint(usable_left, usable_right - rect_w + 1)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            if r0 in used_edge_rows or r1 in used_edge_rows:
                continue
            if c0 in used_edge_cols or c1 in used_edge_cols:
                continue
            if occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                continue
            rectangles.append({"bbox": (r0, c0, r1, c1)})
            occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            used_edge_rows.update((r0, r1))
            used_edge_cols.update((c0, c1))
            placed = True
            break
        if not placed:
            return None
    return rectangles


def _side_keep_points(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return [
        (r0, _randint(c0 + 1, c1 - 1)),
        (r1, _randint(c0 + 1, c1 - 1)),
        (_randint(r0 + 1, r1 - 1), c0),
        (_randint(r0 + 1, r1 - 1), c1),
    ]


def _sample_missing(perimeter: list[tuple[int, int]], high_branch: bool) -> list[tuple[int, int]]:
    r0, c0 = perimeter[0]
    r1, c1 = perimeter[-1]
    keep = set(_side_keep_points(r0, c0, r1, c1))
    movable = [coord for coord in perimeter if coord not in keep]
    quarter = len(perimeter) // 4
    if high_branch:
        miss_count = _randint(quarter + 1, len(movable))
    else:
        miss_count = _randint(1, quarter)
    picked = _RNG.choice(len(movable), size=miss_count, replace=False)
    return [movable[int(i)] for i in np.atleast_1d(picked)]


def _render_base(case: dict) -> np.ndarray:
    height = int(case["height"])
    width = int(case["width"])
    grid = np.full((height, width), int(case["bg"]), dtype=int)

    legend_r0, legend_r1 = _legend_rows(height, bool(case["legend_top"]))
    start_col = int(case["legend_col"])
    for offset, rect in enumerate(case["rectangles"]):
        col = start_col + offset
        grid[legend_r0, col] = int(rect["border"])
        grid[legend_r1, col] = int(rect["fill"])

    for rect in case["rectangles"]:
        r0, c0, r1, c1 = map(int, rect["bbox"])
        perimeter = _perimeter_coords(r0, c0, r1, c1)
        for r, c in perimeter:
            grid[r, c] = int(rect["border"])
    return grid


def _render_input(case: dict) -> np.ndarray:
    grid = _render_base(case)
    noise = int(case["noise"])
    for rect in case["rectangles"]:
        for r, c in rect["missing"]:
            grid[r, c] = noise
    return grid


def _render_output(case: dict) -> np.ndarray:
    grid = _render_base(case)
    for rect in case["rectangles"]:
        if bool(rect["high_branch"]):
            r0, c0, r1, c1 = map(int, rect["bbox"])
            grid[r0 + 1 : r1, c0 + 1 : c1] = int(rect["fill"])
    return grid


def _build_case() -> dict | None:
    n_rects = _randint(2, 4)
    min_side = max(16, 11 + 3 * n_rects)
    height = _randint(min_side, 30)
    width = _randint(min_side, 30)
    legend_top = bool(_randint(0, 1))

    rectangles = _place_rectangles(height, width, n_rects, legend_top)
    if rectangles is None:
        return None

    palette = list(_RNG.permutation(_COLORS))
    bg = int(palette.pop())
    needed = 2 * n_rects + 1
    if len(palette) < needed:
        return None
    border_colors = [int(palette.pop()) for _ in range(n_rects)]
    fill_colors = [int(palette.pop()) for _ in range(n_rects)]
    noise = int(palette.pop())

    legend_col = _randint(1, width - n_rects - 1)
    high_flags = _branch_flags(n_rects)
    for rect, border_color, fill_color, high_branch in zip(rectangles, border_colors, fill_colors, high_flags):
        r0, c0, r1, c1 = map(int, rect["bbox"])
        perimeter = _perimeter_coords(r0, c0, r1, c1)
        rect["border"] = border_color
        rect["fill"] = fill_color
        rect["high_branch"] = bool(high_branch)
        rect["missing"] = _sample_missing(perimeter, bool(high_branch))

    return {
        "height": height,
        "width": width,
        "bg": bg,
        "noise": noise,
        "legend_top": legend_top,
        "legend_col": legend_col,
        "rectangles": rectangles,
    }


def _gen_attempt() -> np.ndarray | None:
    case = _build_case()
    if case is None:
        return None
    inp = _render_input(case)
    expected = _render_output(case)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid noisy-rectangle input")
