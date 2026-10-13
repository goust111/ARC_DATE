from __future__ import annotations

import random

import numpy as np


BG = 8
SEP = 5
PALETTE = (0, 1, 2, 3, 4, 6, 7, 9)
MIN_SEG_LEN = 2
MAX_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    row_hits = [
        r
        for r in range(h)
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    col_hits = [
        c
        for c in range(w)
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    assert len(row_hits) + len(col_hits) == 1, "expected one full separator line"
    if row_hits:
        r = int(row_hits[0])
        return "horizontal", r, int(grid[r, 0])
    c = int(col_hits[0])
    return "vertical", c, int(grid[0, c])


def _components(grid: np.ndarray, mask: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            colors = set()
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                colors.add(int(grid[cr, cc]))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            assert len(colors) == 1, "each stripe must be monochromatic"
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": int(next(iter(colors))),
                    "coords": coords,
                    "r0": int(min(rows)),
                    "r1": int(max(rows)),
                    "c0": int(min(cols)),
                    "c1": int(max(cols)),
                }
            )
    return comps


def _side_type(comps: list[dict[str, object]]) -> str:
    assert comps, "each side must contain stripes"
    if all(int(comp["r0"]) == int(comp["r1"]) and int(comp["c1"]) > int(comp["c0"]) for comp in comps):
        return "horizontal"
    if all(int(comp["c0"]) == int(comp["c1"]) and int(comp["r1"]) > int(comp["r0"]) for comp in comps):
        return "vertical"
    raise AssertionError("could not classify stripe orientation")


def _distance_to_separator(comp: dict[str, object], sep_orient: str, axis: int, side: str) -> int:
    if sep_orient == "vertical":
        if side == "left":
            return int(axis - int(comp["c1"]) - 1)
        return int(int(comp["c0"]) - axis - 1)
    if side == "top":
        return int(axis - int(comp["r1"]) - 1)
    return int(int(comp["r0"]) - axis - 1)


def _collect_stripes(
    grid: np.ndarray, sep_orient: str, axis: int, bg: int, sep_color: int
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    if sep_orient == "vertical":
        regions = [
            ("left", grid[:, :axis]),
            ("right", grid[:, axis + 1 :]),
        ]
    else:
        regions = [
            ("top", grid[:axis, :]),
            ("bottom", grid[axis + 1 :, :]),
        ]

    typed: list[tuple[str, str, list[dict[str, object]]]] = []
    for side, region in regions:
        if sep_orient == "vertical":
            mask = (region != bg) & (region != sep_color)
            comps = _components(grid[:, :axis] if side == "left" else grid[:, axis + 1 :], mask)
            if side == "right":
                for comp in comps:
                    comp["coords"] = [(r, c + axis + 1) for r, c in comp["coords"]]
                    comp["c0"] = int(comp["c0"]) + axis + 1
                    comp["c1"] = int(comp["c1"]) + axis + 1
            typed.append((side, _side_type(comps), comps))
        else:
            mask = (region != bg) & (region != sep_color)
            comps = _components(grid[:axis, :] if side == "top" else grid[axis + 1 :, :], mask)
            if side == "bottom":
                for comp in comps:
                    comp["coords"] = [(r + axis + 1, c) for r, c in comp["coords"]]
                    comp["r0"] = int(comp["r0"]) + axis + 1
                    comp["r1"] = int(comp["r1"]) + axis + 1
            typed.append((side, _side_type(comps), comps))

    assert {kind for _side, kind, _comps in typed} == {"horizontal", "vertical"}

    horizontal: dict[int, tuple[int, int]] = {}
    vertical: dict[int, tuple[int, int]] = {}
    for side, kind, comps in typed:
        for comp in comps:
            dist = _distance_to_separator(comp, sep_orient, axis, side)
            assert dist >= 0, "stripe must stay on one side of the separator"
            color = int(comp["color"])
            if kind == "horizontal":
                row = int(comp["r0"])
                assert row not in horizontal, "horizontal stripes must use distinct rows"
                horizontal[row] = (color, dist)
            else:
                col = int(comp["c0"])
                assert col not in vertical, "vertical stripes must use distinct columns"
                vertical[col] = (color, dist)

    assert horizontal and vertical
    return horizontal, vertical


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the horizontal and vertical stripe families across the full canvas.
    At intersections, the stripe closer to the separator wins; ties keep
    the horizontal stripe color. The grey separator itself is preserved.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _bg_color(g)
    sep_orient, axis, sep_color = _find_separator(g, bg)
    horizontal, vertical = _collect_stripes(g, sep_orient, axis, bg, sep_color)

    out = np.full((h, w), bg, dtype=int)
    if sep_orient == "vertical":
        for row, (color, _dist) in horizontal.items():
            out[row, :] = color
        for col, (color, _dist) in vertical.items():
            out[:, col] = color
        for row, (h_color, h_dist) in horizontal.items():
            for col, (v_color, v_dist) in vertical.items():
                out[row, col] = h_color if h_dist <= v_dist else v_color
        out[:, axis] = sep_color
    else:
        for row, (color, _dist) in horizontal.items():
            out[row, :] = color
        for col, (color, _dist) in vertical.items():
            out[:, col] = color
        for row, (h_color, h_dist) in horizontal.items():
            for col, (v_color, v_dist) in vertical.items():
                out[row, col] = h_color if h_dist <= v_dist else v_color
        out[axis, :] = sep_color
    return out


def _sample_spaced_positions(length: int, count: int) -> list[int]:
    assert 0 < count <= (length + 1) // 2
    base = sorted(random.sample(range(length - count + 1), count))
    return [int(v + i) for i, v in enumerate(base)]


def _extend_spaced_distances(base: list[int], max_distance: int, max_count: int) -> list[int]:
    out = sorted(base)
    if len(out) >= max_count:
        return out
    target = random.randint(len(out), max_count)
    candidates = [d for d in range(max_distance + 1) if d not in out]
    random.shuffle(candidates)
    for d in candidates:
        if len(out) >= target:
            break
        if all(abs(d - prev) >= 2 for prev in out):
            out.append(d)
    return sorted(out)


def _extend_unique_distances(base: list[int], max_distance: int, max_count: int) -> list[int]:
    out = sorted(base)
    if len(out) >= max_count:
        return out
    target = random.randint(len(out), max_count)
    remaining = [d for d in range(max_distance + 1) if d not in out]
    random.shuffle(remaining)
    out.extend(remaining[: target - len(out)])
    return sorted(out)


def _build_distance_sets(pos_max: int, trunc_max: int, pos_count_max: int, trunc_count_max: int) -> tuple[list[int], list[int]]:
    limit = min(pos_max, trunc_max)
    assert limit >= 2
    center = random.randint(1, limit - 1)
    pos_base = [center - 1, center + 1]
    trunc_base = [center - 1, center, center + 1]
    pos_dists = _extend_spaced_distances(pos_base, pos_max, pos_count_max)
    trunc_dists = _extend_unique_distances(trunc_base, trunc_max, trunc_count_max)

    has_tie = bool(set(pos_dists) & set(trunc_dists))
    has_pos_smaller = any(pd < td for pd in pos_dists for td in trunc_dists)
    has_pos_larger = any(pd > td for pd in pos_dists for td in trunc_dists)
    assert has_tie and has_pos_smaller and has_pos_larger
    return pos_dists, trunc_dists


def _paint_horizontal(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    grid[row, c0 : c1 + 1] = color


def _paint_vertical(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    grid[r0 : r1 + 1, col] = color


def _vertical_scene(h: int, w: int) -> np.ndarray | None:
    sep_col = random.randint(4, w - 5)
    left_w = sep_col
    right_w = w - sep_col - 1
    horizontal_side = random.choice(("left", "right"))
    vertical_side = "right" if horizontal_side == "left" else "left"

    trunc_span = left_w if horizontal_side == "left" else right_w
    pos_span = left_w if vertical_side == "left" else right_w
    trunc_max = trunc_span - MIN_SEG_LEN
    pos_max = pos_span - 1
    if trunc_max < 2 or pos_max < 2:
        return None

    horiz_count_max = min(4, (h + 1) // 2, trunc_max + 1)
    vert_count_max = min(4, (pos_span + 1) // 2)
    if horiz_count_max < 3 or vert_count_max < 2:
        return None

    pos_dists, trunc_dists = _build_distance_sets(pos_max, trunc_max, vert_count_max, horiz_count_max)
    horiz_rows = _sample_spaced_positions(h, len(trunc_dists))

    grid = np.full((h, w), BG, dtype=int)
    grid[:, sep_col] = SEP
    colors = random.sample(PALETTE, len(trunc_dists) + len(pos_dists))

    for row, dist, color in zip(horiz_rows, trunc_dists, colors[: len(trunc_dists)]):
        if horizontal_side == "left":
            end = sep_col - dist - 1
            start = random.randint(0, end - MIN_SEG_LEN + 1)
        else:
            start = sep_col + dist + 1
            end = random.randint(start + MIN_SEG_LEN - 1, w - 1)
        _paint_horizontal(grid, row, start, end, color)

    for dist, color in zip(pos_dists, colors[len(trunc_dists) :]):
        col = sep_col - dist - 1 if vertical_side == "left" else sep_col + dist + 1
        r0 = random.randint(0, h - MIN_SEG_LEN)
        r1 = random.randint(r0 + MIN_SEG_LEN - 1, h - 1)
        _paint_vertical(grid, col, r0, r1, color)

    return grid


def _horizontal_scene(h: int, w: int) -> np.ndarray | None:
    sep_row = random.randint(4, h - 5)
    top_h = sep_row
    bottom_h = h - sep_row - 1
    horizontal_side = random.choice(("top", "bottom"))
    vertical_side = "bottom" if horizontal_side == "top" else "top"

    pos_span = top_h if horizontal_side == "top" else bottom_h
    trunc_span = top_h if vertical_side == "top" else bottom_h
    pos_max = pos_span - 1
    trunc_max = trunc_span - MIN_SEG_LEN
    if pos_max < 2 or trunc_max < 2:
        return None

    horiz_count_max = min(4, (pos_span + 1) // 2)
    vert_count_max = min(4, (w + 1) // 2, trunc_max + 1)
    if horiz_count_max < 2 or vert_count_max < 3:
        return None

    pos_dists, trunc_dists = _build_distance_sets(pos_max, trunc_max, horiz_count_max, vert_count_max)
    vert_cols = _sample_spaced_positions(w, len(trunc_dists))

    grid = np.full((h, w), BG, dtype=int)
    grid[sep_row, :] = SEP
    colors = random.sample(PALETTE, len(pos_dists) + len(trunc_dists))

    for dist, color in zip(pos_dists, colors[: len(pos_dists)]):
        row = sep_row - dist - 1 if horizontal_side == "top" else sep_row + dist + 1
        c0 = random.randint(0, w - MIN_SEG_LEN)
        c1 = random.randint(c0 + MIN_SEG_LEN - 1, w - 1)
        _paint_horizontal(grid, row, c0, c1, color)

    for col, dist, color in zip(vert_cols, trunc_dists, colors[len(pos_dists) :]):
        if vertical_side == "top":
            end = sep_row - dist - 1
            start = random.randint(0, end - MIN_SEG_LEN + 1)
        else:
            start = sep_row + dist + 1
            end = random.randint(start + MIN_SEG_LEN - 1, h - 1)
        _paint_vertical(grid, col, start, end, color)

    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape and not np.array_equal(out, grid)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        h = random.randint(10, 18)
        w = random.randint(10, 18)
        if random.random() < 0.5:
            grid = _vertical_scene(h, w)
        else:
            grid = _horizontal_scene(h, w)
        if grid is not None and _is_valid_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid weaving input")
