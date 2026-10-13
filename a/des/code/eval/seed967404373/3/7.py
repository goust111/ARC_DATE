import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
GRID_SIZES = (15, 17, 19, 21, 23)
PAIR_COLOR_POOL = tuple(c for c in range(10) if c not in {BG, GREY})
MAX_SEGMENT_REACH = 3
MAX_GENERATION_ATTEMPTS = 400


def _center(shape: tuple[int, int]) -> tuple[float, float]:
    h, w = shape
    return (h - 1) / 2.0, (w - 1) / 2.0


def _dist2(shape: tuple[int, int], pos: tuple[int, int]) -> float:
    cr, cc = _center(shape)
    r, c = pos
    return (r - cr) ** 2 + (c - cc) ** 2


def _choose_center_closer(
    shape: tuple[int, int],
    a: tuple[int, int],
    b: tuple[int, int],
) -> tuple[int, int]:
    da = _dist2(shape, a)
    db = _dist2(shape, b)
    if da < db:
        return a
    if db < da:
        return b
    return min(a, b)


def _in_bounds(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _in_legend(pos: tuple[int, int], legend_size: int) -> bool:
    return pos[0] < legend_size and pos[1] < legend_size


def _detect_legend(grid: np.ndarray) -> tuple[int, list[int], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    border = int(g[0, 0])
    assert border == GREY, "top-left legend must use a grey border"

    top_run = 0
    while top_run < w and int(g[0, top_run]) == border:
        top_run += 1
    left_run = 0
    while left_run < h and int(g[left_run, 0]) == border:
        left_run += 1
    legend_size = min(top_run, left_run)
    assert legend_size >= 5, "legend must be at least 5x5"

    patch = g[:legend_size, :legend_size]
    assert np.all(patch[0, :] == border)
    assert np.all(patch[-1, :] == border)
    assert np.all(patch[:, 0] == border)
    assert np.all(patch[:, -1] == border)

    inner = patch[1:-1, 1:-1]
    header_row = [int(v) for v in inner[0, 1:]]
    header_col = [int(v) for v in inner[1:, 0]]
    assert header_row == header_col
    assert len(header_row) >= 2
    assert len(set(header_row)) == len(header_row)

    table: dict[tuple[int, int], int] = {}
    for i, color_a in enumerate(header_row):
        for j, color_b in enumerate(header_row):
            result = int(inner[i + 1, j + 1])
            mirror = int(inner[j + 1, i + 1])
            assert result == mirror, "legend fusion table must be symmetric"
            table[(color_a, color_b)] = result
    return legend_size, header_row, table


def _collapse_pair(
    p0: tuple[int, int],
    p1: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        left, right = sorted((c0, c1))
        while right - left > 1:
            left += 1
            right -= 1
        if left == right:
            return (r0, left)
        return _choose_center_closer(shape, (r0, left), (r0, right))
    assert c0 == c1, "same-color pair must share a row or a column"
    top, bottom = sorted((r0, r1))
    while bottom - top > 1:
        top += 1
        bottom -= 1
    if top == bottom:
        return (top, c0)
    return _choose_center_closer(shape, (top, c0), (bottom, c0))


def _find_adjacent_fusion(
    active: dict[tuple[int, int], int]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for r, c in sorted(active):
        for nb in ((r, c + 1), (r + 1, c)):
            if nb in active and active[nb] != active[(r, c)]:
                return (r, c), nb
    return None


def _react_pixels(
    active: dict[tuple[int, int], int],
    table: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> dict[tuple[int, int], int]:
    pixels = dict(active)
    while True:
        pair = _find_adjacent_fusion(pixels)
        if pair is None:
            return pixels
        p0, p1 = pair
        c0 = pixels[p0]
        c1 = pixels[p1]
        assert (c0, c1) in table, "fusion pair missing from legend"
        dest = _choose_center_closer(shape, p0, p1)
        color = table[(c0, c1)]
        del pixels[p0]
        del pixels[p1]
        assert dest not in pixels
        pixels[dest] = color


def _render_pixels(
    shape: tuple[int, int],
    legend: np.ndarray,
    pixels: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[: legend.shape[0], : legend.shape[1]] = legend
    for (r, c), color in pixels.items():
        assert not _in_legend((r, c), legend.shape[0])
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    shape = tuple(map(int, g.shape))

    legend_size, legend_colors, table = _detect_legend(g)
    legend = g[:legend_size, :legend_size].copy()

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g != BG):
        rr, cc = int(r), int(c)
        if _in_legend((rr, cc), legend_size):
            continue
        by_color.setdefault(int(g[rr, cc]), []).append((rr, cc))

    assert by_color, "expected non-legend colored pairs"
    assert set(by_color).issubset(set(legend_colors))
    assert all(len(coords) == 2 for coords in by_color.values())

    merged: dict[tuple[int, int], int] = {}
    for color, coords in by_color.items():
        pos = _collapse_pair(coords[0], coords[1], shape)
        assert not _in_legend(pos, legend_size)
        assert pos not in merged
        merged[pos] = color

    final_pixels = _react_pixels(merged, table, shape)
    return _render_pixels(shape, legend, final_pixels)


def _sample_mode() -> str:
    return random.choice(("none", "one", "two"))


def _build_motifs(input_colors: list[int], mode: str) -> list[dict[str, object]]:
    colors = input_colors[:]
    random.shuffle(colors)
    motifs: list[dict[str, object]] = []
    if mode == "none":
        for color in colors:
            motifs.append({"kind": "iso", "colors": (color,)})
        return motifs
    if mode == "one":
        motifs.append(
            {
                "kind": "fuse",
                "colors": (colors[0], colors[1]),
                "axis": random.choice(("h", "v")),
            }
        )
        for color in colors[2:]:
            motifs.append({"kind": "iso", "colors": (color,)})
        random.shuffle(motifs)
        return motifs
    assert mode == "two"
    motifs.append(
        {
            "kind": "fuse",
            "colors": (colors[0], colors[1]),
            "axis": random.choice(("h", "v")),
        }
    )
    motifs.append(
        {
            "kind": "fuse",
            "colors": (colors[2], colors[3]),
            "axis": random.choice(("h", "v")),
        }
    )
    random.shuffle(motifs)
    return motifs


def _build_table(
    legend_colors: list[int],
    fuse_pairs: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    colors = legend_colors[:]
    forced = {tuple(sorted(pair)) for pair in fuse_pairs}
    table: dict[tuple[int, int], int] = {}
    for color in colors:
        table[(color, color)] = color
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            key = tuple(sorted((a, b)))
            if key in forced:
                choices = [c for c in colors if c not in key]
                result = random.choice(choices if choices else colors)
            else:
                result = random.choice(colors)
            table[(a, b)] = result
            table[(b, a)] = result
    return table


def _build_legend(legend_colors: list[int], table: dict[tuple[int, int], int]) -> np.ndarray:
    n = len(legend_colors)
    size = n + 3
    patch = np.full((size, size), GREY, dtype=int)
    inner = np.full((n + 1, n + 1), BG, dtype=int)
    inner[0, 1:] = legend_colors
    inner[1:, 0] = legend_colors
    for i, a in enumerate(legend_colors):
        for j, b in enumerate(legend_colors):
            inner[i + 1, j + 1] = table[(a, b)]
    patch[1:-1, 1:-1] = inner
    return patch


def _candidate_cells(
    shape: tuple[int, int],
    legend_size: int,
    kind: str,
) -> list[list[tuple[int, int]]]:
    h, w = shape
    out: list[list[tuple[int, int]]] = []
    if kind == "iso":
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                cell = (r, c)
                if not _in_legend(cell, legend_size):
                    out.append([cell])
        return out
    if kind == "hpair":
        for r in range(1, h - 1):
            for c in range(1, w - 2):
                cells = [(r, c), (r, c + 1)]
                if all(not _in_legend(cell, legend_size) for cell in cells):
                    out.append(cells)
        return out
    assert kind == "vpair"
    for r in range(1, h - 2):
        for c in range(1, w - 1):
            cells = [(r, c), (r + 1, c)]
            if all(not _in_legend(cell, legend_size) for cell in cells):
                out.append(cells)
    return out


def _clear_of_existing(
    candidate: list[tuple[int, int]],
    existing: list[tuple[int, int]],
) -> bool:
    for r0, c0 in candidate:
        for r1, c1 in existing:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _place_targets(
    shape: tuple[int, int],
    legend_size: int,
    motifs: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[tuple[int, int], int], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    objects: list[dict[str, object]] = []
    active: dict[tuple[int, int], int] = {}
    fuse_positions: list[tuple[tuple[int, int], tuple[int, int]]] = []
    occupied_cells: list[tuple[int, int]] = []

    for motif in motifs:
        kind = str(motif["kind"])
        candidate_kind = "iso" if kind == "iso" else ("hpair" if motif["axis"] == "h" else "vpair")
        candidates = _candidate_cells(shape, legend_size, candidate_kind)
        random.shuffle(candidates)
        placed = False
        for cells in candidates:
            if not _clear_of_existing(cells, occupied_cells):
                continue
            colors = list(motif["colors"])
            if kind == "iso":
                pos = cells[0]
                color = int(colors[0])
                objects.append({"color": color, "merged": pos, "axes": ["h", "v"]})
                active[pos] = color
                occupied_cells.extend(cells)
                placed = True
            else:
                pos0, pos1 = cells
                color0, color1 = map(int, colors)
                axis_pref = "v" if motif["axis"] == "h" else "h"
                objects.append({"color": color0, "merged": pos0, "axes": [axis_pref]})
                objects.append({"color": color1, "merged": pos1, "axes": [axis_pref]})
                active[pos0] = color0
                active[pos1] = color1
                fuse_positions.append((pos0, pos1))
                occupied_cells.extend(cells)
                placed = True
            if placed:
                break
        if not placed:
            return None
    return objects, active, fuse_positions


def _segment_plan(
    shape: tuple[int, int],
    legend_size: int,
    merged: tuple[int, int],
    axis: str,
    sign: int,
    reach: int,
) -> dict[str, object] | None:
    dr, dc = (0, sign) if axis == "h" else (sign, 0)
    partner = (merged[0] + dr, merged[1] + dc)
    if not _in_bounds(shape, partner):
        return None
    if _in_legend(partner, legend_size):
        return None
    if _dist2(shape, merged) >= _dist2(shape, partner):
        return None

    start = (merged[0] - reach * dr, merged[1] - reach * dc)
    end = (partner[0] + reach * dr, partner[1] + reach * dc)
    if not _in_bounds(shape, start) or not _in_bounds(shape, end):
        return None

    cells = []
    if axis == "h":
        step = 1 if end[1] >= start[1] else -1
        for c in range(start[1], end[1] + step, step):
            cells.append((start[0], c))
    else:
        step = 1 if end[0] >= start[0] else -1
        for r in range(start[0], end[0] + step, step):
            cells.append((r, start[1]))
    if any(_in_legend(cell, legend_size) for cell in cells):
        return None
    return {"cells": tuple(cells), "endpoints": (start, end)}


def _all_segment_plans(
    shape: tuple[int, int],
    legend_size: int,
    obj: dict[str, object],
) -> list[dict[str, object]]:
    merged = obj["merged"]
    plans: list[dict[str, object]] = []
    for axis in obj["axes"]:
        for sign in random.sample([-1, 1], 2):
            reaches = list(range(MAX_SEGMENT_REACH + 1))
            random.shuffle(reaches)
            for reach in reaches:
                plan = _segment_plan(shape, legend_size, merged, axis, sign, reach)
                if plan is not None:
                    plans.append(plan)
    random.shuffle(plans)
    return plans


def _assign_pairs(
    shape: tuple[int, int],
    legend_size: int,
    objects: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    targets = {tuple(obj["merged"]) for obj in objects}
    with_options = []
    for obj in objects:
        plans = _all_segment_plans(shape, legend_size, obj)
        if not plans:
            return None
        with_options.append((len(plans), obj, plans))
    with_options.sort(key=lambda item: item[0])

    reserved: set[tuple[int, int]] = set()
    assigned: list[dict[str, object]] = []
    for _count, obj, plans in with_options:
        merged = tuple(obj["merged"])
        choices = []
        for plan in plans:
            cells = set(plan["cells"])
            if cells & reserved:
                continue
            if any(cell in targets and cell != merged for cell in cells):
                continue
            choices.append(plan)
        if not choices:
            return None
        chosen = random.choice(choices)
        reserved.update(chosen["cells"])
        assigned.append(
            {
                "color": int(obj["color"]),
                "merged": merged,
                "endpoints": tuple(chosen["endpoints"]),
            }
        )
    return assigned


def _expected_pixels(
    active: dict[tuple[int, int], int],
    table: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> dict[tuple[int, int], int]:
    return _react_pixels(active, table, shape)


def _gen_attempt() -> np.ndarray | None:
    mode = _sample_mode()
    if mode == "two":
        legend_count = 4
        pair_count = 4
    else:
        legend_count = random.choice((3, 4))
        if mode == "none":
            pair_count = random.randint(2, legend_count)
        else:
            pair_count = random.randint(2, legend_count)

    legend_colors = random.sample(list(PAIR_COLOR_POOL), legend_count)
    input_colors = random.sample(legend_colors, pair_count)
    motifs = _build_motifs(input_colors, mode)
    fuse_color_pairs = [tuple(motif["colors"]) for motif in motifs if motif["kind"] == "fuse"]
    table = _build_table(legend_colors, fuse_color_pairs)
    legend = _build_legend(legend_colors, table)

    h = random.choice(GRID_SIZES)
    w = random.choice(GRID_SIZES)
    if h < legend.shape[0] + 4 or w < legend.shape[1] + 4:
        return None
    shape = (h, w)

    placement = _place_targets(shape, legend.shape[0], motifs)
    if placement is None:
        return None
    objects, active, _fuse_positions = placement

    assigned = _assign_pairs(shape, legend.shape[0], objects)
    if assigned is None:
        return None

    inp = np.full(shape, BG, dtype=int)
    inp[: legend.shape[0], : legend.shape[1]] = legend
    for obj in assigned:
        a, b = obj["endpoints"]
        inp[a] = obj["color"]
        inp[b] = obj["color"]

    expected = _render_pixels(shape, legend, _expected_pixels(active, table, shape))
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return np.ascontiguousarray(grid)
    raise RuntimeError("failed to generate a valid input for puzzle 7")
