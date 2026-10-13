from __future__ import annotations

import random
from itertools import permutations

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 600


def _runs_1d(arr: np.ndarray, bg: int) -> list[dict]:
    runs: list[dict] = []
    start = None
    color = None
    for idx, value in enumerate(map(int, arr)):
        if value == bg:
            if color is not None:
                runs.append(
                    {
                        "color": int(color),
                        "start": int(start),
                        "end": int(idx - 1),
                        "length": int(idx - start),
                    }
                )
                start = None
                color = None
            continue
        if color is None:
            start = idx
            color = value
            continue
        if value != color:
            runs.append(
                {
                    "color": int(color),
                    "start": int(start),
                    "end": int(idx - 1),
                    "length": int(idx - start),
                }
            )
            start = idx
            color = value
    if color is not None:
        runs.append(
            {
                "color": int(color),
                "start": int(start),
                "end": int(arr.size - 1),
                "length": int(arr.size - start),
            }
        )
    return runs


def _legend_order(grid: np.ndarray, bg: int) -> list[int]:
    row = np.asarray(grid[-1, 1:], dtype=int)
    colors = [int(x) for x in row if int(x) != bg]
    assert colors, "legend strip must contain at least two colors"
    assert row.size >= len(colors), "legend row malformed"
    assert np.all(row[: len(colors)] != bg), "legend must be a non-background prefix"
    assert np.all(row[len(colors) :] == bg), "legend tail must be background"
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return colors


def _parse_specs(grid: np.ndarray) -> tuple[int, list[int], dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 6 and w >= 7, "grid is too small for three strips and rectangles"

    bg = int(g[1, 1])
    assert int(g[0, 0]) == bg and int(g[-1, 0]) == bg, "strip intersections must be background"

    interior = g[1:-1, 1:]
    assert np.all(interior == bg), "input interior must be empty background"

    legend = _legend_order(g, bg)

    top_runs = _runs_1d(g[0, 1:], bg)
    left_runs = _runs_1d(g[1:-1, 0], bg)
    assert top_runs and left_runs

    top_map = {
        int(run["color"]): {
            "c0": int(run["start"]) + 1,
            "height": int(run["length"]),
        }
        for run in top_runs
    }
    left_map = {
        int(run["color"]): {
            "r0": int(run["start"]) + 1,
            "width": int(run["length"]),
        }
        for run in left_runs
    }

    colors = set(legend)
    assert colors == set(top_map) == set(left_map), "top, left, and legend strips must describe the same color set"

    specs: dict[int, dict] = {}
    for color in legend:
        spec = {
            "color": int(color),
            "r0": int(left_map[color]["r0"]),
            "c0": int(top_map[color]["c0"]),
            "height": int(top_map[color]["height"]),
            "width": int(left_map[color]["width"]),
        }
        r1 = spec["r0"] + spec["height"] - 1
        c1 = spec["c0"] + spec["width"] - 1
        assert r1 <= h - 2, "rectangle must stay above the bottom legend strip"
        assert c1 <= w - 1, "rectangle must stay inside the grid"
        specs[int(color)] = spec
    return bg, legend, specs


def _paint_rectangles(shape: tuple[int, int], legend: list[int], specs: dict[int, dict], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color in legend:
        spec = specs[int(color)]
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        r1 = r0 + int(spec["height"])
        c1 = c0 + int(spec["width"])
        out[r0:r1, c0:c1] = int(color)
    return out


def _boxes_overlap(a: dict, b: dict) -> bool:
    ar1 = int(a["r0"]) + int(a["height"]) - 1
    ac1 = int(a["c0"]) + int(a["width"]) - 1
    br1 = int(b["r0"]) + int(b["height"]) - 1
    bc1 = int(b["c0"]) + int(b["width"]) - 1
    return not (ar1 < int(b["r0"]) or br1 < int(a["r0"]) or ac1 < int(b["c0"]) or bc1 < int(a["c0"]))


def _painted_color_counts(canvas: np.ndarray, colors: list[int]) -> dict[int, int]:
    return {int(color): int(np.count_nonzero(canvas == int(color))) for color in colors}


def _literal_misread(shape: tuple[int, int], legend: list[int], specs: dict[int, dict], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color in legend:
        spec = specs[int(color)]
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        r1 = r0 + int(spec["width"])
        c1 = c0 + int(spec["height"])
        out[r0:r1, c0:c1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild overlapping solid rectangles from top and left code strips, then paint them in bottom-legend order.

    Input structure:
    - The top row and left column are code strips; each foreground color appears in one contiguous top segment and one contiguous left segment.
    - The bottom row, from column 1 onward, is a legend strip listing the same colors once each in drawing order; the two strip-intersection corner cells stay background.
    - Every other cell in the interior is background.

    Rule:
    - For each color, take the top segment's start column as the rectangle's left edge and the top segment's length as the rectangle's height.
    - Take the left segment's start row as the rectangle's top edge and the left segment's length as the rectangle's width.
    - Draw the solid rectangles into the interior from left to right along the bottom legend strip; later rectangles overwrite earlier ones.
    - Preserve the three strips exactly.

    Key insight:
    - Neither strip is sufficient by itself: the top strip supplies horizontal anchoring plus vertical extent, the left strip supplies vertical anchoring plus horizontal extent, and only the bottom legend resolves overlap ownership.
    """
    g = np.asarray(grid, dtype=int)
    bg, legend, specs = _parse_specs(g)

    out = g.copy()
    interior = _paint_rectangles(g.shape, legend, specs, bg)
    out[1:-1, 1:] = interior[1:-1, 1:]
    return out


def _sample_rectangles(h: int, w: int, colors: list[int]) -> dict[int, dict] | None:
    specs: dict[int, dict] = {}
    top_runs: list[tuple[int, int]] = []
    left_runs: list[tuple[int, int]] = []
    order = colors[:]
    random.shuffle(order)

    for color in order:
        for _ in range(100):
            r0 = random.randint(1, h - 3)
            c0 = random.randint(1, w - 2)
            side_limit = min(h - 1 - r0, w - c0, 6)
            if side_limit < 2:
                continue
            height = random.randint(2, side_limit)
            width = random.randint(2, side_limit)
            top_run = (c0, c0 + height - 1)
            left_run = (r0, r0 + width - 1)
            if any(not (top_run[1] < lo or hi < top_run[0]) for lo, hi in top_runs):
                continue
            if any(not (left_run[1] < lo or hi < left_run[0]) for lo, hi in left_runs):
                continue
            specs[int(color)] = {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "height": int(height),
                "width": int(width),
            }
            top_runs.append(top_run)
            left_runs.append(left_run)
            break
        else:
            return None
    return specs


def _semantic_checks(shape: tuple[int, int], legend: list[int], specs: dict[int, dict], bg: int) -> bool:
    correct = _paint_rectangles(shape, legend, specs, bg)
    colors = [int(x) for x in legend]

    if not any(_boxes_overlap(specs[a], specs[b]) for i, a in enumerate(colors) for b in colors[i + 1 :]):
        return False

    if len(set(tuple(p) for p in permutations(colors))) > 1:
        changed = False
        for alt in (list(reversed(colors)), sorted(colors), sorted(colors, reverse=True)):
            if alt == colors:
                continue
            if not np.array_equal(correct, _paint_rectangles(shape, alt, specs, bg)):
                changed = True
                break
        if not changed:
            return False

    if np.array_equal(correct, _literal_misread(shape, legend, specs, bg)):
        return False

    counts = _painted_color_counts(correct, colors)
    if any(counts[color] == 0 for color in colors):
        return False

    union = np.zeros(shape, dtype=bool)
    overlap_cells = np.zeros(shape, dtype=int)
    for color in colors:
        spec = specs[int(color)]
        mask = np.zeros(shape, dtype=bool)
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        r1 = r0 + int(spec["height"])
        c1 = c0 + int(spec["width"])
        mask[r0:r1, c0:c1] = True
        overlap_cells += mask.astype(int)
        union |= mask
    overlap_area = int(np.count_nonzero(overlap_cells >= 2))
    union_area = int(np.count_nonzero(union))
    if overlap_area == 0 or overlap_area * 3 > union_area:
        return False

    top_order = [color for color, _spec in sorted(specs.items(), key=lambda kv: (int(kv[1]["c0"]), int(kv[0])))]
    left_order = [color for color, _spec in sorted(specs.items(), key=lambda kv: (int(kv[1]["r0"]), int(kv[0])))]
    if legend == top_order or legend == left_order:
        return False
    if top_order == left_order:
        return False

    return True


def _build_input(shape: tuple[int, int], legend: list[int], specs: dict[int, dict], bg: int) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((h, w), bg, dtype=int)
    grid[0, 0] = bg
    grid[-1, 0] = bg
    for color, spec in specs.items():
        c0 = int(spec["c0"])
        r0 = int(spec["r0"])
        grid[0, c0 : c0 + int(spec["height"])] = int(color)
        grid[r0 : r0 + int(spec["width"]), 0] = int(color)
    for idx, color in enumerate(legend, start=1):
        grid[-1, idx] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(2, 4)
    h = random.randint(max(8, color_count + 5), 14)
    w = random.randint(max(9, color_count + 6), 15)
    colors = random.sample(PALETTE, color_count)

    for _ in range(250):
        specs = _sample_rectangles(h, w, colors)
        if specs is None:
            continue

        top_order = [color for color, _spec in sorted(specs.items(), key=lambda kv: (int(kv[1]["c0"]), int(kv[0])))]
        left_order = [color for color, _spec in sorted(specs.items(), key=lambda kv: (int(kv[1]["r0"]), int(kv[0])))]
        legend_options = [
            list(p)
            for p in permutations(colors)
            if list(p) != top_order and list(p) != left_order
        ]
        if not legend_options:
            continue
        legend = random.choice(legend_options)

        if not _semantic_checks((h, w), legend, specs, BG):
            continue

        inp = _build_input((h, w), legend, specs, BG)
        try:
            out = main(inp)
        except AssertionError:
            continue

        expected = inp.copy()
        painted = _paint_rectangles(inp.shape, legend, specs, BG)
        expected[1:-1, 1:] = painted[1:-1, 1:]
        if not np.array_equal(out, expected):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection-rebuild sample")
