import random
from collections import Counter

import numpy as np


BLACK = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(1, 10))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _separator_color(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    counts = Counter()
    for color in np.unique(g):
        color = int(color)
        if color == BLACK:
            continue
        row_hits = int(np.sum(np.all(g == color, axis=1)))
        col_hits = int(np.sum(np.all(g == color, axis=0)))
        if row_hits or col_hits:
            counts[color] = row_hits + col_hits
    assert counts, "expected a separator color with full rows/cols"
    return max(counts.items(), key=lambda kv: (kv[1], np.sum(g == kv[0])))[0]


def _regions_from_separator(grid: np.ndarray, sep: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    sep_rows = [int(i) for i in np.flatnonzero(np.all(g == sep, axis=1))]
    sep_cols = [int(i) for i in np.flatnonzero(np.all(g == sep, axis=0))]
    assert len(sep_rows) >= 2 and len(sep_cols) >= 2

    regions: list[dict] = []
    for top, bottom in zip(sep_rows, sep_rows[1:]):
        if bottom - top <= 1:
            continue
        for left, right in zip(sep_cols, sep_cols[1:]):
            if right - left <= 1:
                continue
            regions.append(
                {
                    "r0": top + 1,
                    "r1": bottom - 1,
                    "c0": left + 1,
                    "c1": right - 1,
                    "top_sep": top,
                    "bottom_sep": bottom,
                    "left_sep": left,
                    "right_sep": right,
                }
            )
    assert regions, "expected at least one enclosed region"
    return regions


def _find_region(regions: list[dict], cell: tuple[int, int]) -> dict:
    r, c = cell
    hits = [reg for reg in regions if reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"]]
    assert len(hits) == 1, "each marker stem must lie in exactly one region"
    return hits[0]


def _parse_t_marker(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    if len(comp) != 3:
        return None
    colors = Counter(int(grid[r, c]) for r, c in comp)
    if sorted(colors.values()) != [1, 2]:
        return None
    bar_color = next(color for color, cnt in colors.items() if cnt == 2)
    stem_color = next(color for color, cnt in colors.items() if cnt == 1)
    bar = sorted((r, c) for r, c in comp if int(grid[r, c]) == bar_color)
    stem = next((r, c) for r, c in comp if int(grid[r, c]) == stem_color)

    if bar[0][0] == bar[1][0] and abs(bar[0][1] - bar[1][1]) == 1:
        row = bar[0][0]
        if stem[0] == row + 1 and stem[1] in {bar[0][1], bar[1][1]}:
            direction = "down"
        elif stem[0] == row - 1 and stem[1] in {bar[0][1], bar[1][1]}:
            direction = "up"
        else:
            return None
        return {
            "bar_color": bar_color,
            "stem_color": stem_color,
            "bar_cells": bar,
            "stem_cell": stem,
            "direction": direction,
            "start": min(c for _, c in bar),
        }

    if bar[0][1] == bar[1][1] and abs(bar[0][0] - bar[1][0]) == 1:
        col = bar[0][1]
        if stem[1] == col + 1 and stem[0] in {bar[0][0], bar[1][0]}:
            direction = "right"
        elif stem[1] == col - 1 and stem[0] in {bar[0][0], bar[1][0]}:
            direction = "left"
        else:
            return None
        return {
            "bar_color": bar_color,
            "stem_color": stem_color,
            "bar_cells": bar,
            "stem_cell": stem,
            "direction": direction,
            "start": min(r for r, _ in bar),
        }

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each stem-selected rectangle and mark the opposite boundary."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep = _separator_color(g)
    regions = _regions_from_separator(g, sep)

    mask = (g != BLACK) & (g != sep)
    markers = []
    for comp in _components(mask):
        marker = _parse_t_marker(g, comp)
        if marker is not None:
            markers.append(marker)
    assert markers, "expected at least one T marker"

    out = g.copy()
    for marker in markers:
        reg = _find_region(regions, marker["stem_cell"])

        out[reg["r0"] : reg["r1"] + 1, reg["c0"] : reg["c1"] + 1][
            out[reg["r0"] : reg["r1"] + 1, reg["c0"] : reg["c1"] + 1] == BLACK
        ] = marker["bar_color"]

        if marker["direction"] == "down":
            out[reg["bottom_sep"], marker["start"] : marker["start"] + 2] = marker["stem_color"]
        elif marker["direction"] == "up":
            out[reg["top_sep"], marker["start"] : marker["start"] + 2] = marker["stem_color"]
        elif marker["direction"] == "right":
            out[marker["start"] : marker["start"] + 2, reg["right_sep"]] = marker["stem_color"]
        else:
            out[marker["start"] : marker["start"] + 2, reg["left_sep"]] = marker["stem_color"]

    return out


def _split_lengths(rng: random.Random, parts: int, lo: int, hi: int) -> list[int]:
    return [rng.randint(lo, hi) for _ in range(parts)]


def _build_grid(row_heights: list[int], col_widths: list[int], sep: int) -> tuple[np.ndarray, list[dict]]:
    h = sum(row_heights) + len(row_heights) + 1
    w = sum(col_widths) + len(col_widths) + 1
    grid = np.full((h, w), BLACK, dtype=int)

    sep_rows = [0]
    cur = 0
    for height in row_heights:
        cur += height + 1
        sep_rows.append(cur)
    sep_cols = [0]
    cur = 0
    for width in col_widths:
        cur += width + 1
        sep_cols.append(cur)

    grid[sep_rows, :] = sep
    grid[:, sep_cols] = sep
    regions = _regions_from_separator(grid, sep)
    return grid, regions


def _place_marker(
    grid: np.ndarray,
    reg: dict,
    direction: str,
    bar_color: int,
    stem_color: int,
    rng: random.Random,
) -> tuple[tuple[int, int], ...]:
    if direction == "down":
        start = rng.randint(reg["c0"], reg["c1"] - 1)
        attach_right = rng.choice([False, True])
        grid[reg["r0"], start : start + 2] = bar_color
        grid[reg["r0"] + 1, start + int(attach_right)] = stem_color
        return ((reg["bottom_sep"], start), (reg["bottom_sep"], start + 1))

    if direction == "up":
        start = rng.randint(reg["c0"], reg["c1"] - 1)
        attach_right = rng.choice([False, True])
        grid[reg["r1"], start : start + 2] = bar_color
        grid[reg["r1"] - 1, start + int(attach_right)] = stem_color
        return ((reg["top_sep"], start), (reg["top_sep"], start + 1))

    if direction == "right":
        start = rng.randint(reg["r0"], reg["r1"] - 1)
        attach_bottom = rng.choice([False, True])
        grid[start : start + 2, reg["c0"]] = bar_color
        grid[start + int(attach_bottom), reg["c0"] + 1] = stem_color
        return ((start, reg["right_sep"]), (start + 1, reg["right_sep"]))

    start = rng.randint(reg["r0"], reg["r1"] - 1)
    attach_bottom = rng.choice([False, True])
    grid[start : start + 2, reg["c1"]] = bar_color
    grid[start + int(attach_bottom), reg["c1"] - 1] = stem_color
    return ((start, reg["left_sep"]), (start + 1, reg["left_sep"]))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_rows = rng.randint(2, 4)
    region_cols = rng.randint(2, 4)
    row_heights = _split_lengths(rng, region_rows, 4, 7)
    col_widths = _split_lengths(rng, region_cols, 4, 7)

    sep = rng.choice(PALETTE)
    grid, regions = _build_grid(row_heights, col_widths, sep)

    available_regions = regions[:]
    rng.shuffle(available_regions)
    marker_count = rng.randint(1, min(4, len(available_regions)))
    chosen_regions = available_regions[:marker_count]

    reserved_output: set[tuple[int, int]] = set()
    dirs = ["up", "down", "left", "right"]
    rng.shuffle(dirs)

    colors = [c for c in PALETTE if c != sep]
    for i, reg in enumerate(chosen_regions):
        if i < 4:
            direction = dirs[i]
        else:
            direction = rng.choice(dirs)

        feasible = []
        if direction in {"up", "down"} and reg["c1"] - reg["c0"] + 1 >= 2:
            starts = list(range(reg["c0"], reg["c1"]))
            rng.shuffle(starts)
            for start in starts:
                if direction == "down":
                    target = {(reg["bottom_sep"], start), (reg["bottom_sep"], start + 1)}
                else:
                    target = {(reg["top_sep"], start), (reg["top_sep"], start + 1)}
                if not (target & reserved_output):
                    feasible.append((direction, start))
        if direction in {"left", "right"} and reg["r1"] - reg["r0"] + 1 >= 2:
            starts = list(range(reg["r0"], reg["r1"]))
            rng.shuffle(starts)
            for start in starts:
                if direction == "right":
                    target = {(start, reg["right_sep"]), (start + 1, reg["right_sep"])}
                else:
                    target = {(start, reg["left_sep"]), (start + 1, reg["left_sep"])}
                if not (target & reserved_output):
                    feasible.append((direction, start))
        if not feasible:
            return None

        direction, forced_start = feasible[0]
        rng.shuffle(colors)
        bar_color, stem_color = colors[:2]

        if direction == "down":
            attach_right = rng.choice([False, True])
            grid[reg["r0"], forced_start : forced_start + 2] = bar_color
            grid[reg["r0"] + 1, forced_start + int(attach_right)] = stem_color
            reserved_output |= {(reg["bottom_sep"], forced_start), (reg["bottom_sep"], forced_start + 1)}
        elif direction == "up":
            attach_right = rng.choice([False, True])
            grid[reg["r1"], forced_start : forced_start + 2] = bar_color
            grid[reg["r1"] - 1, forced_start + int(attach_right)] = stem_color
            reserved_output |= {(reg["top_sep"], forced_start), (reg["top_sep"], forced_start + 1)}
        elif direction == "right":
            attach_bottom = rng.choice([False, True])
            grid[forced_start : forced_start + 2, reg["c0"]] = bar_color
            grid[forced_start + int(attach_bottom), reg["c0"] + 1] = stem_color
            reserved_output |= {(forced_start, reg["right_sep"]), (forced_start + 1, reg["right_sep"])}
        else:
            attach_bottom = rng.choice([False, True])
            grid[forced_start : forced_start + 2, reg["c1"]] = bar_color
            grid[forced_start + int(attach_bottom), reg["c1"] - 1] = stem_color
            reserved_output |= {(forced_start, reg["left_sep"]), (forced_start + 1, reg["left_sep"])}

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
