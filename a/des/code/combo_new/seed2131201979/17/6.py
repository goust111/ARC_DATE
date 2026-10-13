from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[Coord]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: list[list[Coord]] = []
    next_id = 0

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != 0:
                continue
            next_id += 1
            q: deque[Coord] = deque([(r, c)])
            labels[r, c] = next_id
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            comps.append(comp)

    return labels, comps


def _straight_segment(coords: list[Coord]) -> tuple[str, int, int, int] | None:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})

    if len(rows) == 1:
        row = rows[0]
        c0, c1 = cols[0], cols[-1]
        if len(coords) == c1 - c0 + 1:
            return ("horizontal", row, c0, c1)

    if len(cols) == 1:
        col = cols[0]
        r0, r1 = rows[0], rows[-1]
        if len(coords) == r1 - r0 + 1:
            return ("vertical", col, r0, r1)

    return None


def _analyze_marker_candidate(g: np.ndarray, bg: int, marker: int) -> dict | None:
    h, w = g.shape
    allowed = (g == bg) | (g == marker)
    river_rows = [r for r in range(h) if bool(allowed[r].all())]
    river_cols = [c for c in range(w) if bool(allowed[:, c].all())]

    marker_mask = g == marker
    _labels, comps = _label_components(marker_mask)
    if not comps:
        return None

    bridges: list[dict] = []
    for comp in comps:
        seg = _straight_segment(comp)
        if seg is None:
            return None

        if seg[0] == "horizontal":
            row, c0, c1 = seg[1], seg[2], seg[3]
            hit_cols = [c for c in river_cols if c0 <= c <= c1]
            if row in river_rows or len(hit_cols) != 1:
                return None
            river_col = hit_cols[0]
            if not (c0 < river_col < c1):
                return None
            if c0 == 0 or c1 == w - 1:
                return None
            if int(g[row, c0 - 1]) in (bg, marker) or int(g[row, c1 + 1]) in (bg, marker):
                return None
            bridges.append(
                {
                    "orientation": "horizontal",
                    "cells": list(comp),
                    "touch_a": (row, c0 - 1),
                    "touch_b": (row, c1 + 1),
                }
            )
        else:
            col, r0, r1 = seg[1], seg[2], seg[3]
            hit_rows = [r for r in river_rows if r0 <= r <= r1]
            if col in river_cols or len(hit_rows) != 1:
                return None
            river_row = hit_rows[0]
            if not (r0 < river_row < r1):
                return None
            if r0 == 0 or r1 == h - 1:
                return None
            if int(g[r0 - 1, col]) in (bg, marker) or int(g[r1 + 1, col]) in (bg, marker):
                return None
            bridges.append(
                {
                    "orientation": "vertical",
                    "cells": list(comp),
                    "touch_a": (r0 - 1, col),
                    "touch_b": (r1 + 1, col),
                }
            )

    return {
        "marker": int(marker),
        "river_rows": river_rows,
        "river_cols": river_cols,
        "bridges": bridges,
    }


def _analyze_puzzle(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    counts = Counter(map(int, g.ravel()))
    bg = int(counts.most_common(1)[0][0])

    candidates: list[dict] = []
    for color in counts:
        color = int(color)
        if color == bg:
            continue
        info = _analyze_marker_candidate(g, bg, color)
        if info is None:
            continue

        land_mask = (g != bg) & (g != color)
        labels, comps = _label_components(land_mask)
        if not comps:
            continue

        comp_counts = [Counter(int(g[r, c]) for r, c in comp) for comp in comps]
        winners: list[int] = []
        valid = True
        for bridge in info["bridges"]:
            a = int(labels[bridge["touch_a"]])
            b = int(labels[bridge["touch_b"]])
            if a <= 0 or b <= 0 or a == b:
                valid = False
                break
            total = comp_counts[a - 1] + comp_counts[b - 1]
            total.pop(bg, None)
            total.pop(color, None)
            if not total:
                valid = False
                break
            ranked = total.most_common(2)
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                valid = False
                break
            winners.append(int(ranked[0][0]))

        if valid:
            cand = dict(info)
            cand["bg"] = bg
            cand["labels"] = labels
            cand["winners"] = winners
            candidates.append(cand)

    if not candidates:
        raise AssertionError("failed to identify a unique bridge marker color")

    return max(candidates, key=lambda info: (len(info["bridges"]), len(info["river_rows"]) + len(info["river_cols"])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each short bridge by the dominant foreground color on the two land masses it connects.

    Input structure:
    - The grid has one dominant background color.
    - One-pixel-wide background rivers span full rows or full columns, optionally
      interrupted by a special bridge-marker color.
    - The remaining non-background cells form land blocks separated by those rivers;
      each land block has one visually dominant foreground color with possible sparse noise.

    Rule:
    - Identify the bridge-marker color whose connected components are short straight
      segments crossing exactly one river and extending into land on both sides.
    - Remove those bridge pixels conceptually, then split the remaining non-background
      cells into land components.
    - For each bridge, look at the two land components touched beyond its endpoints,
      count foreground colors across their union, and repaint the whole bridge with
      the unique most common color.
    - Leave all non-bridge pixels unchanged.

    Key insight:
    - The bridge color is not intrinsic to the bridge itself; it is inferred from the
      topology of river-separated land masses and a majority vote taken on both banks.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_puzzle(g)

    out = g.copy()
    for bridge, winner in zip(info["bridges"], info["winners"]):
        for r, c in bridge["cells"]:
            out[r, c] = winner
    return out


GRID_MIN = 12
GRID_MAX = 28
MIN_BLOCK = 3
MAX_RIVERS_PER_AXIS = 3
GEN_ATTEMPTS = 400
ARC_COLORS = tuple(range(10))


def _max_internal_rivers(length: int) -> int:
    return max(0, (length - 2 - MIN_BLOCK) // (MIN_BLOCK + 1))


def _sample_internal_rivers(length: int, count: int) -> list[int] | None:
    if count == 0:
        return []
    for _ in range(80):
        picks = sorted(random.sample(range(1, length - 1), count))
        bounds = [0] + picks + [length - 1]
        if min(bounds[i + 1] - bounds[i] - 1 for i in range(len(bounds) - 1)) >= MIN_BLOCK:
            return picks
    return None


def _rect_cells(rect: tuple[int, int, int, int]) -> list[Coord]:
    r0, r1, c0, c1 = rect
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _far_from_reserved(cells: list[Coord], reserved: set[Coord], h: int, w: int) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in reserved:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in reserved and (nr, nc) not in cells:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    mode = random.choice(("row", "col", "both"))

    max_row = min(MAX_RIVERS_PER_AXIS, _max_internal_rivers(h))
    max_col = min(MAX_RIVERS_PER_AXIS, _max_internal_rivers(w))
    if max_row == 0 or max_col == 0:
        return None

    if mode == "row":
        n_row = random.randint(1, max_row)
        n_col = random.randint(0, max_col)
    elif mode == "col":
        n_row = random.randint(0, max_row)
        n_col = random.randint(1, max_col)
    else:
        n_row = random.randint(1, max_row)
        n_col = random.randint(1, max_col)

    row_rivers = _sample_internal_rivers(h, n_row)
    col_rivers = _sample_internal_rivers(w, n_col)
    if row_rivers is None or col_rivers is None:
        return None

    palette = list(ARC_COLORS)
    random.shuffle(palette)
    bg = palette.pop()
    marker = palette.pop()
    land_colors = palette[: random.randint(3, 5)]

    grid = np.full((h, w), bg, dtype=int)
    row_bounds = [0] + row_rivers + [h - 1]
    col_bounds = [0] + col_rivers + [w - 1]

    blocks: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for i in range(len(row_bounds) - 1):
        for j in range(len(col_bounds) - 1):
            r0, r1 = row_bounds[i] + 1, row_bounds[i + 1] - 1
            c0, c1 = col_bounds[j] + 1, col_bounds[j + 1] - 1
            rect = (r0, r1, c0, c1)
            blocks[(i, j)] = rect
            color = random.choice(land_colors)
            grid[r0 : r1 + 1, c0 : c1 + 1] = color

            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            noise_n = random.randint(0, min(3, area // 6))
            if noise_n > 0:
                coords = random.sample(_rect_cells(rect), noise_n)
                for rr, cc in coords:
                    alts = [c for c in land_colors if c != int(grid[rr, cc])]
                    grid[rr, cc] = random.choice(alts)

    if len(set(map(int, np.unique(grid))) - {bg}) < 3:
        return None

    reserved: set[Coord] = set()
    bridge_count = 0

    row_targets = []
    col_targets = []
    if mode in ("row", "both"):
        row_targets = random.sample(row_rivers, random.randint(1, len(row_rivers)))
    if mode in ("col", "both"):
        col_targets = random.sample(col_rivers, random.randint(1, len(col_rivers)))

    for river_row in row_targets:
        river_idx = row_bounds.index(river_row)
        placed = False
        for _ in range(120):
            strip = random.randrange(len(col_bounds) - 1)
            c0 = col_bounds[strip] + 1
            c1 = col_bounds[strip + 1] - 1
            col = random.randint(c0, c1)
            depth = random.choice((1, 2))
            cells = [(river_row + d, col) for d in range(-depth, depth + 1)]
            if not _far_from_reserved(cells, reserved, h, w):
                continue
            if any(int(grid[r, c]) == marker for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = marker
                reserved.add((r, c))
            bridge_count += 1
            placed = True
            break
        if not placed:
            return None

    for river_col in col_targets:
        river_idx = col_bounds.index(river_col)
        placed = False
        for _ in range(120):
            strip = random.randrange(len(row_bounds) - 1)
            r0 = row_bounds[strip] + 1
            r1 = row_bounds[strip + 1] - 1
            row = random.randint(r0, r1)
            depth = random.choice((1, 2))
            cells = [(row, river_col + d) for d in range(-depth, depth + 1)]
            if not _far_from_reserved(cells, reserved, h, w):
                continue
            if any(int(grid[r, c]) == marker for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = marker
                reserved.add((r, c))
            bridge_count += 1
            placed = True
            break
        if not placed:
            return None

    if bridge_count == 0:
        return None

    counts = Counter(map(int, grid.ravel()))
    other_counts = [ct for color, ct in counts.items() if int(color) != bg]
    if not other_counts or counts[bg] <= max(other_counts):
        return None

    try:
        info = _analyze_puzzle(grid)
    except AssertionError:
        return None

    if int(info["bg"]) != bg or int(info["marker"]) != marker:
        return None
    if len(info["bridges"]) != bridge_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid river-bridge puzzle input")
