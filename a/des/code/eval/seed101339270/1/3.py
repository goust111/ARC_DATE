from __future__ import annotations

from collections import Counter
import random

import numpy as np


YELLOW = 4
SPLIT_COLOR = 5
MARKER_COLOR = 9
ALL_COLORS = tuple(range(10))
GEN_RETRIES = 400
SCENARIO_RETRIES = 200


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _quadrant_bounds(h: int, w: int, split_r: int, split_c: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "tl": (0, split_r - 1, 0, split_c - 1),
        "tr": (0, split_r - 1, split_c + 1, w - 1),
        "bl": (split_r + 1, h - 1, 0, split_c - 1),
        "br": (split_r + 1, h - 1, split_c + 1, w - 1),
    }


def _seed_footprint(r: int, c: int, bounds: tuple[int, int, int, int], length: int) -> set[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    cells = {(int(r), int(c))}
    for step in range(1, int(length) + 1):
        if r - step >= r0:
            cells.add((int(r - step), int(c)))
        if r + step <= r1:
            cells.add((int(r + step), int(c)))
        if c - step >= c0:
            cells.add((int(r), int(c - step)))
        if c + step <= c1:
            cells.add((int(r), int(c + step)))
    return cells


def _corner_component_size(grid: np.ndarray, start: tuple[int, int], color: int) -> int:
    h, w = grid.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in seen and int(grid[rr, cc]) == int(color):
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen)


def _find_split(grid: np.ndarray, bg: int, marker_color: int) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in {int(bg), int(marker_color)}:
            continue
        rows = [r for r in range(1, h - 1) if np.all(grid[r, :] == color)]
        cols = [c for c in range(1, w - 1) if np.all(grid[:, c] == color)]
        if not rows or not cols:
            continue
        best_row = min(rows, key=lambda r: abs(r - h // 2))
        best_col = min(cols, key=lambda c: abs(c - w // 2))
        score = abs(best_row - h // 2) + abs(best_col - w // 2)
        candidates.append((score, color, best_row, best_col))

    assert candidates, "expected one split-line color with a full row and full column"
    candidates.sort()
    _score, split_color, split_r, split_c = candidates[0]
    return int(split_color), int(split_r), int(split_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed into a same-color cross whose arm length is encoded by the marker
    in that seed's quadrant; different-color overlaps turn yellow.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    corner_colors = [int(g[0, 0]), int(g[0, w - 1]), int(g[h - 1, 0]), int(g[h - 1, w - 1])]
    marker_color = int(Counter(corner_colors).most_common(1)[0][0])
    assert all(color == marker_color for color in corner_colors), "all four corner markers must share one color"

    bg = _most_common_color(g)
    split_color, split_r, split_c = _find_split(g, bg, marker_color)
    quadrants = _quadrant_bounds(h, w, split_r, split_c)
    lengths = {
        "tl": _corner_component_size(g, (0, 0), marker_color),
        "tr": _corner_component_size(g, (0, w - 1), marker_color),
        "bl": _corner_component_size(g, (h - 1, 0), marker_color),
        "br": _corner_component_size(g, (h - 1, w - 1), marker_color),
    }

    claims = [[set() for _ in range(w)] for _ in range(h)]
    ignored_colors = {int(bg), int(split_color), int(marker_color)}

    for name, bounds in quadrants.items():
        length = int(lengths[name])
        r0, r1, c0, c1 = bounds
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                color = int(g[r, c])
                if color in ignored_colors:
                    continue
                for rr, cc in _seed_footprint(r, c, bounds, length):
                    claims[rr][cc].add(color)

    out = g.copy()
    for r in range(h):
        for c in range(w):
            if r == split_r or c == split_c:
                continue
            if int(g[r, c]) == int(marker_color):
                continue
            if not claims[r][c]:
                continue
            out[r, c] = YELLOW if len(claims[r][c]) >= 2 else next(iter(claims[r][c]))

    return out


def _marker_cells(h: int, w: int, corner: str, length: int, orient: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(0, c) for c in range(length)] if orient == "h" else [(r, 0) for r in range(length)]
    if corner == "tr":
        return [(0, c) for c in range(w - length, w)] if orient == "h" else [(r, w - 1) for r in range(length)]
    if corner == "bl":
        return [(h - 1, c) for c in range(length)] if orient == "h" else [(r, 0) for r in range(h - length, h)]
    return [(h - 1, c) for c in range(w - length, w)] if orient == "h" else [(r, w - 1) for r in range(h - length, h)]


def _valid_seed_cell(
    pos: tuple[int, int],
    bounds: tuple[int, int, int, int],
    length: int,
    blocked: set[tuple[int, int]],
) -> bool:
    if pos in blocked:
        return False
    return _seed_footprint(pos[0], pos[1], bounds, length).isdisjoint(blocked)


def _place_diff_pair(
    bounds: tuple[int, int, int, int],
    length: int,
    palette: list[int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int, int]] | None:
    r0, r1, c0, c1 = bounds
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if _valid_seed_cell((r, c), bounds, length, blocked)]
    if len(cells) < 2:
        return None

    for _ in range(SCENARIO_RETRIES):
        (r1s, c1s), (r2s, c2s) = random.sample(cells, 2)
        if abs(r1s - r2s) == 0 or abs(c1s - c2s) == 0:
            continue
        if abs(r1s - r2s) > length or abs(c1s - c2s) > length:
            continue
        color1, color2 = random.sample(palette, 2)
        fp1 = _seed_footprint(r1s, c1s, bounds, length)
        fp2 = _seed_footprint(r2s, c2s, bounds, length)
        inter = fp1 & fp2
        if not any(cell not in {(r1s, c1s), (r2s, c2s)} for cell in inter):
            continue
        return [(r1s, c1s, color1), (r2s, c2s, color2)]
    return None


def _place_same_pair(
    bounds: tuple[int, int, int, int],
    length: int,
    palette: list[int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int, int]] | None:
    r0, r1, c0, c1 = bounds
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if _valid_seed_cell((r, c), bounds, length, blocked)]
    if len(cells) < 2:
        return None

    for _ in range(SCENARIO_RETRIES):
        same_color = random.choice(palette)
        orient = random.choice(("h", "v"))
        if orient == "h":
            row = random.randint(r0, r1)
            row_cells = [cell for cell in cells if cell[0] == row]
            if len(row_cells) < 2:
                continue
            (ra, ca), (rb, cb) = random.sample(row_cells, 2)
            if abs(ca - cb) > 2 * length:
                continue
        else:
            col = random.randint(c0, c1)
            col_cells = [cell for cell in cells if cell[1] == col]
            if len(col_cells) < 2:
                continue
            (ra, ca), (rb, cb) = random.sample(col_cells, 2)
            if abs(ra - rb) > 2 * length:
                continue
        fp1 = _seed_footprint(ra, ca, bounds, length)
        fp2 = _seed_footprint(rb, cb, bounds, length)
        inter = fp1 & fp2
        if not any(cell not in {(ra, ca), (rb, cb)} for cell in inter):
            continue
        return [(ra, ca, same_color), (rb, cb, same_color)]
    return None


def _place_boundary_seed(
    bounds: tuple[int, int, int, int],
    length: int,
    palette: list[int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int, int]] | None:
    r0, r1, c0, c1 = bounds
    candidates = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if not _valid_seed_cell((r, c), bounds, length, blocked):
                continue
            if min(r - r0, r1 - r, c - c0, c1 - c) < length:
                candidates.append((r, c))
    if not candidates:
        return None

    pos = random.choice(candidates)
    seeds = [(pos[0], pos[1], random.choice(palette))]
    if random.random() < 0.5:
        base_fp = _seed_footprint(pos[0], pos[1], bounds, length)
        extras = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) == pos or not _valid_seed_cell((r, c), bounds, length, blocked):
                    continue
                fp = _seed_footprint(r, c, bounds, length)
                if fp.isdisjoint(base_fp):
                    extras.append((r, c))
        if extras:
            rr, cc = random.choice(extras)
            extra_color = random.choice([color for color in palette if color != seeds[0][2]])
            seeds.append((rr, cc, extra_color))
    return seeds


def _place_random_seeds(
    bounds: tuple[int, int, int, int],
    length: int,
    palette: list[int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int, int]] | None:
    r0, r1, c0, c1 = bounds
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if _valid_seed_cell((r, c), bounds, length, blocked)]
    if not cells:
        return None

    random.shuffle(cells)
    seeds: list[tuple[int, int, int]] = []
    footprints: list[set[tuple[int, int]]] = []
    target = random.randint(1, 2)

    for r, c in cells:
        fp = _seed_footprint(r, c, bounds, length)
        if any(not fp.isdisjoint(prev) for prev in footprints):
            continue
        used_colors = {color for _rr, _cc, color in seeds}
        choices = [color for color in palette if color not in used_colors] or palette[:]
        seeds.append((r, c, random.choice(choices)))
        footprints.append(fp)
        if len(seeds) == target:
            break

    return seeds or None


def _gen_attempt() -> np.ndarray | None:
    h = random.choice((13, 15, 17, 19))
    w = random.choice((13, 15, 17, 19))
    split_r = h // 2
    split_c = w // 2

    bg_choices = [color for color in ALL_COLORS if color not in {YELLOW, SPLIT_COLOR, MARKER_COLOR}]
    bg = random.choice(bg_choices)
    grid = np.full((h, w), bg, dtype=int)
    grid[split_r, :] = SPLIT_COLOR
    grid[:, split_c] = SPLIT_COLOR

    quadrants = _quadrant_bounds(h, w, split_r, split_c)
    blocked: set[tuple[int, int]] = {(split_r, c) for c in range(w)} | {(r, split_c) for r in range(h)}
    lengths: dict[str, int] = {}

    for name in ("tl", "tr", "bl", "br"):
        length = random.randint(1, 4)
        orient = random.choice(("h", "v"))
        lengths[name] = length
        for r, c in _marker_cells(h, w, name, length, orient):
            grid[r, c] = MARKER_COLOR
            blocked.add((r, c))

    palette = [color for color in ALL_COLORS if color not in {bg, YELLOW, SPLIT_COLOR, MARKER_COLOR}]
    random.shuffle(palette)

    scenario_order = ["diff", "same", "boundary", "random"]
    random.shuffle(scenario_order)
    all_seeds: list[tuple[int, int, int]] = []

    for name, scenario in zip(("tl", "tr", "bl", "br"), scenario_order):
        bounds = quadrants[name]
        length = lengths[name]
        if scenario == "diff":
            seeds = _place_diff_pair(bounds, length, palette, blocked)
        elif scenario == "same":
            seeds = _place_same_pair(bounds, length, palette, blocked)
        elif scenario == "boundary":
            seeds = _place_boundary_seed(bounds, length, palette, blocked)
        else:
            seeds = _place_random_seeds(bounds, length, palette, blocked)
        if seeds is None or not seeds:
            return None
        all_seeds.extend(seeds)

    for r, c, color in all_seeds:
        if (r, c) in blocked:
            return None
        grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out == YELLOW):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Build a random four-quadrant seed-growth puzzle with both same-color and
    different-color overlaps represented across the sample set.
    """
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
