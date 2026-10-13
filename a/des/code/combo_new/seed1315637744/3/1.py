import random
from math import ceil

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = ("up", "down", "left", "right")


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def _is_block_2x2(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    if len(coords) != 4:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return _normalize(coords) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def _arrow_direction(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> str | None:
    pts = set(coords)
    if len(pts) != 4:
        return None

    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})

    for r in rows:
        row_cols = sorted(c for rr, c in pts if rr == r)
        if len(row_cols) != 3:
            continue
        if row_cols[1] != row_cols[0] + 1 or row_cols[2] != row_cols[1] + 1:
            continue
        extras = [(rr, cc) for rr, cc in pts if rr != r]
        if len(extras) != 1:
            continue
        er, ec = extras[0]
        if abs(er - r) == 1 and ec == row_cols[0]:
            return "left"
        if abs(er - r) == 1 and ec == row_cols[2]:
            return "right"

    for c in cols:
        col_rows = sorted(r for r, cc in pts if cc == c)
        if len(col_rows) != 3:
            continue
        if col_rows[1] != col_rows[0] + 1 or col_rows[2] != col_rows[1] + 1:
            continue
        extras = [(rr, cc) for rr, cc in pts if cc != c]
        if len(extras) != 1:
            continue
        er, ec = extras[0]
        if abs(ec - c) == 1 and er == col_rows[0]:
            return "up"
        if abs(ec - c) == 1 and er == col_rows[2]:
            return "down"

    return None


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    dc = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return max(dr, dc)


def _legend_map(grid: np.ndarray, bg: int) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        comps = _components(grid == color)
        blocks = [comp for comp in comps if _is_block_2x2(comp)]
        arrows = [(comp, _arrow_direction(comp)) for comp in comps]
        arrows = [(comp, direction) for comp, direction in arrows if direction is not None]

        best: tuple[tuple[int, int], str] | None = None
        for block in blocks:
            bb = _bbox(block)
            for arrow, direction in arrows:
                ab = _bbox(arrow)
                gap = _bbox_gap(bb, ab)
                union_h = max(bb[2], ab[2]) - min(bb[0], ab[0]) + 1
                union_w = max(bb[3], ab[3]) - min(bb[1], ab[1]) + 1
                if gap > 2 or union_h > 6 or union_w > 6:
                    continue
                score = (gap, union_h * union_w)
                if best is None or score < best[0]:
                    best = (score, direction)
        if best is not None:
            mapping[color] = best[1]
    assert mapping, "expected at least one legend color"
    return mapping


def _side_status(line: list[bool]) -> tuple[str, int | None, int | None] | None:
    if not line[0] or not line[-1]:
        return None
    interior = line[1:-1]
    missing = [i for i, value in enumerate(interior) if not value]
    if not missing:
        return ("full", None, None)
    if len(missing) < 2:
        return None
    if missing != list(range(missing[0], missing[-1] + 1)):
        return None
    return ("gap", missing[0] + 1, missing[-1] + 1)


def _frame_info(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    pts = set(coords)
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in pts):
        return None

    top = [(r0, c) in pts for c in range(c0, c1 + 1)]
    bottom = [(r1, c) in pts for c in range(c0, c1 + 1)]
    left = [(r, c0) in pts for r in range(r0, r1 + 1)]
    right = [(r, c1) in pts for r in range(r0, r1 + 1)]

    states = {
        "up": _side_status(top),
        "down": _side_status(bottom),
        "left": _side_status(left),
        "right": _side_status(right),
    }
    if any(state is None for state in states.values()):
        return None

    gap_sides = [(side, state[1], state[2]) for side, state in states.items() if state[0] == "gap"]
    if len(gap_sides) != 1:
        return None

    direction, gap0, gap1 = gap_sides[0]
    assert gap0 is not None and gap1 is not None
    gap_len = gap1 - gap0 + 1
    expected_area = 2 * h + 2 * w - 4 - gap_len
    if len(pts) != expected_area:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "direction": direction,
        "gap_span": (gap0, gap1),
        "cells": pts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the interior pattern color as a legend key, then push each framed pattern outward through the matching gap.

    Input structure:
    - The background is black.
    - The grid contains several one-pixel-thick rectangular borders, each in a unique color, and each border has exactly one continuous gap on one side.
    - Inside every border there is one connected non-background sprite whose color differs from the border color.
    - A corner legend contains, for each sprite color, one solid color block and one same-color arrow made from a 3-pixel line plus a 1-pixel tip.

    Rule:
    - Read each interior sprite color and find the legend arrow with the same color.
    - The arrow direction tells which side gap belongs to that frame.
    - Clear the sprite from the frame interior and translate it rigidly through that side so the moved sprite sits fully outside the frame, with its nearest edge one cell beyond the frame exterior.
    - Keep every border, legend block, and legend arrow unchanged.

    Key insight:
    - The interior sprite color is not the thing to move toward visually by itself; it is the lookup key for a color-to-direction legend, and that legend determines which gap is the sprite's exit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    legend = _legend_map(g, bg)

    frames: list[dict[str, object]] = []
    for color0 in np.unique(g):
        color = int(color0)
        if color == bg:
            continue
        for comp in _components(g == color):
            info = _frame_info(comp)
            if info is not None:
                info["frame_color"] = color
                frames.append(info)
    assert frames, "expected at least one frame"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[assignment]
        direction = frame["direction"]  # type: ignore[assignment]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = set(map(int, np.unique(interior))) - {bg}
        assert len(colors) == 1, "frame interior must contain exactly one sprite color"
        sprite_color = colors.pop()
        assert sprite_color in legend, "every sprite color must appear in the legend"
        assert legend[sprite_color] == direction, "frame gap must agree with the legend"

        sprite_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if int(g[r, c]) == sprite_color]
        assert sprite_cells, "sprite must be non-empty"

        rows = [r for r, _ in sprite_cells]
        cols = [c for _, c in sprite_cells]
        if direction == "up":
            dr = (r0 - 1) - max(rows)
            dc = 0
        elif direction == "down":
            dr = (r1 + 1) - min(rows)
            dc = 0
        elif direction == "left":
            dr = 0
            dc = (c0 - 1) - max(cols)
        else:
            dr = 0
            dc = (c1 + 1) - min(cols)

        moved = [(r + dr, c + dc) for r, c in sprite_cells]
        for r, c in sprite_cells:
            out[r, c] = bg
        for r, c in moved:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "moved sprite must stay in bounds"
            assert int(g[r, c]) == bg, "moved sprite must land on background"
            out[r, c] = sprite_color

    return out


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
FRAME_COUNT_RANGE = (2, 4)
INTERIOR_H_RANGE = (4, 8)
INTERIOR_W_RANGE = (4, 8)
SPRITE_AREA_RANGE = (2, 6)
LEGEND_ITEM_H = 3
LEGEND_ITEM_W = 6
LEGEND_GAP = 1
PLACEMENT_TRIES = 150
GEN_ATTEMPTS = 400


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(max(0, r - pad), min(h, r + pad + 1)):
            for nc in range(max(0, c - pad), min(w, c + pad + 1)):
                expanded.add((nr, nc))
    return expanded


def _arrow_cells(direction: str, variant: int) -> set[tuple[int, int]]:
    if direction == "right":
        return {(1, 0), (1, 1), (1, 2), (0 if variant == 0 else 2, 2)}
    if direction == "left":
        return {(1, 0), (1, 1), (1, 2), (0 if variant == 0 else 2, 0)}
    if direction == "up":
        return {(0, 1), (1, 1), (2, 1), (0, 0 if variant == 0 else 2)}
    return {(0, 1), (1, 1), (2, 1), (2, 0 if variant == 0 else 2)}


def _legend_cluster_dims(count: int, ncols: int) -> tuple[int, int]:
    nrows = ceil(count / ncols)
    height = nrows * LEGEND_ITEM_H + (nrows - 1) * LEGEND_GAP
    width = ncols * LEGEND_ITEM_W + (ncols - 1) * LEGEND_GAP
    return height, width


def _legend_cells(
    colors: list[int],
    directions: dict[int, str],
    shape: tuple[int, int],
    rng: random.Random,
) -> dict[tuple[int, int], int] | None:
    h, w = shape
    count = len(colors)
    ncols = 2 if count >= 3 and rng.random() < 0.75 else 1
    ncols = min(ncols, count)
    cluster_h, cluster_w = _legend_cluster_dims(count, ncols)
    if cluster_h + 2 > h or cluster_w + 2 > w:
        return None

    vertical = rng.choice(["top", "bottom"])
    horizontal = rng.choice(["left", "right"])
    base_r = 1 if vertical == "top" else h - 1 - cluster_h
    base_c = 1 if horizontal == "left" else w - 1 - cluster_w

    cells: dict[tuple[int, int], int] = {}
    ordered = colors[:]
    rng.shuffle(ordered)
    for idx, color in enumerate(ordered):
        item_row = idx // ncols
        item_col = idx % ncols
        r0 = base_r + item_row * (LEGEND_ITEM_H + LEGEND_GAP)
        c0 = base_c + item_col * (LEGEND_ITEM_W + LEGEND_GAP)

        if horizontal == "left":
            block_origin = (r0, c0)
            arrow_origin = (r0, c0 + 3)
        else:
            arrow_origin = (r0, c0)
            block_origin = (r0, c0 + 4)

        for dr in range(2):
            for dc in range(2):
                cells[(block_origin[0] + dr, block_origin[1] + dc)] = color
        variant = rng.randrange(2)
        for dr, dc in _arrow_cells(directions[color], variant):
            cells[(arrow_origin[0] + dr, arrow_origin[1] + dc)] = color

    return cells


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_sprite(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(1, 3)
        w = rng.randint(1, 3)
        max_area = min(SPRITE_AREA_RANGE[1], h * w)
        if max_area < SPRITE_AREA_RANGE[0]:
            continue
        area = rng.randint(SPRITE_AREA_RANGE[0], max_area)
        seed = (rng.randrange(h), rng.randrange(w))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        if _is_block_2x2(coords):
            continue
        if _arrow_direction(coords) is not None:
            continue
        return mask
    raise RuntimeError("failed to build sprite")


def _choose_gap(span0: int, span1: int, limit: int, rng: random.Random) -> tuple[int, int]:
    base_len = span1 - span0 + 1
    max_len = min(limit, base_len + 2)
    gap_len = rng.randint(max(2, base_len), max_len)
    start_lo = max(1, span1 - gap_len + 1)
    start_hi = min(span0, limit - gap_len + 1)
    start = rng.randint(start_lo, start_hi)
    return start, start + gap_len - 1


def _frame_border_cells(height: int, width: int, direction: str, gap0: int, gap1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(width):
        if direction != "up" or not (gap0 <= c <= gap1):
            cells.add((0, c))
        if direction != "down" or not (gap0 <= c <= gap1):
            cells.add((height - 1, c))
    for r in range(height):
        if direction != "left" or not (gap0 <= r <= gap1):
            cells.add((r, 0))
        if direction != "right" or not (gap0 <= r <= gap1):
            cells.add((r, width - 1))
    return cells


def _translated_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _place_frame(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    pattern_color: int,
    frame_color: int,
    direction: str,
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_TRIES):
        sprite = _random_sprite(rng)
        sh, sw = sprite.shape
        ih = rng.randint(max(INTERIOR_H_RANGE[0], sh + 2), INTERIOR_H_RANGE[1])
        iw = rng.randint(max(INTERIOR_W_RANGE[0], sw + 2), INTERIOR_W_RANGE[1])

        sr = rng.randint(1, ih - sh - 1)
        sc = rng.randint(1, iw - sw - 1)

        if direction in ("up", "down"):
            gap0, gap1 = _choose_gap(1 + sc, 1 + sc + sw - 1, iw, rng)
        else:
            gap0, gap1 = _choose_gap(1 + sr, 1 + sr + sh - 1, ih, rng)

        fh, fw = ih + 2, iw + 2
        r_min = 0
        r_max = h - fh
        c_min = 0
        c_max = w - fw
        if direction == "up":
            r_min = sh
        elif direction == "down":
            r_max = h - fh - sh
        elif direction == "left":
            c_min = sw
        else:
            c_max = w - fw - sw
        if r_min > r_max or c_min > c_max:
            continue

        rel_sprite = {(sr + r, sc + c) for r, c in np.argwhere(sprite)}
        border = _frame_border_cells(fh, fw, direction, gap0, gap1)

        for _ in range(40):
            fr0 = rng.randint(r_min, r_max)
            fc0 = rng.randint(c_min, c_max)
            abs_border = _translated_cells(border, fr0, fc0)
            abs_sprite = _translated_cells(rel_sprite, fr0 + 1, fc0 + 1)

            if direction == "up":
                moved = _translated_cells(abs_sprite, -max(r for r, _ in abs_sprite) + (fr0 - 1), 0)
            elif direction == "down":
                shift = (fr0 + fh) - min(r for r, _ in abs_sprite)
                moved = _translated_cells(abs_sprite, shift, 0)
            elif direction == "left":
                shift = (fc0 - 1) - max(c for _, c in abs_sprite)
                moved = _translated_cells(abs_sprite, 0, shift)
            else:
                shift = (fc0 + fw) - min(c for _, c in abs_sprite)
                moved = _translated_cells(abs_sprite, 0, shift)

            all_cells = abs_border | abs_sprite | moved
            if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
                continue
            if all_cells & blocked:
                continue

            for r, c in abs_border:
                grid[r, c] = frame_color
            for r, c in abs_sprite:
                grid[r, c] = pattern_color
            return abs_border | abs_sprite, moved
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.randint(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1])
    legend_count = rng.randint(2, min(4, frame_count))
    palette = PALETTE[:]
    rng.shuffle(palette)
    pattern_colors = palette[:legend_count]
    frame_colors = palette[legend_count : legend_count + frame_count]

    directions = rng.sample(list(ALL_DIRS), legend_count)
    legend_dirs = {color: direction for color, direction in zip(pattern_colors, directions)}

    assignments = pattern_colors[:]
    while len(assignments) < frame_count:
        assignments.append(rng.choice(pattern_colors))
    rng.shuffle(assignments)

    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    grid = np.full((h, w), BG, dtype=int)

    legend = _legend_cells(pattern_colors, legend_dirs, grid.shape, rng)
    if legend is None:
        return None

    blocked: set[tuple[int, int]] = set()
    legend_cells = set(legend)
    blocked |= _expand_cells(legend_cells, grid.shape, pad=1)
    for (r, c), color in legend.items():
        grid[r, c] = color

    specs = list(zip(assignments, frame_colors))
    rng.shuffle(specs)
    for pattern_color, frame_color in specs:
        placed = _place_frame(grid, blocked, pattern_color, frame_color, legend_dirs[pattern_color], rng)
        if placed is None:
            return None
        input_cells, output_cells = placed
        blocked |= _expand_cells(input_cells | output_cells, grid.shape, pad=1)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
