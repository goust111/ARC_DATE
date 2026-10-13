from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (
                        int(arr[:, 0].min()),
                        int(arr[:, 1].min()),
                        int(arr[:, 0].max()),
                        int(arr[:, 1].max()),
                    ),
                }
            )
    return comps


def cup_cavity_rows(coords: np.ndarray) -> list[np.ndarray]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    rows: list[np.ndarray] = []
    for lr in range(local.shape[0]):
        cols = np.flatnonzero(local[lr])
        if cols.size < 2:
            continue
        interior = np.flatnonzero(~local[lr, cols[0] + 1 : cols[-1]])
        if interior.size == 0:
            continue
        first = int(cols[0] + 1 + interior[0])
        last = int(cols[0] + 1 + interior[-1])
        assert not local[lr, first : last + 1].any(), "cavity row must be uninterrupted"
        row = np.array(
            [(r0 + lr, c0 + cc) for cc in range(first, last + 1)],
            dtype=int,
        )
        rows.append(row)

    assert rows, "expected at least one cavity row"
    row_ids = [int(row[0, 0]) for row in rows]
    assert row_ids == list(range(row_ids[0], row_ids[0] + len(row_ids))), "cavity rows must be contiguous"
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert color-matched scattered pixels into water volumes and level-fill the matching cups.

    Input structure:
    - The background is black and contains several open-top cup outlines, each in a unique color.
    - Cup interiors are empty background cells, and their widths may vary by row because the inner walls step inward.
    - The remaining foreground cells are isolated one-pixel scatter markers; some share a cup color and some are distractor colors.

    Rule:
    - Count the isolated markers whose color matches each cup outline; erase every marker regardless of whether it matches a cup.
    - For each cup, recover its cavity rows and interpret the marker count as a water amount that fills whole rows from the bottom upward.
    - Recolor exactly those filled cavity cells with the cup's own color while leaving all outlines unchanged.

    Key insight:
    - Marker positions do not matter; only the per-color counts matter, and each count is converted into a bottom-up fill height using that cup's row-by-row cavity geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    comps = monochrome_components(g, bg)

    cups = [comp for comp in comps if int(comp["area"]) > 1]
    dots = [comp for comp in comps if int(comp["area"]) == 1]
    assert cups, "expected at least one cup outline"

    cup_colors = [int(comp["color"]) for comp in cups]
    assert len(set(cup_colors)) == len(cup_colors), "cup colors must be unique"
    water_counts = Counter(
        int(comp["color"]) for comp in dots if int(comp["color"]) in set(cup_colors)
    )

    out = g.copy()
    for comp in dots:
        rr, cc = map(int, comp["coords"][0])
        out[rr, cc] = bg

    for cup in cups:
        color = int(cup["color"])
        cavity_rows = cup_cavity_rows(cup["coords"])
        target = int(water_counts[color])
        if target == 0:
            continue

        running = 0
        filled_rows = 0
        for row in reversed(cavity_rows):
            width = int(row.shape[0])
            if running + width > target:
                raise AssertionError("water amount must fill whole cavity rows")
            if running + width == target:
                filled_rows += 1
                running += width
                break
            filled_rows += 1
            running += width

        assert running == target, "water amount must match an integer number of rows"

        for row in cavity_rows[len(cavity_rows) - filled_rows :]:
            out[row[:, 0], row[:, 1]] = color

    return out


def _step_choices(prev_left: int, prev_right: int, min_width: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for dl in (0, 1):
        for dr in (0, 1):
            left = prev_left + dl
            right = prev_right - dr
            if right - left + 1 < min_width:
                continue
            options.append((left, right))
    return options


def _make_cavity_profile(height: int) -> list[tuple[int, int]]:
    while True:
        top_width = random.randint(4, 7)
        min_width = random.randint(2, min(4, top_width))
        rows = [(0, top_width - 1)]
        changed = False

        for i in range(1, height):
            prev_left, prev_right = rows[-1]
            choices = _step_choices(prev_left, prev_right, min_width)
            changing = [opt for opt in choices if opt != (prev_left, prev_right)]
            if not changed and i == height - 1 and changing:
                choices = changing
            elif not changed and changing and random.random() < 0.7:
                choices = changing

            left, right = random.choice(choices)
            changed |= (left != prev_left or right != prev_right)
            rows.append((left, right))

        if changed:
            return rows


def _build_cup_patch(color: int) -> tuple[np.ndarray, list[np.ndarray]]:
    height = random.randint(3, 6)
    cavity = _make_cavity_profile(height)
    wall_left = [left - 1 for left, _right in cavity]
    wall_right = [right + 1 for _left, right in cavity]
    shift = -min(wall_left)
    wall_left = [col + shift for col in wall_left]
    wall_right = [col + shift for col in wall_right]
    cavity = [(left + shift, right + shift) for left, right in cavity]

    patch = np.zeros((height + 1, max(wall_right) + 1), dtype=int)
    for r, (left, right) in enumerate(cavity):
        patch[r, wall_left[r]] = color
        patch[r, wall_right[r]] = color
        if r == height - 1:
            patch[height, wall_left[r] : wall_right[r] + 1] = color
            continue

        next_left, next_right = cavity[r + 1]
        if next_left > left:
            patch[r + 1, wall_left[r] : wall_left[r + 1] + 1] = color
        if next_right < right:
            patch[r + 1, wall_right[r + 1] : wall_right[r] + 1] = color

    cavity_rows = [
        np.array([(r, c) for c in range(left, right + 1)], dtype=int)
        for r, (left, right) in enumerate(cavity)
    ]
    return patch, cavity_rows


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    rr0 = max(0, top - 1)
    cc0 = max(0, left - 1)
    rr1 = min(H, top + h + 1)
    cc1 = min(W, left + w + 1)
    return not np.any(grid[rr0:rr1, cc0:cc1] != BG)


def _isolated_empty_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    H, W = grid.shape
    cells: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != BG:
                continue
            blocked = False
            for dr, dc in NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) != BG:
                    blocked = True
                    break
            if not blocked:
                cells.append((r, c))
    random.shuffle(cells)
    return cells


def _place_scatter_points(grid: np.ndarray, color_counts: list[tuple[int, int]]) -> bool:
    pool = _isolated_empty_cells(grid)
    taken: list[tuple[int, int]] = []
    for color, count in color_counts:
        for _ in range(count):
            placed = False
            for r, c in pool:
                if int(grid[r, c]) != BG:
                    continue
                if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in taken):
                    continue
                grid[r, c] = int(color)
                taken.append((r, c))
                placed = True
                break
            if not placed:
                return False
    return True


def _generate_attempt() -> np.ndarray | None:
    H = random.randint(18, 28)
    W = random.randint(18, 28)
    grid = np.full((H, W), BG, dtype=int)

    cup_count = random.randint(2, 3)
    cup_colors = random.sample(PALETTE, cup_count)
    filler_rows: list[int] = []
    dot_counts: list[tuple[int, int]] = []

    for i, color in enumerate(cup_colors):
        patch, cavity_rows = _build_cup_patch(color)
        max_fill_rows = len(cavity_rows)
        fill_rows = random.randint(0, max_fill_rows)
        if i == 0 and fill_rows == 0:
            fill_rows = random.randint(1, max_fill_rows)
        filler_rows.append(fill_rows)
        water = sum(int(row.shape[0]) for row in cavity_rows[-fill_rows:]) if fill_rows else 0
        dot_counts.append((color, water))

        placed = False
        top_min = H // 3
        top_max = H - patch.shape[0] - 1
        left_max = W - patch.shape[1] - 1
        if top_min > top_max or left_max < 1:
            return None
        positions = [
            (r, c)
            for r in range(top_min, top_max + 1)
            for c in range(1, left_max + 1)
        ]
        random.shuffle(positions)
        for top, left in positions:
            if not _can_place_patch(grid, patch, top, left):
                continue
            sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
            sub[patch != BG] = patch[patch != BG]
            placed = True
            break
        if not placed:
            return None

    distractor_palette = [color for color in PALETTE if color not in cup_colors]
    distractor_colors = random.sample(distractor_palette, random.randint(1, min(3, len(distractor_palette))))
    for color in distractor_colors:
        dot_counts.append((color, random.randint(1, 4)))

    if not _place_scatter_points(grid, dot_counts):
        return None

    comp_sizes = [comp["area"] for comp in monochrome_components(grid, BG)]
    if not any(area == 1 for area in comp_sizes):
        return None
    if sum(1 for area in comp_sizes if area > 1) != cup_count:
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-fill input")
