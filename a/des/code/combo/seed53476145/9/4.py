from __future__ import annotations

import random

import numpy as np


BG = 7
ARROW = 5
SPRITE_COLORS = [1, 2, 3, 4, 6, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue

        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in DIR4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)

    return comps


def _classify_arrow(cells: list[tuple[int, int]], h: int, w: int) -> tuple[int, int]:
    assert len(cells) == 3, "each arrow must have exactly three pixels"

    top = sorted(c for r, c in cells if r == 0)
    if len(top) == 2 and top[1] == top[0] + 1:
        inner = [(r, c) for r, c in cells if r != 0]
        if len(inner) == 1 and inner[0][0] == 1 and inner[0][1] in top:
            return (-1, 0) if inner[0][1] == top[0] else (1, 0)

    bottom = sorted(c for r, c in cells if r == h - 1)
    if len(bottom) == 2 and bottom[1] == bottom[0] + 1:
        inner = [(r, c) for r, c in cells if r != h - 1]
        if len(inner) == 1 and inner[0][0] == h - 2 and inner[0][1] in bottom:
            return (-1, 0) if inner[0][1] == bottom[0] else (1, 0)

    left = sorted(r for r, c in cells if c == 0)
    if len(left) == 2 and left[1] == left[0] + 1:
        inner = [(r, c) for r, c in cells if c != 0]
        if len(inner) == 1 and inner[0][1] == 1 and inner[0][0] in left:
            return (0, -1) if inner[0][0] == left[0] else (0, 1)

    right = sorted(r for r, c in cells if c == w - 1)
    if len(right) == 2 and right[1] == right[0] + 1:
        inner = [(r, c) for r, c in cells if c != w - 1]
        if len(inner) == 1 and inner[0][1] == w - 2 and inner[0][0] in right:
            return (0, -1) if inner[0][0] == right[0] else (0, 1)

    raise AssertionError("grey component is not a valid edge L-arrow")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sum all edge-arrow directions, then translate the central sprite by the
    resulting vector. Grey arrows and the sprite's original position are erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    arrow_mask = g == ARROW
    dx = 0
    dy = 0
    for comp in _components(arrow_mask):
        add_x, add_y = _classify_arrow(comp, h, w)
        dx += add_x
        dy += add_y

    sprite_mask = (g != BG) & (g != ARROW)
    sprite_comps = _components(sprite_mask)
    assert len(sprite_comps) == 1, "expected exactly one non-arrow sprite"

    out = g.copy()
    out[arrow_mask | sprite_mask] = BG
    for r, c in sprite_comps[0]:
        nr, nc = r + dy, c + dx
        if 0 <= nr < h and 0 <= nc < w:
            out[nr, nc] = g[r, c]
    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 16)
GRID_W_RANGE = (10, 16)
SIDES = ("top", "bottom", "left", "right")
DIRS_BY_SIDE = {
    "top": ("left", "right"),
    "bottom": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}
VEC_BY_DIR = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}
MAX_ATTEMPTS = 400


def _sample_side_counts() -> dict[str, int]:
    require_both_axes = random.random() < 0.6
    for _ in range(100):
        counts = {side: random.randint(0, 2) for side in SIDES}
        total = sum(counts.values())
        has_horizontal = counts["top"] + counts["bottom"] > 0
        has_vertical = counts["left"] + counts["right"] > 0
        if not 2 <= total <= 6:
            continue
        if require_both_axes and not (has_horizontal and has_vertical):
            continue
        return counts
    raise RuntimeError("failed to sample arrow counts")


def _sample_starts(edge_len: int, count: int) -> list[int]:
    candidates = list(range(2, edge_len - 3))
    if count == 0:
        return []
    if count == 1:
        return [random.choice(candidates)]

    pairs = [
        (a, b)
        for i, a in enumerate(candidates)
        for b in candidates[i + 1 :]
        if b - a >= 3
    ]
    a, b = random.choice(pairs)
    return [a, b]


def _arrow_cells(side: str, start: int, direction: str, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return (
            [(0, start), (0, start + 1), (1, start)]
            if direction == "left"
            else [(0, start), (0, start + 1), (1, start + 1)]
        )
    if side == "bottom":
        return (
            [(h - 1, start), (h - 1, start + 1), (h - 2, start)]
            if direction == "left"
            else [(h - 1, start), (h - 1, start + 1), (h - 2, start + 1)]
        )
    if side == "left":
        return (
            [(start, 0), (start + 1, 0), (start, 1)]
            if direction == "up"
            else [(start, 0), (start + 1, 0), (start + 1, 1)]
        )
    assert side == "right"
    return (
        [(start, w - 1), (start + 1, w - 1), (start, w - 2)]
        if direction == "up"
        else [(start, w - 1), (start + 1, w - 1), (start + 1, w - 2)]
    )


def _sample_arrows(h: int, w: int) -> list[dict[str, object]]:
    counts = _sample_side_counts()
    arrows: list[dict[str, object]] = []
    for side in SIDES:
        edge_len = w if side in {"top", "bottom"} else h
        for start in _sample_starts(edge_len, counts[side]):
            direction = random.choice(DIRS_BY_SIDE[side])
            arrows.append(
                {
                    "side": side,
                    "direction": direction,
                    "cells": _arrow_cells(side, start, direction, h, w),
                }
            )
    random.shuffle(arrows)
    return arrows


def _random_shape(max_h: int, max_w: int) -> tuple[list[tuple[int, int]], int, int] | None:
    for _ in range(100):
        shape_h = random.randint(2, max_h)
        shape_w = random.randint(2, max_w)
        max_cells = min(shape_h * shape_w, 9)
        target = random.randint(4, max_cells)

        start = (random.randint(0, shape_h - 1), random.randint(0, shape_w - 1))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < shape_h and 0 <= nc < shape_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(cells) != target or len(rows) < 2 or len(cols) < 2:
            continue

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        final_h = max(r for r, _ in norm) + 1
        final_w = max(c for _, c in norm) + 1
        return norm, final_h, final_w
    return None


def _placed_cells(shape: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _translated_kind(cells: list[tuple[int, int]], dx: int, dy: int, h: int, w: int) -> str:
    kept = sum(0 <= r + dy < h and 0 <= c + dx < w for r, c in cells)
    if kept == len(cells):
        return "inside"
    if kept == 0:
        return "gone"
    return "clipped"


def _choose_sprite_position(
    shape: list[tuple[int, int]],
    shape_h: int,
    shape_w: int,
    dx: int,
    dy: int,
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    positions = {"inside": [], "clipped": []}
    for top in range(2, h - shape_h - 1):
        for left in range(2, w - shape_w - 1):
            cells = _placed_cells(shape, top, left)
            kind = _translated_kind(cells, dx, dy, h, w)
            if kind in positions:
                positions[kind].append(cells)

    want_clipped = (dx != 0 or dy != 0) and random.random() < 0.4
    preferred = "clipped" if want_clipped else "inside"
    if positions[preferred]:
        return random.choice(positions[preferred])
    fallback = "inside" if positions["inside"] else "clipped"
    if positions[fallback]:
        return random.choice(positions[fallback])
    return None


def _render_input(
    h: int,
    w: int,
    arrows: list[dict[str, object]],
    sprite_cells: list[tuple[int, int]],
    sprite_color: int,
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[r, c] = ARROW
    for r, c in sprite_cells:
        grid[r, c] = sprite_color
    return grid


def _render_expected(
    h: int,
    w: int,
    sprite_cells: list[tuple[int, int]],
    sprite_color: int,
    dx: int,
    dy: int,
) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for r, c in sprite_cells:
        nr, nc = r + dy, c + dx
        if 0 <= nr < h and 0 <= nc < w:
            out[nr, nc] = sprite_color
    return out


def _attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    arrows = _sample_arrows(h, w)

    dx = 0
    dy = 0
    for arrow in arrows:
        add_x, add_y = VEC_BY_DIR[arrow["direction"]]
        dx += add_x
        dy += add_y

    shape = _random_shape(max_h=min(5, h - 4), max_w=min(5, w - 4))
    if shape is None:
        return None
    shape_cells, shape_h, shape_w = shape

    sprite_cells = _choose_sprite_position(shape_cells, shape_h, shape_w, dx, dy, h, w)
    if sprite_cells is None:
        return None

    sprite_color = random.choice(SPRITE_COLORS)
    inp = _render_input(h, w, arrows, sprite_cells, sprite_color)
    expected = _render_expected(h, w, sprite_cells, sprite_color, dx, dy)

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
