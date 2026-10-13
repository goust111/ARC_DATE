from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
MAROON = 9
ARROW_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)

DIRECTION_INFO = {
    "up": {
        "move": (-1, 0),
        "tails": ((1, -1), (1, 1)),
        "axis": "col",
        "family": "vertical",
    },
    "down": {
        "move": (1, 0),
        "tails": ((-1, -1), (-1, 1)),
        "axis": "col",
        "family": "vertical",
    },
    "left": {
        "move": (0, -1),
        "tails": ((-1, 1), (1, 1)),
        "axis": "row",
        "family": "horizontal",
    },
    "right": {
        "move": (0, 1),
        "tails": ((-1, -1), (1, -1)),
        "axis": "row",
        "family": "horizontal",
    },
}


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                comp.append((int(cr), int(cc)))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _match_arrow(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], str]:
    pts = set(coords)
    assert len(pts) == 3, "each arrow component must have exactly three pixels"
    matches: list[tuple[tuple[int, int], str]] = []
    for hr, hc in pts:
        for direction, info in DIRECTION_INFO.items():
            needed = {(hr, hc)}
            for dr, dc in info["tails"]:
                needed.add((hr + dr, hc + dc))
            if needed == pts:
                matches.append(((int(hr), int(hc)), direction))
    assert len(matches) == 1, "each three-pixel component must match exactly one arrow template"
    return matches[0]


def _extract_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    arrows: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _components8(g == color):
            head, direction = _match_arrow(comp)
            arrows.append(
                {
                    "color": int(color),
                    "head": head,
                    "direction": direction,
                    "cells": tuple(sorted(comp)),
                }
            )
    assert arrows, "expected at least one arrow"
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each colored arrow emits a ray in its pointing direction.

    The ray length equals the count of same-color pixels on the arrow head's row
    (for left/right arrows) or on the arrow head's column (for up/down arrows).
    Arrow pixels remain unchanged. If rays of different colors hit the same empty
    cell, that cell becomes maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    arrows = _extract_arrows(g)
    arrow_cells = g != BG
    claims: dict[tuple[int, int], set[int]] = defaultdict(set)

    for arrow in arrows:
        color = int(arrow["color"])
        hr, hc = arrow["head"]
        info = DIRECTION_INFO[str(arrow["direction"])]
        if info["axis"] == "row":
            length = int(np.sum(g[hr, :] == color))
        else:
            length = int(np.sum(g[:, hc] == color))

        dr, dc = info["move"]
        for step in range(1, length + 1):
            nr = hr + dr * step
            nc = hc + dc * step
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if arrow_cells[nr, nc]:
                continue
            claims[(int(nr), int(nc))].add(color)

    out = g.copy()
    for (r, c), colors in claims.items():
        out[r, c] = MAROON if len(colors) >= 2 else next(iter(colors))
    return out


def _arrow_cells(head: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    hr, hc = head
    info = DIRECTION_INFO[direction]
    cells = [(int(hr), int(hc))]
    for dr, dc in info["tails"]:
        cells.append((int(hr + dr), int(hc + dc)))
    return cells


def _can_place(blocked: np.ndarray, head: tuple[int, int], direction: str) -> bool:
    h, w = blocked.shape
    for r, c in _arrow_cells(head, direction):
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_arrow(
    grid: np.ndarray,
    blocked: np.ndarray,
    head: tuple[int, int],
    direction: str,
    color: int,
) -> dict[str, object] | None:
    if not _can_place(blocked, head, direction):
        return None
    cells = _arrow_cells(head, direction)
    for r, c in cells:
        grid[r, c] = int(color)
    _mark_blocked(blocked, cells)
    return {"color": int(color), "head": head, "direction": direction, "cells": tuple(cells)}


def _safe_head_options(height: int, width: int, direction: str) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if _can_place(np.zeros((height, width), dtype=bool), (r, c), direction):
                options.append((int(r), int(c)))
    return options


def _place_random_arrow(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    *,
    direction_choices: tuple[str, ...] = ("up", "down", "left", "right"),
    fixed_row: int | None = None,
    fixed_col: int | None = None,
    attempts: int = 120,
) -> dict[str, object] | None:
    height, width = grid.shape
    for _ in range(attempts):
        direction = random.choice(direction_choices)
        options = _safe_head_options(height, width, direction)
        if fixed_row is not None:
            options = [head for head in options if head[0] == fixed_row]
        if fixed_col is not None:
            options = [head for head in options if head[1] == fixed_col]
        random.shuffle(options)
        for head in options:
            placed = _place_arrow(grid, blocked, head, direction, color)
            if placed is not None:
                return placed
    return None


def _visible_ray_pixels(inp: np.ndarray, out: np.ndarray) -> int:
    return int(np.count_nonzero((inp == BG) & (out != BG)))


def _directions_present(arrows: list[dict[str, object]]) -> set[str]:
    return {str(arrow["direction"]) for arrow in arrows}


def _gen_crossing(height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    colors = random.sample(ARROW_COLORS, 2)
    h_color, v_color = int(colors[0]), int(colors[1])
    h_dir = random.choice(("left", "right"))
    v_dir = random.choice(("up", "down"))

    ir = random.randint(3, height - 4)
    ic = random.randint(3, width - 4)
    dh = random.randint(2, 3)
    dv = random.randint(2, 3)

    if h_dir == "right":
        h_head = (ir, ic - dh)
    else:
        h_head = (ir, ic + dh)
    if v_dir == "down":
        v_head = (ir - dv, ic)
    else:
        v_head = (ir + dv, ic)

    main_h = _place_arrow(grid, blocked, h_head, h_dir, h_color)
    if main_h is None:
        return None
    main_v = _place_arrow(grid, blocked, v_head, v_dir, v_color)
    if main_v is None:
        return None

    for _ in range(dh - 1):
        if _place_random_arrow(
            grid,
            blocked,
            h_color,
            direction_choices=("left", "right"),
            fixed_row=ir,
        ) is None:
            return None

    for _ in range(dv - 1):
        if _place_random_arrow(
            grid,
            blocked,
            v_color,
            direction_choices=("up", "down"),
            fixed_col=ic,
        ) is None:
            return None

    arrows = _extract_arrows(grid)
    if len(arrows) < 4:
        return None
    if len(_directions_present(arrows)) < 2:
        return None

    out = main(grid)
    if not np.any(out == MAROON):
        return None
    if _visible_ray_pixels(grid, out) < 4:
        return None
    if np.any((grid != BG) & (out == MAROON)):
        return None
    return grid


def _gen_non_crossing(height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    colors = random.sample(ARROW_COLORS, random.choice((2, 3)))
    pair_color = int(colors[0])
    other_colors = [int(c) for c in colors[1:]]

    base_dir = random.choice(("up", "down", "left", "right"))
    base = _place_random_arrow(grid, blocked, pair_color, direction_choices=(base_dir,))
    if base is None:
        return None

    family = DIRECTION_INFO[base_dir]["family"]
    if family == "horizontal":
        partner = _place_random_arrow(
            grid,
            blocked,
            pair_color,
            direction_choices=("left", "right"),
            fixed_row=int(base["head"][0]),
        )
    else:
        partner = _place_random_arrow(
            grid,
            blocked,
            pair_color,
            direction_choices=("up", "down"),
            fixed_col=int(base["head"][1]),
        )
    if partner is None:
        return None

    total_arrows = random.randint(3, 6)
    while len(_extract_arrows(grid)) < total_arrows:
        color = random.choice(other_colors if other_colors else list(ARROW_COLORS))
        if _place_random_arrow(grid, blocked, int(color)) is None:
            break

    arrows = _extract_arrows(grid)
    if len(arrows) < 3:
        return None
    if len({int(arrow["color"]) for arrow in arrows}) < 2:
        return None
    if len(_directions_present(arrows)) < 2:
        return None

    out = main(grid)
    if np.any(out == MAROON):
        return None
    if _visible_ray_pixels(grid, out) < 3:
        return None
    if np.any((grid != BG) & (out != grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        height = random.randint(10, 18)
        width = random.randint(10, 18)
        use_crossing = random.random() < 0.55
        grid = _gen_crossing(height, width) if use_crossing else _gen_non_crossing(height, width)
        if grid is None:
            continue
        _ = main(grid)
        return grid
    raise RuntimeError("failed to generate a valid input")
