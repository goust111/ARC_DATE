from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (14, 23)
GRID_W_RANGE = (14, 23)
SPRITE_COUNT_RANGE = (3, 7)
LOCAL_BOX_RANGE = (2, 6)
SHAPE_TRIES = 80
PLACEMENT_TRIES = 120
GENERATION_TRIES = 400

NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS8:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each sprite based on how many distinct foreground colors it contains.

    - monochrome sprite: unchanged
    - two-color sprite: swap the two colors
    - three-or-more-color sprite: recolor the whole sprite to grey
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for comp in _components8(g != BG):
        colors = list({int(g[r, c]) for r, c in comp})
        if len(colors) == 1:
            continue
        if len(colors) == 2:
            a, b = colors
            for r, c in comp:
                out[r, c] = int(b if int(g[r, c]) == int(a) else a)
            continue
        for r, c in comp:
            out[r, c] = GREY
    return out


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in coords)


def _grow_shape(box_h: int, box_w: int, cell_count: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(box_h), random.randrange(box_w))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in NEIGHBORS8:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < cell_count and frontier:
        chosen = random.choice(tuple(frontier))
        frontier.remove(chosen)
        if chosen in cells:
            continue
        cells.add(chosen)
        add_frontier(*chosen)

    if len(cells) != cell_count:
        return None
    return _normalize(list(cells))


def _shape_dims(coords: list[tuple[int, int]]) -> tuple[int, int]:
    height = 1 + max(r for r, _c in coords)
    width = 1 + max(c for _r, c in coords)
    return int(height), int(width)


def _colorize(coords: list[tuple[int, int]], kind: str) -> dict[tuple[int, int], int]:
    shuffled = coords[:]
    random.shuffle(shuffled)

    if kind == "mono":
        color = int(random.choice(SPRITE_COLORS))
        return {cell: color for cell in coords}

    if kind == "swap":
        a, b = random.sample(SPRITE_COLORS, 2)
        split = random.randint(1, len(shuffled) - 1)
        colored: dict[tuple[int, int], int] = {}
        for idx, cell in enumerate(shuffled):
            colored[cell] = int(a if idx < split else b)
        return colored

    color_count = random.randint(3, min(4, len(shuffled), len(SPRITE_COLORS)))
    palette = random.sample(SPRITE_COLORS, color_count)
    colored = {}
    for idx, cell in enumerate(shuffled):
        if idx < color_count:
            colored[cell] = int(palette[idx])
        else:
            colored[cell] = int(random.choice(palette))
    return colored


def _make_sprite(kind: str) -> dict[str, object] | None:
    min_cells = {"mono": 1, "swap": 2, "multi": 3}[kind]
    for _ in range(SHAPE_TRIES):
        box_h = random.randint(*LOCAL_BOX_RANGE)
        box_w = random.randint(*LOCAL_BOX_RANGE)
        area = int(box_h * box_w)
        if area < min_cells:
            continue
        upper = max(min_cells, int(round(area * random.uniform(0.45, 0.85))))
        cell_count = random.randint(min_cells, min(area, upper))
        coords = _grow_shape(box_h, box_w, cell_count)
        if coords is None:
            continue
        colors = _colorize(coords, kind)
        height, width = _shape_dims(coords)
        return {
            "kind": kind,
            "coords": coords,
            "colors": colors,
            "height": height,
            "width": width,
        }
    return None


def _can_place(blocked: np.ndarray, top: int, left: int, coords: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in coords:
        rr = int(top + r)
        cc = int(left + c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if bool(blocked[rr, cc]):
            return False
    return True


def _stamp_sprite(
    grid: np.ndarray,
    blocked: np.ndarray,
    top: int,
    left: int,
    sprite: dict[str, object],
) -> None:
    coords = sprite["coords"]
    colors = sprite["colors"]
    assert isinstance(coords, list)
    assert isinstance(colors, dict)
    for r, c in coords:
        rr = int(top + r)
        cc = int(left + c)
        grid[rr, cc] = int(colors[(r, c)])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    blocked[nr, nc] = True


def _branch_label(color_count: int) -> str:
    if color_count == 1:
        return "mono"
    if color_count == 2:
        return "swap"
    return "multi"


def _validate_grid(grid: np.ndarray, expected_kinds: list[str]) -> bool:
    comps = _components8(grid != BG)
    if len(comps) != len(expected_kinds):
        return False

    actual = Counter()
    for comp in comps:
        color_count = len({int(grid[r, c]) for r, c in comp})
        actual[_branch_label(color_count)] += 1
    if actual != Counter(expected_kinds):
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    sprite_count = random.randint(*SPRITE_COUNT_RANGE)
    kinds = ["mono", "swap", "multi"]
    while len(kinds) < sprite_count:
        kinds.append(random.choice(("mono", "swap", "multi")))
    random.shuffle(kinds)

    sprites: list[dict[str, object]] = []
    for kind in kinds:
        sprite = _make_sprite(kind)
        if sprite is None:
            return None
        sprites.append(sprite)

    order = list(range(len(sprites)))
    random.shuffle(order)
    for idx in order:
        sprite = sprites[idx]
        sprite_h = int(sprite["height"])
        sprite_w = int(sprite["width"])
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(0, h - sprite_h)
            left = random.randint(0, w - sprite_w)
            coords = sprite["coords"]
            assert isinstance(coords, list)
            if not _can_place(blocked, top, left, coords):
                continue
            _stamp_sprite(grid, blocked, top, left, sprite)
            placed = True
            break
        if not placed:
            return None

    if not _validate_grid(grid, kinds):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_TRIES} attempts")
