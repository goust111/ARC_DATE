import random
from collections import Counter, deque

import numpy as np


BG = 0
YELLOW = 4
SPRITE_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
QUADRANTS = ("NW", "NE", "SW", "SE")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_monochrome_components(grid: np.ndarray, excluded_colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if color in excluded_colors:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(len(cells)),
                }
            )
    return out


def _infer_pivot(yellow_coords: np.ndarray) -> tuple[int, int]:
    assert yellow_coords.shape == (3, 2), "expected exactly three yellow corner markers"
    rows = sorted(set(map(int, yellow_coords[:, 0])))
    cols = sorted(set(map(int, yellow_coords[:, 1])))
    assert len(rows) == 2 and len(cols) == 2, "yellow markers must occupy three corners of an axis-aligned square"
    assert rows[1] - rows[0] == cols[1] - cols[0] > 0, "yellow markers must define a square"
    corners = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    present = {tuple(map(int, xy)) for xy in yellow_coords}
    missing = list(corners - present)
    assert len(missing) == 1, "square pivot must be uniquely determined"
    return missing[0]


def _rotate_point_cw(point: tuple[int, int], pivot: tuple[int, int], turns: int) -> tuple[int, int]:
    pr, pc = pivot
    dr = int(point[0]) - pr
    dc = int(point[1]) - pc
    for _ in range(int(turns) % 4):
        dr, dc = dc, -dr
    return pr + dr, pc + dc


def _rotate_coords_cw(coords: np.ndarray, pivot: tuple[int, int], turns: int) -> np.ndarray:
    return np.array([_rotate_point_cw((int(r), int(c)), pivot, turns) for r, c in coords], dtype=int)


def _extract_entities(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BG, "background must be black"

    yellow_coords = np.argwhere(g == YELLOW)
    pivot = _infer_pivot(yellow_coords)

    components = _find_monochrome_components(g, excluded_colors={bg, YELLOW})
    assert components, "expected legends and sprites"

    legend_heights: dict[int, int] = {}
    legend_cells: dict[int, set[tuple[int, int]]] = {}
    sprites: dict[int, np.ndarray] = {}

    for comp in components:
        color = int(comp["color"])
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        if r0 == 0:
            assert c0 == c1, "legend blocks must be one column wide"
            assert comp["area"] == r1 - r0 + 1, "legend blocks must be solid vertical stacks"
            assert 1 <= comp["area"] <= 3, "legend height encodes 1-3 clockwise quarter-turns"
            assert color not in legend_heights, "legend colors must be unique"
            legend_heights[color] = int(comp["area"])
            legend_cells[color] = {tuple(map(int, rc)) for rc in coords}
        else:
            assert color not in sprites, "expected one sprite per color"
            sprites[color] = coords

    assert legend_heights, "expected at least one legend block"
    assert set(legend_heights) == set(sprites), "legend colors must match sprite colors one-to-one"
    return {
        "grid": g,
        "bg": bg,
        "pivot": pivot,
        "yellow": {tuple(map(int, rc)) for rc in yellow_coords},
        "legend_heights": legend_heights,
        "legend_cells": legend_cells,
        "sprites": sprites,
    }


def _render_rotated(grid: np.ndarray, entities: dict) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = int(entities["bg"])
    h, w = out.shape

    for coords in entities["sprites"].values():
        out[coords[:, 0], coords[:, 1]] = bg

    for color, coords in entities["sprites"].items():
        turns = int(entities["legend_heights"][color])
        rotated = _rotate_coords_cw(coords, entities["pivot"], turns)
        for rr, cc in rotated:
            assert 0 <= rr < h and 0 <= cc < w, "rotated sprite left the canvas"
            assert (rr, cc) != entities["pivot"], "pivot cell must remain empty"
            current = int(out[rr, cc])
            assert current in (bg, color), "rotated sprites must not collide with fixed structures or other colors"
            out[rr, cc] = int(color)
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored sprite clockwise around the square corner implied by three yellow markers.

    Input structure:
    - The background is black.
    - Three yellow singleton markers occupy three corners of an axis-aligned square; the fourth corner is absent.
    - Several distinct-color monochrome sprites are scattered elsewhere, and matching vertical legend blocks touch the top edge at the upper left.

    Rule:
    - Infer the missing square corner from the three yellow markers and treat that empty cell as the rotation pivot.
    - For each sprite color K, read the height k of the top-touching legend block of the same color and rotate the entire sprite clockwise by 90 degrees times k around the pivot.
    - Clear the original sprite positions, keep the yellow markers and legend blocks unchanged, and draw only the rotated sprites.

    Key insight:
    - The pivot is never painted directly: it is recovered as the missing fourth square corner, while each legend's height encodes a quarter-turn count rather than a geometric size.
    """
    entities = _extract_entities(np.asarray(input_grid, dtype=int))
    return _render_rotated(entities["grid"], entities)


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _rotate_shape(shape: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    cells = {(int(r), int(c)) for r, c in shape}
    for _ in range(int(turns) % 4):
        cells = {(c, -r) for r, c in cells}
        cells = set(_normalize_cells(cells))
    return tuple(sorted(cells))


def _random_shape() -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        target = random.randint(3, 6)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells and -2 <= nr <= 2 and -2 <= nc <= 2:
                        frontier.add((nr, nc))
            if not frontier:
                break
            candidate = random.choice(sorted(frontier))
            trial = set(cells)
            trial.add(candidate)
            shape = _normalize_cells(trial)
            h, w = _shape_dims(shape)
            if h <= 4 and w <= 4:
                cells = trial
        shape = _normalize_cells(cells)
        h, w = _shape_dims(shape)
        if len(shape) != target or h < 2 or w < 2:
            continue
        if any(_rotate_shape(shape, t) == shape for t in (1, 2, 3)):
            continue
        return shape
    raise RuntimeError("failed to sample a non-symmetric sprite")


def _expand(cells: set[tuple[int, int]], radius: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((r + dr, c + dc))
    return out


def _marker_cells(pivot: tuple[int, int], side: int, missing_corner: str) -> set[tuple[int, int]]:
    pr, pc = pivot
    if missing_corner == "tl":
        return {(pr, pc + side), (pr + side, pc), (pr + side, pc + side)}
    if missing_corner == "tr":
        return {(pr, pc - side), (pr + side, pc), (pr + side, pc - side)}
    if missing_corner == "bl":
        return {(pr - side, pc), (pr, pc + side), (pr - side, pc + side)}
    if missing_corner == "br":
        return {(pr - side, pc), (pr, pc - side), (pr - side, pc - side)}
    raise ValueError("invalid missing corner")


def _quadrant_anchor_bounds(
    quadrant: str, pivot: tuple[int, int], shape_h: int, shape_w: int, h: int, w: int
) -> tuple[int, int, int, int] | None:
    pr, pc = pivot
    if quadrant == "NW":
        r0, r1 = 4, pr - shape_h - 2
        c0, c1 = 0, pc - shape_w - 2
    elif quadrant == "NE":
        r0, r1 = 4, pr - shape_h - 2
        c0, c1 = pc + 2, w - shape_w - 1
    elif quadrant == "SW":
        r0, r1 = pr + 2, h - shape_h - 1
        c0, c1 = 0, pc - shape_w - 2
    else:
        r0, r1 = pr + 2, h - shape_h - 1
        c0, c1 = pc + 2, w - shape_w - 1
    if r0 > r1 or c0 > c1:
        return None
    return r0, r1, c0, c1


def _generate_case() -> np.ndarray | None:
    h = random.randint(20, 29)
    w = random.randint(20, 29)
    grid = np.zeros((h, w), dtype=int)

    sprite_count = random.choice((2, 3, 3, 4))
    colors = random.sample(SPRITE_COLORS, sprite_count)
    if sprite_count >= 3:
        turns = [1, 2, 3] + [random.choice((1, 2, 3)) for _ in range(sprite_count - 3)]
    else:
        turns = [random.choice((1, 3)), 2]
    random.shuffle(turns)

    max_legend_col = 2 * (sprite_count - 1)
    if max_legend_col >= w - 2:
        return None
    legend_cells = set()
    for idx, (color, turn) in enumerate(zip(colors, turns)):
        col = 2 * idx
        for rr in range(turn):
            legend_cells.add((rr, col))
            grid[rr, col] = color

    pivot = (random.randint(8, h - 9), random.randint(8, w - 9))
    max_side = min(pivot[0] - 1, h - 1 - pivot[0], pivot[1] - 1, w - 1 - pivot[1], 8)
    if max_side < 4:
        return None
    side = random.randint(4, max_side)
    missing_corner = random.choice(("tl", "tr", "bl", "br"))
    markers = _marker_cells(pivot, side, missing_corner)
    if pivot in markers:
        return None
    for r, c in markers:
        grid[r, c] = YELLOW

    fixed_cells = set(legend_cells) | set(markers)
    input_blocked = _expand(fixed_cells, radius=1)
    output_blocked = _expand(fixed_cells, radius=1)
    placed_input = set()
    placed_output = set()

    quadrants = random.sample(QUADRANTS, sprite_count)
    for color, turn, quadrant in zip(colors, turns, quadrants):
        placed = False
        for _ in range(300):
            shape = _random_shape()
            shape_h, shape_w = _shape_dims(shape)
            bounds = _quadrant_anchor_bounds(quadrant, pivot, shape_h, shape_w, h, w)
            if bounds is None:
                return None
            r0_min, r0_max, c0_min, c0_max = bounds
            anchor_r = random.randint(r0_min, r0_max)
            anchor_c = random.randint(c0_min, c0_max)
            orig = {(anchor_r + r, anchor_c + c) for r, c in shape}
            rotated = {_rotate_point_cw(cell, pivot, turn) for cell in orig}
            if any(not (0 <= r < h and 0 <= c < w) for r, c in rotated):
                continue
            if pivot in rotated:
                continue
            if orig & input_blocked:
                continue
            if rotated & output_blocked:
                continue
            if orig & placed_output:
                continue
            if rotated & placed_input:
                continue
            if orig & rotated:
                continue
            if any(r == 0 for r, _ in orig):
                continue
            for r, c in orig:
                grid[r, c] = color
            placed_input |= orig
            placed_output |= rotated
            input_blocked |= _expand(orig, radius=1)
            output_blocked |= _expand(rotated, radius=1)
            placed = True
            break
        if not placed:
            return None

    if not _generated_case_is_valid(grid):
        return None
    return grid


def _generated_case_is_valid(grid: np.ndarray) -> bool:
    try:
        entities = _extract_entities(grid)
        if len(entities["sprites"]) < 2:
            return False
        turn_values = list(entities["legend_heights"].values())
        if len(set(turn_values)) < 2:
            return False
        if len(turn_values) >= 3 and {1, 2, 3} - set(turn_values):
            return False
        out = main(grid)
        if not isinstance(out, np.ndarray) or out.ndim != 2 or max(out.shape) > 30:
            return False
        if np.array_equal(out, grid):
            return False

        for marker in entities["yellow"]:
            if int(out[marker]) != YELLOW:
                return False

        for color, legend in entities["legend_cells"].items():
            for cell in legend:
                if int(out[cell]) != int(color):
                    return False

        for color, coords in entities["sprites"].items():
            original = {tuple(map(int, rc)) for rc in coords}
            rotated = {tuple(map(int, rc)) for rc in _rotate_coords_cw(coords, entities["pivot"], entities["legend_heights"][color])}
            for cell in rotated:
                if int(out[cell]) != int(color):
                    return False
            for cell in original - rotated:
                if int(out[cell]) != BG:
                    return False
            all_color_cells = {tuple(map(int, rc)) for rc in np.argwhere(out == color)}
            if all_color_cells != rotated | entities["legend_cells"][color]:
                return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-pivot rotation task")
