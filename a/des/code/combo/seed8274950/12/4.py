from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR_TO_TURNS = {
    "up": 0,
    "right": 1,
    "down": 2,
    "left": 3,
}

BG_PALETTE = tuple(range(10))
GRID_HEIGHT_RANGE = (22, 30)
GRID_WIDTH_RANGE = (22, 30)
OBJECT_COUNT_OPTIONS = (4, 5)
CLEARANCE = 1
BORDER = 1
GEN_ATTEMPTS = 300
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 250
ODD_DIMS = (3, 5, 7)
EVEN_DIMS = (4, 6)

_RNG = random.Random()


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            colors = set()
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                colors.add(int(grid[rr, cc]))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            assert len(colors) == 1, "each foreground component must be monochrome"
            comps.append(np.asarray(coords, dtype=int))
    assert comps, "grid must contain at least one foreground object"
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _centroid_bias(coords: np.ndarray) -> tuple[int, int]:
    coords = np.asarray(coords, dtype=int)
    count = int(len(coords))
    r0, c0, r1, c1 = _bbox(coords)
    vertical_bias = int(2 * coords[:, 0].sum() - count * (r0 + r1))
    horizontal_bias = int(2 * coords[:, 1].sum() - count * (c0 + c1))
    return vertical_bias, horizontal_bias


def _dominant_direction(coords: np.ndarray, allow_tie: bool = False) -> str | None:
    vertical_bias, horizontal_bias = _centroid_bias(coords)
    if abs(vertical_bias) == abs(horizontal_bias):
        if allow_tie:
            return None
        raise AssertionError("centroid offset must have a unique dominant axis")
    assert abs(vertical_bias) != abs(horizontal_bias), "centroid offset must have a unique dominant axis"
    if abs(vertical_bias) > abs(horizontal_bias):
        return "up" if vertical_bias < 0 else "down"
    return "left" if horizontal_bias < 0 else "right"


def _rotate_coords(coords: np.ndarray, turns: int, bbox: tuple[int, int, int, int] | None = None) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    turns %= 4
    if turns == 0:
        return coords.copy()

    r0, c0, r1, c1 = _bbox(coords) if bbox is None else bbox
    R = r0 + r1
    C = c0 + c1
    out = []

    for r, c in coords:
        r = int(r)
        c = int(c)
        if turns == 1:
            nr_num = R + 2 * c - C
            nc_num = C - 2 * r + R
            assert nr_num % 2 == 0 and nc_num % 2 == 0, "90-degree rotation must stay on the integer lattice"
            nr, nc = nr_num // 2, nc_num // 2
        elif turns == 2:
            nr, nc = R - r, C - c
        else:
            nr_num = R - 2 * c + C
            nc_num = C + 2 * r - R
            assert nr_num % 2 == 0 and nc_num % 2 == 0, "90-degree rotation must stay on the integer lattice"
            nr, nc = nr_num // 2, nc_num // 2
        out.append((int(nr), int(nc)))

    rotated = np.asarray(out, dtype=int)
    assert len({(int(r), int(c)) for r, c in rotated}) == len(rotated)
    return rotated


def _normalize(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    mins = coords.min(axis=0)
    return coords - mins


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _background_color(g)
    comps = _components(g, bg)

    out = np.full((h, w), bg, dtype=int)
    for comp in comps:
        color = int(g[tuple(comp[0])])
        direction = _dominant_direction(comp)
        turns = DIR_TO_TURNS[direction]
        rotated = _rotate_coords(comp, turns)
        for rr, cc in rotated:
            rr = int(rr)
            cc = int(cc)
            assert 0 <= rr < h and 0 <= cc < w
            assert int(out[rr, cc]) == bg
            out[rr, cc] = color

    return out


def _make_up_shape() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        if _RNG.random() < 0.5:
            h = _RNG.choice(ODD_DIMS)
            w = _RNG.choice(ODD_DIMS)
        else:
            h = _RNG.choice(EVEN_DIMS)
            w = _RNG.choice(EVEN_DIMS)

        cap_h = _RNG.randint(1, min(3, h - 2))
        stem_w = _RNG.randint(1, min(3, w - 1))
        center_col = (w - 1) / 2.0
        stem_left_options = [
            left
            for left in range(w - stem_w + 1)
            if abs(left + (stem_w - 1) / 2.0 - center_col) <= 1.0
        ]
        stem_left = _RNG.choice(stem_left_options)

        mask = np.zeros((h, w), dtype=bool)
        for r in range(cap_h):
            if r == 0 or w <= 3 or _RNG.random() < 0.55:
                left, right = 0, w - 1
            else:
                dent_left = _RNG.randint(0, 1)
                dent_right = _RNG.randint(0, 1)
                if dent_left + dent_right >= w - 1:
                    dent_left = dent_right = 0
                left, right = dent_left, w - 1 - dent_right
            mask[r, left : right + 1] = True

        shoulder_depth = _RNG.randint(0, min(2, h - cap_h))
        shoulder_extra = _RNG.randint(0, min(2, (w - stem_w) // 2))
        for offset, r in enumerate(range(cap_h, h)):
            grow = shoulder_extra if offset < shoulder_depth else 0
            left = max(0, stem_left - grow)
            right = min(w - 1, stem_left + stem_w - 1 + grow)
            mask[r, left : right + 1] = True

        coords = np.argwhere(mask)
        if _dominant_direction(coords, allow_tie=True) == "up":
            return coords
    return None


def _expand(coords: list[tuple[int, int]], h: int, w: int, radius: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))
    return blocked


def _place_object(
    grid: np.ndarray,
    color: int,
    input_coords: np.ndarray,
    output_coords: np.ndarray,
    blocked_input: set[tuple[int, int]],
    blocked_output: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    input_min = input_coords.min(axis=0)
    input_max = input_coords.max(axis=0)
    output_min = output_coords.min(axis=0)
    output_max = output_coords.max(axis=0)

    row_lo = max(BORDER - int(input_min[0]), BORDER - int(output_min[0]))
    row_hi = min((h - 1 - BORDER) - int(input_max[0]), (h - 1 - BORDER) - int(output_max[0]))
    col_lo = max(BORDER - int(input_min[1]), BORDER - int(output_min[1]))
    col_hi = min((w - 1 - BORDER) - int(input_max[1]), (w - 1 - BORDER) - int(output_max[1]))
    if row_lo > row_hi or col_lo > col_hi:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        base_r = _RNG.randint(row_lo, row_hi)
        base_c = _RNG.randint(col_lo, col_hi)
        placed_input = [(base_r + int(r), base_c + int(c)) for r, c in input_coords]
        placed_output = [(base_r + int(r), base_c + int(c)) for r, c in output_coords]
        if any(cell in blocked_input for cell in placed_input):
            continue
        if any(cell in blocked_output for cell in placed_output):
            continue

        for r, c in placed_input:
            grid[r, c] = color
        blocked_input.update(_expand(placed_input, h, w, CLEARANCE))
        blocked_output.update(_expand(placed_output, h, w, CLEARANCE))
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = _RNG.randint(*GRID_HEIGHT_RANGE)
    w = _RNG.randint(*GRID_WIDTH_RANGE)
    bg = _RNG.choice(BG_PALETTE)
    palette = [color for color in range(10) if color != bg]
    object_count = _RNG.choice(OBJECT_COUNT_OPTIONS)
    colors = _RNG.sample(palette, object_count)

    directions = ["up", "right", "down", "left"]
    while len(directions) < object_count:
        directions.append(_RNG.choice(("up", "right", "down", "left")))
    _RNG.shuffle(directions)

    specs = []
    for color, direction in zip(colors, directions):
        base = _make_up_shape()
        if base is None:
            return None
        input_coords = _normalize(_rotate_coords(base, DIR_TO_TURNS[direction], bbox=_bbox(base)))
        if _dominant_direction(input_coords) != direction:
            return None
        output_coords = _rotate_coords(input_coords, DIR_TO_TURNS[direction], bbox=_bbox(input_coords))
        specs.append(
            {
                "color": color,
                "input_coords": input_coords,
                "output_coords": output_coords,
                "footprint": max(
                    int(input_coords[:, 0].ptp()) + int(input_coords[:, 1].ptp()),
                    int(output_coords[:, 0].ptp()) + int(output_coords[:, 1].ptp()),
                ),
            }
        )

    specs.sort(key=lambda item: item["footprint"], reverse=True)
    grid = np.full((h, w), bg, dtype=int)
    blocked_input: set[tuple[int, int]] = set()
    blocked_output: set[tuple[int, int]] = set()

    for spec in specs:
        if not _place_object(
            grid,
            int(spec["color"]),
            np.asarray(spec["input_coords"], dtype=int),
            np.asarray(spec["output_coords"], dtype=int),
            blocked_input,
            blocked_output,
        ):
            return None

    if len(_components(grid, bg)) != object_count:
        return None

    out = main(grid)
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
