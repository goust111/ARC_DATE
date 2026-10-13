from __future__ import annotations

import random
from typing import Iterable

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver ---------------


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _square_border_mask(side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], set[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int, set[int]]] = []

    max_r0 = max(1, h // 3 + 1)
    max_c0 = max(1, w // 3 + 1)
    for r0 in range(max_r0):
        for c0 in range(max_c0):
            border_color = int(grid[r0, c0])
            if border_color == bg:
                continue

            max_side = min(h - r0, w - c0)
            for side in range(3, max_side + 1):
                patch = grid[r0 : r0 + side, c0 : c0 + side]
                border_mask = _square_border_mask(side)
                if not np.all(patch[border_mask] == border_color):
                    continue

                interior = patch[1:-1, 1:-1]
                if interior.size == 0:
                    continue
                target_colors = set(map(int, np.unique(interior))) - {bg}
                if not target_colors:
                    continue

                candidates.append((r0 + c0, r0, c0, side, target_colors))
                break

    if not candidates:
        raise AssertionError("main: no legend frame found")

    _score, r0, c0, side, target_colors = min(candidates, key=lambda item: item[:4])
    return (r0, c0, r0 + side - 1, c0 + side - 1), target_colors


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            comp: list[tuple[int, int]] = []
            queue = [(r, c)]
            seen[r, c] = True
            head = 0
            while head < len(queue):
                cr, cc = queue[head]
                head += 1
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            comps.append(comp)

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove every non-legend 4-connected component whose color set contains every
    target color shown inside the framed legend in the upper-left area.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    (r0, c0, r1, c1), target_colors = _find_legend(g, bg)

    mask = g != bg
    mask[r0 : r1 + 1, c0 : c1 + 1] = False

    out = g.copy()
    for comp in _components(mask):
        comp_colors = {int(g[r, c]) for r, c in comp}
        if target_colors.issubset(comp_colors):
            for r, c in comp:
                out[r, c] = bg

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
GRID_H_RANGE = (14, 23)
GRID_W_RANGE = (16, 27)
TARGET_COUNT_RANGE = (2, 4)
OBJECT_COUNT_RANGE = (3, 6)
OBJECT_SIZE_RANGE = (4, 10)
SHAPE_BOX_H_RANGE = (2, 5)
SHAPE_BOX_W_RANGE = (2, 6)
SHAPE_ATTEMPTS = 50
PLACEMENT_ATTEMPTS = 120
COLOR_POOL = tuple(range(10))


def _normalize_cells(cells: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = list(cells)
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_shape(rng: random.Random, min_cells: int, max_cells: int) -> list[tuple[int, int]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = rng.randint(*SHAPE_BOX_H_RANGE)
        box_w = rng.randint(*SHAPE_BOX_W_RANGE)
        capacity = box_h * box_w
        if capacity < min_cells:
            continue

        target_size = rng.randint(min_cells, min(max_cells, capacity))
        cells = {(rng.randrange(box_h), rng.randrange(box_w))}

        stalled = 0
        while len(cells) < target_size and stalled < target_size * 12:
            cr, cc = rng.choice(tuple(cells))
            options = []
            for dr, dc in NEIGHBORS4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    options.append((nr, nc))
            if options:
                cells.add(rng.choice(options))
                stalled = 0
            else:
                stalled += 1

        if len(cells) != target_size:
            continue

        norm = _normalize_cells(cells)
        rows = {r for r, _c in norm}
        cols = {c for _r, c in norm}
        if len(rows) == 1 and len(cols) == 1:
            continue
        return norm

    return None


def _reserve_buffer(blocked: np.ndarray, coords: Iterable[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _place_shape(
    rng: random.Random,
    blocked: np.ndarray,
    shape: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    shape_h = max(r for r, _c in shape) + 1
    shape_w = max(c for _r, c in shape) + 1

    if shape_h > h or shape_w > w:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        top = rng.randint(0, h - shape_h)
        left = rng.randint(0, w - shape_w)
        coords = [(top + r, left + c) for r, c in shape]
        if any(blocked[r, c] for r, c in coords):
            continue
        return coords

    return None


def _assign_colors(
    rng: random.Random,
    cell_count: int,
    required_colors: list[int],
    allowed_colors: list[int],
) -> list[int]:
    palette = list(dict.fromkeys(required_colors))
    extras = [color for color in allowed_colors if color not in palette]

    if len(palette) < 2:
        if not extras:
            raise AssertionError("generator: need a second visible color")
        palette.append(rng.choice(extras))
        extras = [color for color in extras if color not in palette]

    max_extra = min(2, cell_count - len(palette), len(extras))
    if max_extra > 0:
        extra_count = rng.randint(0, max_extra)
        if extra_count:
            palette.extend(rng.sample(extras, extra_count))

    colors = palette.copy()
    while len(colors) < cell_count:
        colors.append(rng.choice(palette))
    rng.shuffle(colors)
    return colors


def _object_spec(
    rng: random.Random,
    target_colors: list[int],
    usable_colors: list[int],
    removable: bool,
) -> tuple[list[int], list[int]]:
    if removable:
        required = list(target_colors)
        allowed = usable_colors
    else:
        missing = rng.choice(target_colors)
        present_targets = [color for color in target_colors if color != missing]
        subset_size = rng.randint(1, len(present_targets))
        required = rng.sample(present_targets, subset_size)
        allowed = [color for color in usable_colors if color != missing]
    return required, allowed


def _validate_instance(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    bg: int,
    objects: list[tuple[list[tuple[int, int]], bool]],
) -> bool:
    if _background_color(grid) != bg:
        return False

    out = main(grid)
    r0, c0, r1, c1 = legend_bbox
    if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], grid[r0 : r1 + 1, c0 : c1 + 1]):
        return False

    removed_seen = False
    kept_seen = False
    for coords, removable in objects:
        if removable:
            removed_seen = True
            if any(int(out[r, c]) != bg for r, c in coords):
                return False
        else:
            kept_seen = True
            if any(int(out[r, c]) != int(grid[r, c]) for r, c in coords):
                return False

    return removed_seen and kept_seen and bool(np.any(out != grid))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(COLOR_POOL)
    grid = np.full((h, w), bg, dtype=int)

    usable_colors = [color for color in COLOR_POOL if color != bg]
    target_count = rng.randint(*TARGET_COUNT_RANGE)
    target_colors = rng.sample(usable_colors, target_count)
    frame_color = rng.choice([color for color in usable_colors if color not in target_colors])
    object_colors = [color for color in usable_colors if color != frame_color]

    inner_side = max(2, min(4, int(np.ceil(np.sqrt(target_count + rng.randint(0, 3))))))
    side = inner_side + 2
    if side >= h or side >= w:
        return None

    border_mask = _square_border_mask(side)
    grid[:side, :side][border_mask] = frame_color

    interior_cells = [(r, c) for r in range(1, side - 1) for c in range(1, side - 1)]
    rng.shuffle(interior_cells)
    for color, (r, c) in zip(target_colors, interior_cells):
        grid[r, c] = color
    extra_interior = interior_cells[len(target_colors) :]
    duplicate_count = rng.randint(0, min(len(extra_interior), target_count))
    for r, c in extra_interior[:duplicate_count]:
        grid[r, c] = rng.choice(target_colors)

    blocked = np.zeros((h, w), dtype=bool)
    legend_coords = [(r, c) for r in range(side) for c in range(side)]
    _reserve_buffer(blocked, legend_coords)

    total_objects = rng.randint(*OBJECT_COUNT_RANGE)
    removable_count = rng.randint(1, total_objects - 1)
    removable_flags = [True] * removable_count + [False] * (total_objects - removable_count)
    rng.shuffle(removable_flags)

    placed_objects: list[tuple[list[tuple[int, int]], bool]] = []
    for removable in removable_flags:
        required_colors, allowed_colors = _object_spec(rng, target_colors, object_colors, removable)
        min_cells = max(len(required_colors) + 1, OBJECT_SIZE_RANGE[0])
        max_cells = max(min_cells, OBJECT_SIZE_RANGE[1])

        shape = _random_shape(rng, min_cells, max_cells)
        if shape is None:
            return None
        coords = _place_shape(rng, blocked, shape)
        if coords is None:
            return None

        colors = _assign_colors(rng, len(coords), required_colors, allowed_colors)
        for (r, c), color in zip(coords, colors):
            grid[r, c] = color
        _reserve_buffer(blocked, coords)
        placed_objects.append((coords, removable))

    legend_bbox = (0, 0, side - 1, side - 1)
    if not _validate_instance(grid, legend_bbox, bg, placed_objects):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")
