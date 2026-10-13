from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


RED = 2
YELLOW = 4
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_SHAPES = {
    "up": [(0, 1), (1, 0), (1, 2)],
    "down": [(0, 0), (0, 2), (1, 1)],
    "left": [(0, 1), (1, 0), (2, 1)],
    "right": [(0, 0), (1, 1), (2, 0)],
}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _decode_arrow_direction(yellow_pixels: list[tuple[int, int]]) -> str:
    rows = Counter(r for r, _ in yellow_pixels)
    cols = Counter(c for _, c in yellow_pixels)

    if len(rows) == 2 and sorted(rows.values()) == [1, 2]:
        tip_row = int(next(r for r, count in rows.items() if count == 1))
        wing_row = int(next(r for r, count in rows.items() if count == 2))
        tip_cols = [c for r, c in yellow_pixels if r == tip_row]
        wing_cols = sorted(c for r, c in yellow_pixels if r == wing_row)
        if len(tip_cols) == 1 and len(wing_cols) == 2 and wing_cols[1] - wing_cols[0] == 2:
            if int(tip_cols[0]) == int(wing_cols[0] + 1):
                return "up" if tip_row < wing_row else "down"

    if len(cols) == 2 and sorted(cols.values()) == [1, 2]:
        tip_col = int(next(c for c, count in cols.items() if count == 1))
        wing_col = int(next(c for c, count in cols.items() if count == 2))
        tip_rows = [r for r, c in yellow_pixels if c == tip_col]
        wing_rows = sorted(r for r, c in yellow_pixels if c == wing_col)
        if len(tip_rows) == 1 and len(wing_rows) == 2 and wing_rows[1] - wing_rows[0] == 2:
            if int(tip_rows[0]) == int(wing_rows[0] + 1):
                return "left" if tip_col < wing_col else "right"

    raise ValueError(f"invalid arrow pixels: {yellow_pixels}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each connected object along its embedded arrow.

    The background is the most common color. Each foreground component contains
    exactly one yellow 3-pixel arrow and zero or more red marker pixels. The
    arrow direction chooses the translation axis, and the red-pixel count is the
    translation distance. Objects are applied in input top-left order, so later
    objects overwrite earlier ones on overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _background_color(g)

    objects = []
    for coords in _components(g != bg):
        pixels = [(int(r), int(c), int(g[int(r), int(c)])) for r, c in coords]
        yellow_pixels = [(r, c) for r, c, color in pixels if color == YELLOW]
        if len(yellow_pixels) != 3:
            continue
        direction = _decode_arrow_direction(yellow_pixels)
        red_count = sum(1 for _r, _c, color in pixels if color == RED)
        top_left = (min(r for r, _c, _color in pixels), min(c for _r, c, _color in pixels))
        objects.append(
            {
                "pixels": pixels,
                "direction": direction,
                "distance": int(red_count),
                "top_left": top_left,
            }
        )

    objects.sort(key=lambda obj: obj["top_left"])

    out = np.full((h, w), int(bg), dtype=int)
    for obj in objects:
        dr, dc = DIR_VECTORS[obj["direction"]]
        shift_r = int(dr * obj["distance"])
        shift_c = int(dc * obj["distance"])
        for r, c, color in obj["pixels"]:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(color)
    return out


# --------------- generator (Codex) ---------------

BG = 0
BASE_COLORS = [1, 3, 5, 6, 7, 8, 9]
GRID_HEIGHT_RANGE = (16, 26)
GRID_WIDTH_RANGE = (16, 26)
OBJECT_COUNT_RANGE = (2, 4)
CORE_HEIGHT_RANGE = (5, 7)
CORE_WIDTH_RANGE = (5, 8)
SIDE_MARGIN_CHOICES = [0, 0, 1, 1, 2]
RED_COUNT_CAP = 5
OVERLAP_PROB = 0.35
GEN_ATTEMPTS = 400


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(int(bounds[0]), int(bounds[1]))


def _choose_arrow_cells(direction: str, core_top: int, core_left: int, core_h: int, core_w: int) -> list[tuple[int, int]] | None:
    offsets = ARROW_SHAPES[direction]
    max_r = max(r for r, _ in offsets)
    max_c = max(c for _, c in offsets)

    row_starts = range(int(core_top + 1), int(core_top + core_h - 1 - max_r))
    col_starts = range(int(core_left + 1), int(core_left + core_w - 1 - max_c))
    origins = [(r0, c0) for r0 in row_starts for c0 in col_starts]
    if not origins:
        return None
    origin_r, origin_c = random.choice(origins)
    return [(int(origin_r + dr), int(origin_c + dc)) for dr, dc in offsets]


def _build_object_patch(base_color: int, direction: str) -> dict[str, object] | None:
    core_h = _randint(CORE_HEIGHT_RANGE)
    core_w = _randint(CORE_WIDTH_RANGE)
    extra_top = random.choice(SIDE_MARGIN_CHOICES)
    extra_bottom = random.choice(SIDE_MARGIN_CHOICES)
    extra_left = random.choice(SIDE_MARGIN_CHOICES)
    extra_right = random.choice(SIDE_MARGIN_CHOICES)

    patch_h = int(core_h + extra_top + extra_bottom)
    patch_w = int(core_w + extra_left + extra_right)
    mask = np.zeros((patch_h, patch_w), dtype=bool)

    core_top = int(extra_top)
    core_left = int(extra_left)
    mask[core_top : core_top + core_h, core_left : core_left + core_w] = True

    if extra_top > 0 and random.random() < 0.7:
        span = random.randint(1, max(1, core_w - 2))
        start = random.randint(1, core_w - span - 1)
        mask[:extra_top, core_left + start : core_left + start + span] = True
    if extra_bottom > 0 and random.random() < 0.7:
        span = random.randint(1, max(1, core_w - 2))
        start = random.randint(1, core_w - span - 1)
        mask[core_top + core_h :, core_left + start : core_left + start + span] = True
    if extra_left > 0 and random.random() < 0.7:
        span = random.randint(1, max(1, core_h - 2))
        start = random.randint(1, core_h - span - 1)
        mask[core_top + start : core_top + start + span, :extra_left] = True
    if extra_right > 0 and random.random() < 0.7:
        span = random.randint(1, max(1, core_h - 2))
        start = random.randint(1, core_h - span - 1)
        mask[core_top + start : core_top + start + span, core_left + core_w :] = True

    arrow_cells = _choose_arrow_cells(direction, core_top, core_left, core_h, core_w)
    if arrow_cells is None:
        return None

    interior_cells = [
        (int(r), int(c))
        for r in range(int(core_top + 1), int(core_top + core_h - 1))
        for c in range(int(core_left + 1), int(core_left + core_w - 1))
    ]
    interior_set = set(interior_cells)
    arrow_set = set(arrow_cells)
    if not arrow_set.issubset(interior_set):
        return None

    red_candidates = sorted(interior_set - arrow_set)
    max_red = min(int(RED_COUNT_CAP), int(len(red_candidates)))
    if max_red < 1:
        return None
    red_count = random.randint(1, max_red)
    red_cells = random.sample(red_candidates, red_count)

    patch = np.full((patch_h, patch_w), int(BG), dtype=int)
    patch[mask] = int(base_color)
    for r, c in red_cells:
        patch[int(r), int(c)] = int(RED)
    for r, c in arrow_cells:
        patch[int(r), int(c)] = int(YELLOW)

    coords = np.argwhere(mask)
    colors = [(int(r), int(c), int(patch[int(r), int(c)])) for r, c in coords]
    return {
        "patch": patch,
        "pixels": colors,
        "shape_cells": [(int(r), int(c)) for r, c in coords],
        "direction": direction,
        "distance": int(red_count),
        "size": int(len(coords)),
        "shape": patch.shape,
    }


def _placement_bounds(grid_h: int, grid_w: int, obj: dict[str, object]) -> tuple[range, range] | None:
    patch_h, patch_w = obj["shape"]
    direction = obj["direction"]
    distance = int(obj["distance"])
    dr, dc = DIR_VECTORS[direction]

    row_lo = 1
    row_hi = int(grid_h - patch_h - 1)
    col_lo = 1
    col_hi = int(grid_w - patch_w - 1)

    if dr == 1:
        row_hi = min(row_hi, int(grid_h - patch_h - distance))
    elif dr == -1:
        row_lo = max(row_lo, int(distance))
    if dc == 1:
        col_hi = min(col_hi, int(grid_w - patch_w - distance))
    elif dc == -1:
        col_lo = max(col_lo, int(distance))

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return range(int(row_lo), int(row_hi) + 1), range(int(col_lo), int(col_hi) + 1)


def _mark_blocked(blocked: np.ndarray, abs_cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in abs_cells:
        for dr, dc in DIR8:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True


def _pairwise_overlap_sizes(cell_sets: list[set[tuple[int, int]]]) -> list[int]:
    overlaps: list[int] = []
    for i in range(len(cell_sets)):
        for j in range(i + 1, len(cell_sets)):
            overlaps.append(int(len(cell_sets[i] & cell_sets[j])))
    return overlaps


def _place_objects(grid_h: int, grid_w: int, objects: list[dict[str, object]]) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), int(BG), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict[str, object]] = []

    order = objects[:]
    random.shuffle(order)
    for obj in order:
        bounds = _placement_bounds(grid_h, grid_w, obj)
        if bounds is None:
            return None
        row_range, col_range = bounds
        positions = [(r0, c0) for r0 in row_range for c0 in col_range]
        random.shuffle(positions)

        chosen = None
        chosen_abs_cells = None
        for r0, c0 in positions:
            abs_cells = [(int(r0 + r), int(c0 + c)) for r, c in obj["shape_cells"]]
            if any(bool(blocked[rr, cc]) for rr, cc in abs_cells):
                continue
            chosen = (int(r0), int(c0))
            chosen_abs_cells = abs_cells
            break
        if chosen is None or chosen_abs_cells is None:
            return None

        r0, c0 = chosen
        abs_pixels = []
        for r, c, color in obj["pixels"]:
            rr = int(r0 + r)
            cc = int(c0 + c)
            grid[rr, cc] = int(color)
            abs_pixels.append((rr, cc, int(color)))

        _mark_blocked(blocked, chosen_abs_cells)
        placed.append(
            {
                "top_left": (int(r0), int(c0)),
                "direction": obj["direction"],
                "distance": int(obj["distance"]),
                "pixels": abs_pixels,
            }
        )

    translated_sets: list[set[tuple[int, int]]] = []
    for obj in placed:
        dr, dc = DIR_VECTORS[obj["direction"]]
        shift_r = int(dr * obj["distance"])
        shift_c = int(dc * obj["distance"])
        translated_sets.append({(int(r + shift_r), int(c + shift_c)) for r, c, _color in obj["pixels"]})

    overlaps = _pairwise_overlap_sizes(translated_sets)
    overlap_pairs = sum(1 for size in overlaps if size > 0)
    if overlap_pairs > 2 or any(size > 8 for size in overlaps):
        return None
    if random.random() < OVERLAP_PROB and overlap_pairs == 0:
        return None

    if _background_color(grid) != BG:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(BASE_COLORS, object_count)
    directions = [random.choice(list(DIR_VECTORS)) for _ in range(object_count)]

    objects: list[dict[str, object]] = []
    for color, direction in zip(colors, directions):
        obj = _build_object_patch(int(color), direction)
        if obj is None:
            return None
        objects.append(obj)

    total_area = sum(int(obj["size"]) for obj in objects)
    min_h = max(int(obj["shape"][0]) + 3 for obj in objects)
    min_w = max(int(obj["shape"][1]) + 3 for obj in objects)

    h_lo = max(GRID_HEIGHT_RANGE[0], min_h, int(np.ceil(np.sqrt(total_area * 2.2))))
    w_lo = max(GRID_WIDTH_RANGE[0], min_w, int(np.ceil(np.sqrt(total_area * 2.2))))
    if h_lo > GRID_HEIGHT_RANGE[1] or w_lo > GRID_WIDTH_RANGE[1]:
        return None

    grid_h = random.randint(int(h_lo), GRID_HEIGHT_RANGE[1])
    grid_w = random.randint(int(w_lo), GRID_WIDTH_RANGE[1])
    if grid_h * grid_w <= total_area * 2:
        return None

    return _place_objects(int(grid_h), int(grid_w), objects)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
