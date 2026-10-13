from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
NON_BG_COLORS = tuple(range(1, 10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- solver ---------------


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clone each instructed monochrome object one bbox-width/height away in the
    instruction direction, recolor the copy with the instruction target color,
    and erase all instruction pairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects: list[dict] = []
    instructions: list[list[tuple[int, int]]] = []
    for coords in _components(g != BG):
        colors = {int(g[r, c]) for r, c in coords}
        area = len(coords)
        if area == 2:
            assert len(colors) == 2, "instruction pair must be two different colors"
            (r0, c0), (r1, c1) = coords
            assert abs(r0 - r1) + abs(c0 - c1) == 1, "instruction pixels must be adjacent"
            instructions.append(coords)
        else:
            assert 4 <= area <= 8, "object area must be between 4 and 8"
            assert len(colors) == 1, "objects must be monochrome"
            color = next(iter(colors))
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )

    assert objects, "expected at least one source object"
    assert instructions, "expected at least one instruction pair"

    object_by_color: dict[int, dict] = {}
    for obj in objects:
        color = int(obj["color"])
        assert color not in object_by_color, "each object color must be unique"
        object_by_color[color] = obj

    out = g.copy()
    for pair in instructions:
        for r, c in pair:
            out[r, c] = BG

    used_source_colors: set[int] = set()
    h, w = g.shape
    for pair in instructions:
        a, b = pair
        ca = int(g[a[0], a[1]])
        cb = int(g[b[0], b[1]])

        candidates = []
        if ca in object_by_color and cb not in object_by_color:
            candidates.append((a, b, ca, cb))
        if cb in object_by_color and ca not in object_by_color:
            candidates.append((b, a, cb, ca))
        assert len(candidates) == 1, "exactly one endpoint must match an object color"

        source_cell, target_cell, source_color, target_color = candidates[0]
        assert target_color != BG and target_color != source_color
        assert source_color not in used_source_colors, "each object may have at most one instruction"
        used_source_colors.add(source_color)

        obj = object_by_color[source_color]
        r0, c0, r1, c1 = obj["bbox"]
        dr = target_cell[0] - source_cell[0]
        dc = target_cell[1] - source_cell[1]
        assert abs(dr) + abs(dc) == 1

        shift_r = dr * (r1 - r0 + 2)
        shift_c = dc * (c1 - c0 + 2)

        for r, c in obj["coords"]:
            nr = r + shift_r
            nc = c + shift_c
            assert 0 <= nr < h and 0 <= nc < w, "copy must stay inside the grid"
            assert out[nr, nc] == BG, "copy region must be empty after erasing instructions"

        for r, c in obj["coords"]:
            out[r + shift_r, c + shift_c] = target_color

    return out


# --------------- generator ---------------

GRID_SIDE_MIN = 17
GRID_SIDE_MAX = 24
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 4
GEN_ATTEMPTS = 300
SHAPE_ATTEMPTS = 100
OBJECT_PLACEMENT_ATTEMPTS = 120
PAIR_PLACEMENT_ATTEMPTS = 120


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_shape() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(1, 4)
        width = random.randint(1, 4)
        if height * width < 4:
            continue

        target_area = random.randint(4, min(8, height * width))
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != target_area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return _trim_mask(mask)

    raise RuntimeError("failed to build a random connected shape")


def _object_layout(
    mask: np.ndarray, direction: tuple[int, int] | None
) -> tuple[int, int, tuple[int, int]]:
    shape_h, shape_w = mask.shape
    if direction is None:
        return shape_h, shape_w, (0, 0)
    if direction == (-1, 0):
        return 2 * shape_h + 1, shape_w, (shape_h + 1, 0)
    if direction == (1, 0):
        return 2 * shape_h + 1, shape_w, (0, 0)
    if direction == (0, -1):
        return shape_h, 2 * shape_w + 1, (0, shape_w + 1)
    if direction == (0, 1):
        return shape_h, 2 * shape_w + 1, (0, 0)
    raise AssertionError("unexpected direction")


def _instruction_layout(direction: tuple[int, int]) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    if direction == (-1, 0):
        return 2, 1, (1, 0), (0, 0)
    if direction == (1, 0):
        return 2, 1, (0, 0), (1, 0)
    if direction == (0, -1):
        return 1, 2, (0, 1), (0, 0)
    if direction == (0, 1):
        return 1, 2, (0, 0), (0, 1)
    raise AssertionError("unexpected direction")


def _rect_clear(mask: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    return not np.any(mask[r0 : r0 + height, c0 : c0 + width])


def _mark_reserved(mask: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + height + 1)
    cc1 = min(w, c0 + width + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = color


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    instruct_all = random.random() < 0.5
    instruction_count = object_count if instruct_all else random.randint(1, object_count - 1)

    palette = list(NON_BG_COLORS)
    random.shuffle(palette)
    source_colors = palette[:object_count]
    target_colors = palette[object_count : object_count + instruction_count]

    instructed_indices = set(random.sample(range(object_count), instruction_count))
    specs: list[dict] = []
    target_i = 0
    for index, color in enumerate(source_colors):
        mask = _random_shape()
        direction = random.choice(DIRS4) if index in instructed_indices else None
        target_color = target_colors[target_i] if direction is not None else None
        if direction is not None:
            target_i += 1
        comp_h, comp_w, src_off = _object_layout(mask, direction)
        specs.append(
            {
                "color": color,
                "mask": mask,
                "direction": direction,
                "target_color": target_color,
                "comp_h": comp_h,
                "comp_w": comp_w,
                "src_off": src_off,
            }
        )

    max_comp_h = max(spec["comp_h"] for spec in specs)
    max_comp_w = max(spec["comp_w"] for spec in specs)
    min_h = max(GRID_SIDE_MIN, max_comp_h + 4, 4 * object_count + 2)
    min_w = max(GRID_SIDE_MIN, max_comp_w + 4, 4 * object_count + 2)
    if min_h > GRID_SIDE_MAX or min_w > GRID_SIDE_MAX:
        return None

    height = random.randint(min_h, GRID_SIDE_MAX)
    width = random.randint(min_w, GRID_SIDE_MAX)
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros_like(grid, dtype=bool)

    for spec in sorted(specs, key=lambda item: item["comp_h"] * item["comp_w"], reverse=True):
        placed = False
        for _ in range(OBJECT_PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - spec["comp_h"])
            c0 = random.randint(0, width - spec["comp_w"])
            if not _rect_clear(reserved, r0, c0, spec["comp_h"], spec["comp_w"]):
                continue

            src_top = r0 + spec["src_off"][0]
            src_left = c0 + spec["src_off"][1]
            _paint_mask(grid, src_top, src_left, spec["mask"], int(spec["color"]))
            _mark_reserved(reserved, r0, c0, spec["comp_h"], spec["comp_w"])
            spec["src_top"] = src_top
            spec["src_left"] = src_left
            placed = True
            break
        if not placed:
            return None

    for spec in specs:
        if spec["direction"] is None:
            continue
        pair_h, pair_w, src_off, dst_off = _instruction_layout(spec["direction"])
        placed = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - pair_h)
            c0 = random.randint(0, width - pair_w)
            if not _rect_clear(reserved, r0, c0, pair_h, pair_w):
                continue

            sr, sc = r0 + src_off[0], c0 + src_off[1]
            tr, tc = r0 + dst_off[0], c0 + dst_off[1]
            grid[sr, sc] = int(spec["color"])
            grid[tr, tc] = int(spec["target_color"])
            _mark_reserved(reserved, r0, c0, pair_h, pair_w)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
