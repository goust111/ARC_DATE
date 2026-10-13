from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
INPUT_COLORS = (1, 2, 3, 4)
AREA_BANDS = ((3, 4), (5, 6), (7, 8), (9, 10))
GRID_MIN = 11
GRID_MAX = 15
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _paint_component(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    out[coords[:, 0], coords[:, 1]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each sprite by cyclic area order and add diagonal corner markers from the opposite cyclic direction.

    Input structure:
    - The grid has black background and several monochrome 4-connected foreground sprites.
    - In the intended task instances there are four sprites, each with a distinct area and one of four distinct colors.
    - Every sprite is small, irregular, and well separated so its bbox corners are visually readable.

    Rule:
    - Find all non-background 4-connected monochrome sprites and sort them by pixel area from smallest to largest.
    - Recolor each sorted sprite with the previous sprite's original color in the cyclic order.
    - For each sprite bbox, paint the four outside diagonal corner cells with the next sprite's original color in the cyclic order.
    - Keep every sprite in place; skip a corner marker if that corner lies outside the grid.

    Key insight:
    - The same area ordering controls two different color transfers at once: body recoloring uses the previous color, while corner decoration uses the next color.
    """
    g = np.asarray(grid, dtype=int)
    comps = _find_components(g, bg=BG)
    if not comps:
        return np.zeros_like(g)

    ordered = sorted(comps, key=lambda comp: (int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1])))
    n = len(ordered)
    out = np.full_like(g, BG)

    for idx, comp in enumerate(ordered):
        prev_color = int(ordered[(idx - 1) % n]["color"])
        next_color = int(ordered[(idx + 1) % n]["color"])
        _paint_component(out, comp["coords"], prev_color)

        r0, c0, r1, c1 = comp["bbox"]
        corners = ((r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1))
        for r, c in corners:
            if _in_bounds(g.shape, int(r), int(c)):
                out[int(r), int(c)] = next_color

    return out


def _grow_connected_shape(area: int, bbox_h: int, bbox_w: int) -> set[tuple[int, int]] | None:
    cells = {(random.randrange(bbox_h), random.randrange(bbox_w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < bbox_h and 0 <= nc < bbox_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _generate_shape(area: int) -> np.ndarray:
    candidates = [
        (h, w)
        for h in range(2, 5)
        for w in range(2, 5)
        if area <= h * w and area / float(h * w) >= 0.45
    ]
    random.shuffle(candidates)
    for bbox_h, bbox_w in candidates:
        for _ in range(80):
            cells = _grow_connected_shape(area, bbox_h, bbox_w)
            if cells is None:
                continue
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            if min(rows) != 0 or max(rows) != bbox_h - 1 or min(cols) != 0 or max(cols) != bbox_w - 1:
                continue
            if len(cells) == bbox_h * bbox_w:
                continue
            patch = np.zeros((bbox_h, bbox_w), dtype=int)
            for r, c in cells:
                patch[r, c] = 1
            return patch
    raise RuntimeError(f"failed to generate shape for area {area}")


def _expanded_zone(top: int, left: int, patch: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = patch.shape
    grid_h, grid_w = shape
    return (
        max(0, top - 1),
        max(0, left - 1),
        min(grid_h - 1, top + h),
        min(grid_w - 1, left + w),
    )


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    rr, cc = np.nonzero(patch)
    grid[top + rr, left + cc] = int(color)


def _place_objects(objects: list[dict], grid_h: int, grid_w: int) -> bool:
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    edge_count = random.choices([0, 1, 2], weights=[3, 5, 2], k=1)[0]
    edge_indices = set(random.sample(range(len(objects)), k=edge_count))

    placement_order = sorted(range(len(objects)), key=lambda idx: objects[idx]["patch"].shape[0] * objects[idx]["patch"].shape[1], reverse=True)
    for order_idx, obj_idx in enumerate(placement_order):
        obj = objects[obj_idx]
        patch = obj["patch"]
        ph, pw = patch.shape
        positions = []
        for top in range(grid_h - ph + 1):
            for left in range(grid_w - pw + 1):
                needs_edge = obj_idx in edge_indices
                if needs_edge and not (top == 0 or left == 0 or top + ph == grid_h or left + pw == grid_w):
                    continue
                zr0, zc0, zr1, zc1 = _expanded_zone(top, left, patch, (grid_h, grid_w))
                if blocked[zr0 : zr1 + 1, zc0 : zc1 + 1].any():
                    continue
                positions.append((top, left))
        if not positions:
            if obj_idx in edge_indices:
                return False
            return False
        if order_idx == 0 and obj_idx not in edge_indices and random.random() < 0.35:
            cornerish = [pos for pos in positions if pos[0] <= 1 or pos[1] <= 1 or pos[0] + ph >= grid_h - 1 or pos[1] + pw >= grid_w - 1]
            if cornerish:
                positions = cornerish
        top, left = random.choice(positions)
        obj["top"] = int(top)
        obj["left"] = int(left)
        zr0, zc0, zr1, zc1 = _expanded_zone(top, left, patch, (grid_h, grid_w))
        blocked[zr0 : zr1 + 1, zc0 : zc1 + 1] = True
    return True


def _render_input(objects: list[dict], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for obj in objects:
        _paste_patch(grid, obj["patch"], int(obj["top"]), int(obj["left"]), int(obj["color"]))
    return grid


def _render_expected(objects: list[dict], grid_h: int, grid_w: int) -> np.ndarray:
    out = np.zeros((grid_h, grid_w), dtype=int)
    ordered = sorted(objects, key=lambda obj: obj["area"])
    n = len(ordered)
    for idx, obj in enumerate(ordered):
        prev_color = int(ordered[(idx - 1) % n]["color"])
        next_color = int(ordered[(idx + 1) % n]["color"])

        rr, cc = np.nonzero(obj["patch"])
        out[int(obj["top"]) + rr, int(obj["left"]) + cc] = prev_color

        top = int(obj["top"])
        left = int(obj["left"])
        ph, pw = obj["patch"].shape
        r0, c0, r1, c1 = top, left, top + ph - 1, left + pw - 1
        corners = ((r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1))
        for r, c in corners:
            if 0 <= r < grid_h and 0 <= c < grid_w:
                out[r, c] = next_color
    return out


def _is_valid_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    comps = _find_components(inp, bg=BG)
    if len(comps) != 4:
        return False
    if sorted(comp["color"] for comp in comps) != list(INPUT_COLORS):
        return False
    areas = [int(comp["area"]) for comp in comps]
    if len(set(areas)) != 4:
        return False
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def _sample_areas() -> list[int]:
    return [random.randint(lo, hi) for lo, hi in AREA_BANDS]


def _attempt_generate() -> np.ndarray | None:
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)
    colors = list(INPUT_COLORS)
    random.shuffle(colors)
    areas = _sample_areas()

    objects = []
    for area, color in zip(areas, colors):
        patch = _generate_shape(area)
        objects.append({"area": int(area), "color": int(color), "patch": patch})

    if not _place_objects(objects, grid_h, grid_w):
        return None

    inp = _render_input(objects, grid_h, grid_w)
    expected = _render_expected(objects, grid_h, grid_w)
    if not _is_valid_instance(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
