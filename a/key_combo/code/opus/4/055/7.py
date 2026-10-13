from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS) not in sys.path:
    sys.path.append(str(_UTILS))

from arc_utils import Color, bbox_w, components_mask, find_components, get_bg


WORK_SHAPE = (28, 28)
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DISTRACTOR_BASES = [
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
]
GEN_ATTEMPTS = 600


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_objects(objects: list[np.ndarray]) -> list[np.ndarray]:
    min_r = min(int(obj[:, 0].min()) for obj in objects)
    min_c = min(int(obj[:, 1].min()) for obj in objects)
    delta = np.array([min_r, min_c], dtype=int)
    return [obj - delta for obj in objects]


def _reflect_h(objects: list[np.ndarray]) -> list[np.ndarray]:
    max_c = max(int(obj[:, 1].max()) for obj in objects)
    mirrored = [np.column_stack([obj[:, 0], max_c - obj[:, 1]]).astype(int) for obj in objects]
    return _normalize_objects(mirrored)


def _coords_from_pairs(pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.array(sorted(set((int(r), int(c)) for r, c in pairs)), dtype=int)


def _shape_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _mask_to_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def _transform_shape(coords: np.ndarray, rot: int, mirror: bool) -> np.ndarray:
    mask = _shape_to_mask(coords)
    if mirror:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, int(rot))
    return _mask_to_coords(mask)


def _paint_objects(shape: tuple[int, int], objects: list[np.ndarray], colors: list[int], bg: int = 0) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for coords, color in zip(objects, colors):
        grid[coords[:, 0], coords[:, 1]] = int(color)
    return grid


def _shift_object_coords(coords: np.ndarray, shift: int) -> np.ndarray:
    return coords + np.array([0, int(shift)], dtype=int)


def _shifted_grid_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=8, monochromatic=False)
    out = np.full_like(g, int(bg))
    occupied = np.zeros(g.shape, dtype=bool)
    h, w = g.shape
    for obj in objects:
        shift = bbox_w(obj["bbox"])
        dst = obj["coords"] + np.array([0, shift], dtype=int)
        assert bool((0 <= dst[:, 0]).all() and (dst[:, 0] < h).all())
        assert bool((0 <= dst[:, 1]).all() and (dst[:, 1] < w).all())
        assert not bool(occupied[dst[:, 0], dst[:, 1]].any())
        out[dst[:, 0], dst[:, 1]] = g[obj["coords"][:, 0], obj["coords"][:, 1]]
        occupied[dst[:, 0], dst[:, 1]] = True
    return out


def _enclosed_regions(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    holes = []
    h, w = grid.shape
    for coords in components_mask(grid == bg, connectivity=4):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in coords):
            continue
        holes.append(coords)
    return holes


def _fill_enclosed_regions(grid: np.ndarray, bg: int) -> np.ndarray:
    out = grid.copy()
    for coords in _enclosed_regions(out, bg):
        out[coords[:, 0], coords[:, 1]] = Color.GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Shift each object right by its own width, then fill the newly enclosed background with grey.

    Input structure:
    - The grid has a black background and 3-5 disconnected 8-connected foreground objects in distinct colors.
    - Each object has enough empty space on its right to move by its own bounding-box width.
    - Before any movement, the foreground does not enclose any background region.

    Rule:
    - Find every 8-connected foreground object and measure its bounding-box width.
    - Erase the original foreground and redraw each object shifted right by exactly that width, preserving its shape and colors.
    - After all shifts, treat all foreground colors as one combined obstacle mask.
    - Any black 4-connected region that does not touch the grid edge is enclosed; recolor all such pixels grey.

    Key insight:
    - The hole to fill does not exist in the input. It emerges only after different objects move by different distances set by their own widths.
    """
    shifted = _shifted_grid_only(np.asarray(grid, dtype=int))
    return _fill_enclosed_regions(shifted, get_bg(shifted))


def _template_c_bar() -> tuple[list[np.ndarray], int]:
    candidates = [(4, 5), (5, 5)]
    h, w = random.choice(candidates)
    stub_row = random.choice([1, h - 2])
    c_shape = _coords_from_pairs(
        [(0, c) for c in range(w)]
        + [(h - 1, c) for c in range(w)]
        + [(r, 0) for r in range(1, h - 1)]
    )
    bar = _coords_from_pairs([(r, w - 1) for r in range(1, h - 1)] + [(stub_row, w), (stub_row, w + 1)])
    objects = _normalize_objects([c_shape, bar])
    if random.random() < 0.5:
        objects = _reflect_h(objects)
    return objects, 2


def _template_four_sides() -> tuple[list[np.ndarray], int]:
    h = random.choice([4, 5, 6])
    w = random.choice([5, 6])
    left_stub = random.choice([1, h - 2])
    right_stub = random.choice([1, h - 2])
    top = _coords_from_pairs([(0, c) for c in range(w)])
    bottom = _coords_from_pairs([(h - 1, c) for c in range(w)])
    left = _coords_from_pairs([(r, 0) for r in range(1, h - 1)] + [(left_stub, -1), (left_stub, -2)])
    right = _coords_from_pairs([(r, w - 1) for r in range(1, h - 1)] + [(right_stub, w), (right_stub, w + 1)])
    objects = _normalize_objects([top, bottom, left, right])
    if random.random() < 0.5:
        objects = _reflect_h(objects)
    return objects, 4


def _random_distractor() -> np.ndarray:
    base = _coords_from_pairs(random.choice(DISTRACTOR_BASES))
    return _transform_shape(base, rot=random.randrange(4), mirror=bool(random.getrandbits(1)))


def _placeable(existing: list[np.ndarray], candidate: np.ndarray) -> bool:
    pts = {tuple(map(int, p)) for obj in existing for p in obj.tolist()}
    for r, c in candidate.tolist():
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (rr + dr, cc + dc) in pts:
                    return False
    return True


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([int(dr), int(dc)], dtype=int)


def _crop_objects(objects: list[np.ndarray]) -> tuple[list[np.ndarray], tuple[int, int]]:
    r0 = min(int(obj[:, 0].min()) for obj in objects)
    c0 = min(int(obj[:, 1].min()) for obj in objects)
    r1 = max(int(obj[:, 0].max()) for obj in objects)
    c1 = max(int(obj[:, 1].max()) for obj in objects)
    margin_top = random.randint(1, 3)
    margin_bottom = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    margin_right = random.randint(1, 3)
    rr0 = max(0, r0 - margin_top)
    cc0 = max(0, c0 - margin_left)
    rr1 = min(WORK_SHAPE[0] - 1, r1 + margin_bottom)
    cc1 = min(WORK_SHAPE[1] - 1, c1 + margin_right)
    cropped = [_translate(obj, -rr0, -cc0) for obj in objects]
    return cropped, (rr1 - rr0 + 1, cc1 - cc0 + 1)


def _uniform_shift_holes(objects: list[np.ndarray], shift: int, shape: tuple[int, int]) -> int:
    moved = []
    h, w = shape
    for coords in objects:
        dst = _shift_object_coords(coords, shift)
        if not bool((0 <= dst[:, 0]).all() and (dst[:, 0] < h).all() and (0 <= dst[:, 1]).all() and (dst[:, 1] < w).all()):
            return -1
        moved.append(dst)
    grid = _paint_objects(shape, moved, list(range(1, len(moved) + 1)))
    return len(_enclosed_regions(grid, bg=0))


def _hole_boundary_ids(objects: list[np.ndarray], shape: tuple[int, int]) -> tuple[list[np.ndarray], set[int]]:
    grid = np.zeros(shape, dtype=int)
    id_grid = np.full(shape, -1, dtype=int)
    for idx, coords in enumerate(objects):
        grid[coords[:, 0], coords[:, 1]] = idx + 1
        id_grid[coords[:, 0], coords[:, 1]] = idx
    holes = _enclosed_regions(grid, bg=0)
    boundary_ids: set[int] = set()
    for coords in holes:
        for r, c in coords.tolist():
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < shape[0] and 0 <= cc < shape[1] and id_grid[rr, cc] >= 0:
                        boundary_ids.add(int(id_grid[rr, cc]))
    return holes, boundary_ids


def _semantic_ok(input_grid: np.ndarray, objects_in: list[np.ndarray], participant_count: int) -> bool:
    shifted = _shifted_grid_only(input_grid)
    holes = _enclosed_regions(shifted, bg=0)
    if len(holes) != 1:
        return False
    if _enclosed_regions(input_grid, bg=0):
        return False
    moved = [_shift_object_coords(obj, _bbox(obj)[3] - _bbox(obj)[1] + 1) for obj in objects_in]
    post_holes, boundary_ids = _hole_boundary_ids(moved, input_grid.shape)
    if len(post_holes) != 1 or boundary_ids != set(range(participant_count)):
        return False
    widths = [_bbox(obj)[3] - _bbox(obj)[1] + 1 for obj in objects_in[:participant_count]]
    if len(set(widths)) < 2:
        return False
    for idx in range(participant_count):
        reduced = moved[:idx] + moved[idx + 1 :]
        reduced_grid = _paint_objects(input_grid.shape, reduced, list(range(1, len(reduced) + 1)))
        if _enclosed_regions(reduced_grid, bg=0):
            return False
    core_objects = objects_in[:participant_count]
    for shift in range(1, max(widths) + 1):
        holes_n = _uniform_shift_holes(core_objects, shift, input_grid.shape)
        if holes_n > 0:
            return False
    expected = _fill_enclosed_regions(shifted, bg=0)
    return bool(np.array_equal(main(input_grid), expected))


def _make_candidate() -> tuple[np.ndarray, list[np.ndarray], int] | None:
    template_fn = _template_c_bar
    final_participants, participant_count = template_fn()
    total_objects = 5 if participant_count == 4 else random.randint(3, 5)
    distractor_count = total_objects - participant_count

    union_r0 = min(int(obj[:, 0].min()) for obj in final_participants)
    union_c0 = min(int(obj[:, 1].min()) for obj in final_participants)
    union_r1 = max(int(obj[:, 0].max()) for obj in final_participants)
    union_c1 = max(int(obj[:, 1].max()) for obj in final_participants)
    union_h = union_r1 - union_r0 + 1
    union_w = union_c1 - union_c0 + 1

    max_shift = max(_bbox(obj)[3] - _bbox(obj)[1] + 1 for obj in final_participants)
    anchor_r = random.randint(3, WORK_SHAPE[0] - union_h - 4)
    anchor_c = random.randint(max_shift + 3, WORK_SHAPE[1] - union_w - 4)
    objects_final = [_translate(obj, anchor_r, anchor_c) for obj in final_participants]

    for _ in range(distractor_count):
        shape = _random_distractor()
        placed = None
        for _ in range(80):
            br0, bc0, br1, bc1 = _bbox(shape)
            dr = random.randint(1, WORK_SHAPE[0] - (br1 - br0 + 1) - 2)
            dc = random.randint(4, WORK_SHAPE[1] - (bc1 - bc0 + 1) - 2)
            candidate = _translate(shape, dr - br0, dc - bc0)
            if _placeable(objects_final, candidate):
                placed = candidate
                break
        if placed is None:
            return None
        objects_final.append(placed)

    objects_input = []
    for coords in objects_final:
        width = _bbox(coords)[3] - _bbox(coords)[1] + 1
        src = _shift_object_coords(coords, -width)
        if not bool((0 <= src[:, 0]).all() and (src[:, 0] < WORK_SHAPE[0]).all()):
            return None
        if not bool((0 <= src[:, 1]).all() and (src[:, 1] < WORK_SHAPE[1]).all()):
            return None
        objects_input.append(src)

    colors = random.sample(COLOR_POOL, k=total_objects)
    cropped_objects, shape = _crop_objects(objects_input + objects_final)
    input_objects = cropped_objects[:total_objects]
    final_objects = cropped_objects[total_objects:]
    inp = _paint_objects(shape, input_objects, colors, bg=0)
    expected_shifted = _paint_objects(shape, final_objects, colors, bg=0)

    if max(shape) > 30:
        return None
    if len(find_components(inp, bg=0, connectivity=8, monochromatic=False)) != total_objects:
        return None
    if not np.array_equal(_shifted_grid_only(inp), expected_shifted):
        return None
    if not _semantic_ok(inp, input_objects, participant_count):
        return None
    return inp, input_objects, participant_count


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _make_candidate()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError("failed to generate a valid emergent-hole input")
