from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


DIR_LEFT = "left"
DIR_RIGHT = "right"
DIR_UP = "up"
DIR_DOWN = "down"

MATCHED_HORIZONTAL = "matched_horizontal"
MATCHED_VERTICAL = "matched_vertical"
UNMATCHED_HORIZONTAL = "unmatched_horizontal"
UNMATCHED_VERTICAL = "unmatched_vertical"
PAIR_MODES = (
    MATCHED_HORIZONTAL,
    MATCHED_VERTICAL,
    UNMATCHED_HORIZONTAL,
    UNMATCHED_VERTICAL,
)

GRID_MIN = 22
GRID_MAX = 28
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
BOX_MIN = 5
BOX_MAX = 7
GAP_MIN = 3
GAP_MAX = 6
BOX_PAD = 1
PAIR_PLACE_TRIES = 120
GEN_ATTEMPTS = 500


@dataclass
class ObjectInfo:
    index: int
    cells: np.ndarray
    base_color: int
    pointer_color: int
    pointer: tuple[int, int]
    bbox: tuple[int, int, int, int]
    direction: str


@dataclass
class PlacedObject:
    cells: np.ndarray
    pointer: tuple[int, int]
    base_color: int
    pointer_color: int
    direction: str
    matched: bool
    irregular: bool


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _direction_from_pointer(pointer: tuple[int, int], bbox_rc: tuple[int, int, int, int]) -> str:
    pr, pc = pointer
    r0, c0, r1, c1 = bbox_rc
    left = pc - c0
    right = c1 - pc
    up = pr - r0
    down = r1 - pr
    horizontal = min(left, right)
    vertical = min(up, down)
    if horizontal <= vertical:
        return DIR_LEFT if left <= right else DIR_RIGHT
    return DIR_UP if up <= down else DIR_DOWN


def _extract_objects(grid: np.ndarray) -> tuple[int, list[ObjectInfo], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    components = _components_mask(g != bg)
    comp_ids = np.full(g.shape, -1, dtype=int)
    objects: list[ObjectInfo] = []
    for idx, coords in enumerate(components):
        comp_ids[coords[:, 0], coords[:, 1]] = idx
        values = g[coords[:, 0], coords[:, 1]]
        counts = Counter(map(int, values.tolist()))
        assert len(counts) == 2, "each object must contain one body color and one pointer color"
        pointer_candidates = [color for color, count in counts.items() if count == 1]
        assert len(pointer_candidates) == 1, "each object must have exactly one single-pixel pointer"
        pointer_color = int(pointer_candidates[0])
        base_candidates = [color for color in counts if color != pointer_color]
        assert len(base_candidates) == 1
        base_color = int(base_candidates[0])
        pointer_index = np.flatnonzero(values == pointer_color)
        assert len(pointer_index) == 1
        pointer = tuple(int(x) for x in coords[int(pointer_index[0])])
        bbox_rc = _bbox(coords)
        direction = _direction_from_pointer(pointer, bbox_rc)
        objects.append(
            ObjectInfo(
                index=idx,
                cells=coords,
                base_color=base_color,
                pointer_color=pointer_color,
                pointer=pointer,
                bbox=bbox_rc,
                direction=direction,
            )
        )
    return bg, objects, comp_ids


def _ray_hit_object(obj: ObjectInfo, grid: np.ndarray, bg: int, comp_ids: np.ndarray) -> int | None:
    pr, pc = obj.pointer
    H, W = grid.shape
    if obj.direction == DIR_LEFT:
        edge = int(obj.cells[obj.cells[:, 0] == pr, 1].min())
        r, c, dr, dc = pr, edge - 1, 0, -1
    elif obj.direction == DIR_RIGHT:
        edge = int(obj.cells[obj.cells[:, 0] == pr, 1].max())
        r, c, dr, dc = pr, edge + 1, 0, 1
    elif obj.direction == DIR_UP:
        edge = int(obj.cells[obj.cells[:, 1] == pc, 0].min())
        r, c, dr, dc = edge - 1, pc, -1, 0
    else:
        edge = int(obj.cells[obj.cells[:, 1] == pc, 0].max())
        r, c, dr, dc = edge + 1, pc, 1, 0

    while 0 <= r < H and 0 <= c < W:
        if int(grid[r, c]) != bg:
            hit = int(comp_ids[r, c])
            return hit if hit >= 0 else None
        r += dr
        c += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each direction-marked color pair when one object's pointer ray reaches its same-body partner.

    Input structure:
    - The background is the most common color.
    - Each foreground object is a 4-connected two-color component: a large body color
      plus one single-pixel interior pointer of another color.
    - Every body color appears in exactly two separate objects.

    Rule:
    - For each object, infer the pointer direction from which side of the object's
      bounding box is closest to the pointer pixel; if a horizontal and vertical side
      tie, prefer the horizontal side.
    - Extend a ray outward from the object's boundary along the pointer pixel's row or
      column and find the first non-background object on that ray.
    - If that first hit has the same body color and the same pointer color, recolor
      both objects entirely to the pointer color; otherwise leave them unchanged.

    Key insight:
    - The body color identifies the only possible partner, but the pointer position
      decides whether the partner is actually activated by encoding a search direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, objects, comp_ids = _extract_objects(g)

    out = g.copy()
    recolor: dict[int, int] = {}
    for obj in objects:
        hit = _ray_hit_object(obj, g, bg, comp_ids)
        if hit is None or hit == obj.index:
            continue
        other = objects[hit]
        if obj.base_color != other.base_color:
            continue
        if obj.pointer_color != other.pointer_color:
            continue
        recolor[obj.index] = obj.pointer_color
        recolor[other.index] = obj.pointer_color

    for idx, color in recolor.items():
        coords = objects[idx].cells
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    H, W = mask.shape
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _protected_cells(h: int, w: int, pointer_rc: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    pr, pc = pointer_rc
    protected = {(pr, pc)}
    if direction == DIR_LEFT:
        for cc in range(0, pc + 1):
            protected.add((pr, cc))
    elif direction == DIR_RIGHT:
        for cc in range(pc, w):
            protected.add((pr, cc))
    elif direction == DIR_UP:
        for rr in range(0, pr + 1):
            protected.add((rr, pc))
    else:
        for rr in range(pr, h):
            protected.add((rr, pc))
    return protected


def _make_shape(h: int, w: int, pointer_rc: tuple[int, int], direction: str) -> tuple[np.ndarray, bool]:
    mask = np.ones((h, w), dtype=bool)
    protected = _protected_cells(h, w, pointer_rc, direction)
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in protected
        and (r, c) != pointer_rc
        and (r in (0, h - 1) or c in (0, w - 1))
    ]
    random.shuffle(candidates)
    target_removals = random.choice([0, 1, 1, 2, 2, 3])
    removed = 0
    irregular = False
    for r, c in candidates:
        if removed >= target_removals:
            break
        mask[r, c] = False
        base_mask = mask.copy()
        base_mask[pointer_rc] = False
        if not _mask_connected(mask) or not _mask_connected(base_mask):
            mask[r, c] = True
            continue
        removed += 1
        irregular = True
    return mask, irregular


def _choose_horizontal_pointer_row(h: int) -> int:
    if random.random() < 0.5:
        return random.choice([1, h - 2])
    inner = [r for r in range(2, h - 2)]
    return random.choice(inner) if inner else random.choice([1, h - 2])


def _choose_vertical_pointer_col(w: int) -> int:
    inner = [c for c in range(2, w - 2)]
    return random.choice(inner) if inner else w // 2


def _cells_from_mask(r0: int, c0: int, mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords.astype(int)


def _box_free(occupied: np.ndarray, reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    r1 = max(0, r0 - BOX_PAD)
    c1 = max(0, c0 - BOX_PAD)
    r2 = min(H, r0 + h + BOX_PAD)
    c2 = min(W, c0 + w + BOX_PAD)
    return not np.any(occupied[r1:r2, c1:c2]) and not np.any(reserved[r0 : r0 + h, c0 : c0 + w])


def _mark_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    H, W = occupied.shape
    r1 = max(0, r0 - BOX_PAD)
    c1 = max(0, c0 - BOX_PAD)
    r2 = min(H, r0 + h + BOX_PAD)
    c2 = min(W, c0 + w + BOX_PAD)
    occupied[r1:r2, c1:c2] = True


def _row_clear(mask_a: np.ndarray, mask_b: np.ndarray, row: int, c0: int, c1: int) -> bool:
    if c0 >= c1:
        return True
    return not np.any(mask_a[row, c0:c1]) and not np.any(mask_b[row, c0:c1])


def _col_clear(mask_a: np.ndarray, mask_b: np.ndarray, col: int, r0: int, r1: int) -> bool:
    if r0 >= r1:
        return True
    return not np.any(mask_a[r0:r1, col]) and not np.any(mask_b[r0:r1, col])


def _place_pair(
    mode: str,
    base_color: int,
    pointer_color: int,
    occupied: np.ndarray,
    reserved: np.ndarray,
    specs: list[PlacedObject],
) -> bool:
    H, W = occupied.shape
    matched = mode in (MATCHED_HORIZONTAL, MATCHED_VERTICAL)

    for _ in range(PAIR_PLACE_TRIES):
        h1 = random.randint(BOX_MIN, BOX_MAX)
        w1 = random.randint(BOX_MIN, BOX_MAX)
        h2 = random.randint(BOX_MIN, BOX_MAX)
        w2 = random.randint(BOX_MIN, BOX_MAX)
        gap = random.randint(GAP_MIN, GAP_MAX)

        if mode in (MATCHED_HORIZONTAL, UNMATCHED_HORIZONTAL):
            pr1 = _choose_horizontal_pointer_row(h1)
            pr2 = _choose_horizontal_pointer_row(h2)
            total_w = w1 + gap + w2
            if total_w > W - 2:
                continue
            ray_low = max(1 + pr1, 1 + pr2)
            ray_high = min(H - h1 - 1 + pr1, H - h2 - 1 + pr2)
            if ray_low > ray_high:
                continue
            ray_row = random.randint(ray_low, ray_high)
            r01, r02 = ray_row - pr1, ray_row - pr2
            c01 = random.randint(1, W - total_w - 1)
            c02 = c01 + w1 + gap
            if not _box_free(occupied, reserved, r01, c01, h1, w1):
                continue
            if not _box_free(occupied, reserved, r02, c02, h2, w2):
                continue
            if mode == MATCHED_HORIZONTAL:
                if not _row_clear(occupied, reserved, ray_row, c01 + w1, c02):
                    continue
                dir1, dir2 = DIR_RIGHT, DIR_LEFT
            else:
                if not _row_clear(occupied, reserved, ray_row, 0, c01):
                    continue
                if not _row_clear(occupied, reserved, ray_row, c02 + w2, W):
                    continue
                dir1, dir2 = DIR_LEFT, DIR_RIGHT
            p1_local = (pr1, w1 - 2 if dir1 == DIR_RIGHT else 1)
            p2_local = (pr2, w2 - 2 if dir2 == DIR_RIGHT else 1)
            shape1, irregular1 = _make_shape(h1, w1, p1_local, dir1)
            shape2, irregular2 = _make_shape(h2, w2, p2_local, dir2)
            cells1 = _cells_from_mask(r01, c01, shape1)
            cells2 = _cells_from_mask(r02, c02, shape2)
            pointer1 = (r01 + p1_local[0], c01 + p1_local[1])
            pointer2 = (r02 + p2_local[0], c02 + p2_local[1])
            specs.append(
                PlacedObject(cells1, pointer1, base_color, pointer_color, dir1, matched, irregular1)
            )
            specs.append(
                PlacedObject(cells2, pointer2, base_color, pointer_color, dir2, matched, irregular2)
            )
            _mark_box(occupied, r01, c01, h1, w1)
            _mark_box(occupied, r02, c02, h2, w2)
            if mode == MATCHED_HORIZONTAL:
                reserved[ray_row, c01 + w1 : c02] = True
            else:
                reserved[ray_row, :c01] = True
                reserved[ray_row, c02 + w2 :] = True
            return True

        pc1 = _choose_vertical_pointer_col(w1)
        pc2 = _choose_vertical_pointer_col(w2)
        total_h = h1 + gap + h2
        if total_h > H - 2:
            continue
        ray_low = max(1 + pc1, 1 + pc2)
        ray_high = min(W - w1 - 1 + pc1, W - w2 - 1 + pc2)
        if ray_low > ray_high:
            continue
        ray_col = random.randint(ray_low, ray_high)
        c01, c02 = ray_col - pc1, ray_col - pc2
        r01 = random.randint(1, H - total_h - 1)
        r02 = r01 + h1 + gap
        if not _box_free(occupied, reserved, r01, c01, h1, w1):
            continue
        if not _box_free(occupied, reserved, r02, c02, h2, w2):
            continue
        if mode == MATCHED_VERTICAL:
            if not _col_clear(occupied, reserved, ray_col, r01 + h1, r02):
                continue
            dir1, dir2 = DIR_DOWN, DIR_UP
        else:
            if not _col_clear(occupied, reserved, ray_col, 0, r01):
                continue
            if not _col_clear(occupied, reserved, ray_col, r02 + h2, H):
                continue
            dir1, dir2 = DIR_UP, DIR_DOWN
        p1_local = (h1 - 2 if dir1 == DIR_DOWN else 1, pc1)
        p2_local = (h2 - 2 if dir2 == DIR_DOWN else 1, pc2)
        shape1, irregular1 = _make_shape(h1, w1, p1_local, dir1)
        shape2, irregular2 = _make_shape(h2, w2, p2_local, dir2)
        cells1 = _cells_from_mask(r01, c01, shape1)
        cells2 = _cells_from_mask(r02, c02, shape2)
        pointer1 = (r01 + p1_local[0], c01 + p1_local[1])
        pointer2 = (r02 + p2_local[0], c02 + p2_local[1])
        specs.append(
            PlacedObject(cells1, pointer1, base_color, pointer_color, dir1, matched, irregular1)
        )
        specs.append(
            PlacedObject(cells2, pointer2, base_color, pointer_color, dir2, matched, irregular2)
        )
        _mark_box(occupied, r01, c01, h1, w1)
        _mark_box(occupied, r02, c02, h2, w2)
        if mode == MATCHED_VERTICAL:
            reserved[r01 + h1 : r02, ray_col] = True
        else:
            reserved[:r01, ray_col] = True
            reserved[r02 + h2 :, ray_col] = True
        return True

    return False


def _sample_modes(pair_count: int) -> list[str]:
    while True:
        modes = [random.choice(PAIR_MODES) for _ in range(pair_count)]
        matched = [mode.startswith("matched") for mode in modes]
        if not any(matched):
            continue
        if pair_count > 1 and all(matched):
            continue
        return modes


def _render_specs(shape: tuple[int, int], bg: int, specs: list[PlacedObject]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full(shape, bg, dtype=int)
    expected = np.full(shape, bg, dtype=int)
    for spec in specs:
        grid[spec.cells[:, 0], spec.cells[:, 1]] = spec.base_color
        expected[spec.cells[:, 0], spec.cells[:, 1]] = spec.base_color
    for spec in specs:
        pr, pc = spec.pointer
        grid[pr, pc] = spec.pointer_color
        if spec.matched:
            expected[spec.cells[:, 0], spec.cells[:, 1]] = spec.pointer_color
        else:
            expected[pr, pc] = spec.pointer_color
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    pair_count = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    modes = _sample_modes(pair_count)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    base_colors = palette[1 : 1 + pair_count]
    pointer_colors = palette[1 + pair_count : 1 + 2 * pair_count]

    occupied = np.zeros((H, W), dtype=bool)
    reserved = np.zeros((H, W), dtype=bool)
    specs: list[PlacedObject] = []

    for pair_idx, mode in enumerate(modes):
        ok = _place_pair(mode, base_colors[pair_idx], pointer_colors[pair_idx], occupied, reserved, specs)
        if not ok:
            return None

    if not any(spec.irregular for spec in specs):
        return None

    inp, expected = _render_specs((H, W), bg, specs)
    if _most_common_color(inp) != bg:
        return None

    try:
        parsed_bg, objects, _ = _extract_objects(inp)
        result = main(inp)
    except AssertionError:
        return None

    if parsed_bg != bg or len(objects) != 2 * pair_count:
        return None
    if not np.array_equal(result, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
