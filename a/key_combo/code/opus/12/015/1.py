from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ANCHOR = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
DIR_BY_HOLES = {
    1: (-1, 0),  # up
    2: (0, 1),   # right
    3: (1, 0),   # down
    4: (0, -1),  # left
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _count_holes(mask: np.ndarray) -> int:
    bg = np.pad(~mask, 1, constant_values=True)
    h, w = bg.shape
    seen = np.zeros_like(bg, dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_border = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _find_anchor_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == ANCHOR)
    assert coords.shape[0] == 4
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == ANCHOR)
    return r0, c0, r1, c1


def _extract_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, ANCHOR) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            bbox = _bbox_from_coords(arr)
            local = np.zeros((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), dtype=bool)
            local[arr[:, 0] - bbox[0], arr[:, 1] - bbox[1]] = True
            holes = _count_holes(local)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox,
                    "holes": holes,
                }
            )
    return objects


def _center2(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + r1, c0 + c1


def _is_adjacent_to_mask(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return True
    return False


def _shift_coords(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    shifted = coords.copy()
    shifted[:, 0] += int(dr)
    shifted[:, 1] += int(dc)
    return shifted


def _can_place(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if occupied[rr, cc]:
            return False
    return True


def _slide_until_contact(coords: np.ndarray, occupied: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    dr, dc = direction
    current = coords.copy()
    if _is_adjacent_to_mask(current, occupied):
        return current
    while True:
        candidate = _shift_coords(current, dr, dc)
        if not _can_place(candidate, occupied):
            return current
        current = candidate
        if _is_adjacent_to_mask(current, occupied):
            return current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each object toward the center anchor in the direction encoded by its enclosed-hole count.

    Input structure:
    - Black background with one fixed teal 2x2 anchor near the center.
    - Several non-black, non-teal 4-connected monochrome objects sit around the anchor.
    - Each object contains one to four enclosed black holes and is positioned so exactly one cardinal slide can approach the anchor.

    Rule:
    - Count each object's enclosed 4-connected black regions inside its own bounding box.
    - Map hole count to slide direction: 1->up, 2->right, 3->down, 4->left.
    - Sort objects by Manhattan distance from object-bbox center to anchor-bbox center, nearest first.
    - Process in that order, sliding each object one cell at a time until it first becomes 4-neighbor adjacent to the anchor or to an already-settled moved object; if it is already adjacent, leave it in place.

    Key insight:
    - The objects' internal topology supplies movement instructions, while distance order turns earlier arrivals into collision blockers for later ones.
    """
    g = np.asarray(grid, dtype=int)
    anchor_bbox = _find_anchor_bbox(g)
    anchor_center2 = _center2(anchor_bbox)
    objects = _extract_objects(g)
    assert objects

    ordered = []
    for obj in objects:
        assert obj["holes"] in DIR_BY_HOLES
        obj_center2 = _center2(obj["bbox"])
        dist2 = abs(obj_center2[0] - anchor_center2[0]) + abs(obj_center2[1] - anchor_center2[1])
        ordered.append((dist2, obj["color"], obj))
    ordered.sort(key=lambda item: (item[0], item[1]))

    out = np.full_like(g, BG)
    occupied = np.zeros_like(g, dtype=bool)
    r0, c0, r1, c1 = anchor_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = ANCHOR
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    for _dist2, _color, obj in ordered:
        final_coords = _slide_until_contact(obj["coords"], occupied, DIR_BY_HOLES[obj["holes"]])
        out[final_coords[:, 0], final_coords[:, 1]] = obj["color"]
        occupied[final_coords[:, 0], final_coords[:, 1]] = True

    return out


def _base_mask(holes: int) -> np.ndarray:
    if holes == 1:
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        return mask
    if holes == 2:
        mask = np.ones((3, 5), dtype=bool)
        mask[1, 1] = False
        mask[1, 3] = False
        return mask
    if holes == 3:
        mask = np.ones((4, 5), dtype=bool)
        for cell in [(1, 1), (1, 3), (2, 2)]:
            mask[cell] = False
        return mask
    mask = np.ones((5, 5), dtype=bool)
    for cell in [(1, 1), (1, 3), (3, 1), (3, 3)]:
        mask[cell] = False
    return mask


def _make_object_mask(holes: int) -> np.ndarray:
    mask = _base_mask(holes)
    canvas = np.zeros((mask.shape[0] + 4, mask.shape[1] + 4), dtype=bool)
    canvas[2 : 2 + mask.shape[0], 2 : 2 + mask.shape[1]] = mask

    for _ in range(random.randint(0, 3)):
        coords = np.argwhere(canvas)
        bbox = _bbox_from_coords(coords)
        candidates = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1] and not canvas[nr, nc]:
                    if bbox[0] <= nr <= bbox[2] and bbox[1] <= nc <= bbox[3]:
                        continue
                    candidates.add((nr, nc))
        candidate_list = list(candidates)
        random.shuffle(candidate_list)
        for nr, nc in candidate_list:
            trial = canvas.copy()
            trial[nr, nc] = True
            cropped = _crop_mask(trial)
            if _count_holes(cropped) == holes and _component_count(cropped) == 1:
                canvas = trial
                break

    cropped = _crop_mask(canvas)
    if random.random() < 0.5:
        cropped = np.fliplr(cropped)
    if random.random() < 0.5:
        cropped = np.flipud(cropped)
    cropped = np.rot90(cropped, random.randrange(4))
    cropped = _crop_mask(cropped)
    assert _component_count(cropped) == 1
    assert _count_holes(cropped) == holes
    return cropped


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _choose_common_lane(anchor_pos: int, sizes: list[int], limit: int) -> int | None:
    lo = max(0, max(anchor_pos - size + 1 for size in sizes))
    hi = min(min(anchor_pos + 1, limit - size) for size in sizes)
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _choose_increasing_steps(capacities: list[int]) -> list[int] | None:
    steps = []
    prev = 0
    for cap in capacities:
        lo = prev + 1
        hi = min(cap, prev + 4)
        if lo > hi:
            return None
        step = random.randint(lo, hi)
        steps.append(step)
        prev = step
    return steps


def _plan_direction(
    direction: int,
    masks: list[np.ndarray],
    anchor_bbox: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
) -> list[tuple[np.ndarray, int, int]] | None:
    h, w = grid_shape
    ar0, ac0, ar1, ac1 = anchor_bbox
    ordered_masks = masks[:]
    random.shuffle(ordered_masks)

    if direction in (2, 4):
        row0 = _choose_common_lane(ar0, [mask.shape[0] for mask in ordered_masks], h)
        if row0 is None:
            return None
        widths = [mask.shape[1] for mask in ordered_masks]
        placed = []
        if direction == 2:
            final_lefts = []
            cursor = ac0
            for ww in widths:
                left = cursor - ww
                if left < 0:
                    return None
                final_lefts.append(left)
                cursor = left
            steps = _choose_increasing_steps(final_lefts)
            if steps is None:
                return None
            for mask, final_left, step in zip(ordered_masks, final_lefts, steps):
                placed.append((mask, row0, final_left - step))
        else:
            final_lefts = []
            cursor = ac1 + 1
            for ww in widths:
                left = cursor
                if left + ww > w:
                    return None
                final_lefts.append(left)
                cursor = left + ww
            capacities = [w - mask.shape[1] - final_left for mask, final_left in zip(ordered_masks, final_lefts)]
            steps = _choose_increasing_steps(capacities)
            if steps is None:
                return None
            for mask, final_left, step in zip(ordered_masks, final_lefts, steps):
                placed.append((mask, row0, final_left + step))
        return placed

    col0 = _choose_common_lane(ac0, [mask.shape[1] for mask in ordered_masks], w)
    if col0 is None:
        return None
    heights = [mask.shape[0] for mask in ordered_masks]
    placed = []
    if direction == 1:
        final_tops = []
        cursor = ar1 + 1
        for hh in heights:
            top = cursor
            if top + hh > h:
                return None
            final_tops.append(top)
            cursor = top + hh
        capacities = [h - mask.shape[0] - final_top for mask, final_top in zip(ordered_masks, final_tops)]
        steps = _choose_increasing_steps(capacities)
        if steps is None:
            return None
        for mask, final_top, step in zip(ordered_masks, final_tops, steps):
            placed.append((mask, final_top + step, col0))
    else:
        final_tops = []
        cursor = ar0
        for hh in heights:
            top = cursor - hh
            if top < 0:
                return None
            final_tops.append(top)
            cursor = top
        steps = _choose_increasing_steps(final_tops)
        if steps is None:
            return None
        for mask, final_top, step in zip(ordered_masks, final_tops, steps):
            placed.append((mask, final_top - step, col0))
    return placed


def _touches_existing(occupied: np.ndarray, top: int, left: int, mask: np.ndarray, radius: int) -> bool:
    h, w = occupied.shape
    hh, ww = mask.shape
    for r in range(hh):
        for c in range(ww):
            if not mask[r, c]:
                continue
            rr = top + r
            cc = left + c
            if occupied[rr, cc]:
                return True
            for nr in range(max(0, rr - radius), min(h, rr + radius + 1)):
                for nc in range(max(0, cc - radius), min(w, cc + radius + 1)):
                    if occupied[nr, nc]:
                        return True
    return False


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    hh, ww = mask.shape
    for r in range(hh):
        for c in range(ww):
            if mask[r, c]:
                grid[top + r, left + c] = int(color)


def _interval_relation(
    bbox: tuple[int, int, int, int], anchor_bbox: tuple[int, int, int, int]
) -> tuple[bool, bool]:
    row_overlap = _overlap_1d(bbox[0], bbox[2], anchor_bbox[0], anchor_bbox[2])
    col_overlap = _overlap_1d(bbox[1], bbox[3], anchor_bbox[1], anchor_bbox[3])
    return row_overlap, col_overlap


def _adjacent_between(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> bool:
    occ = np.zeros(shape, dtype=bool)
    occ[b[:, 0], b[:, 1]] = True
    return _is_adjacent_to_mask(a, occ)


def _validate_scene(grid: np.ndarray, counts_by_direction: dict[int, int]) -> bool:
    anchor_bbox = _find_anchor_bbox(grid)
    anchor_center2 = _center2(anchor_bbox)
    objects = _extract_objects(grid)
    if len(objects) != sum(counts_by_direction.values()):
        return False

    seen_dists = set()
    by_color = {}
    for obj in objects:
        holes = obj["holes"]
        if holes not in DIR_BY_HOLES:
            return False
        bbox = obj["bbox"]
        row_overlap, col_overlap = _interval_relation(bbox, anchor_bbox)
        if row_overlap == col_overlap:
            return False
        if holes == 1 and not (bbox[0] > anchor_bbox[2] and col_overlap):
            return False
        if holes == 2 and not (bbox[3] < anchor_bbox[1] and row_overlap):
            return False
        if holes == 3 and not (bbox[2] < anchor_bbox[0] and col_overlap):
            return False
        if holes == 4 and not (bbox[1] > anchor_bbox[3] and row_overlap):
            return False
        center2 = _center2(bbox)
        dist2 = abs(center2[0] - anchor_center2[0]) + abs(center2[1] - anchor_center2[1])
        if dist2 in seen_dists:
            return False
        seen_dists.add(dist2)
        by_color[obj["color"]] = obj

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    out_objects = {obj["color"]: obj for obj in _extract_objects(out)}
    if set(out_objects) != set(by_color):
        return False

    anchor_occ = np.zeros(out.shape, dtype=bool)
    r0, c0, r1, c1 = anchor_bbox
    anchor_occ[r0 : r1 + 1, c0 : c1 + 1] = True

    collision_blocked = 0
    for color, obj in out_objects.items():
        coords = obj["coords"]
        anchor_adj = _is_adjacent_to_mask(coords, anchor_occ)
        if not anchor_adj:
            for other_color, other in out_objects.items():
                if other_color == color:
                    continue
                if _adjacent_between(coords, other["coords"], out.shape):
                    collision_blocked += 1
                    break

    if any(count == 2 for count in counts_by_direction.values()) and collision_blocked == 0:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(18, 20)
    w = random.randint(18, 20)
    grid = np.full((h, w), BG, dtype=int)
    ar = h // 2 - 1
    ac = w // 2 - 1
    anchor_bbox = (ar, ac, ar + 1, ac + 1)
    grid[ar : ar + 2, ac : ac + 2] = ANCHOR

    counts_by_direction = {1: 1, 2: 1, 3: 1, 4: 1}
    counts_by_direction[random.choice([1, 2, 3])] += 1

    total_objects = sum(counts_by_direction.values())
    colors = random.sample(PALETTE, total_objects)

    placed_specs = []
    for direction in [1, 2, 3, 4]:
        plan = None
        for _ in range(20):
            masks = [_make_object_mask(direction) for _ in range(counts_by_direction[direction])]
            plan = _plan_direction(direction, masks, anchor_bbox, grid.shape)
            if plan is not None:
                break
        if plan is None:
            return None
        placed_specs.extend((direction, mask, top, left) for mask, top, left in plan)

    random.shuffle(placed_specs)
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[ar : ar + 2, ac : ac + 2] = True

    for color, (_, mask, top, left) in zip(colors, placed_specs):
        hh, ww = mask.shape
        if top < 0 or left < 0 or top + hh > h or left + ww > w:
            return None
        if _touches_existing(occupied, top, left, mask, radius=1):
            return None
        _paint_mask(grid, top, left, mask, color)
        for r in range(hh):
            for c in range(ww):
                if mask[r, c]:
                    occupied[top + r, left + c] = True

    if not _validate_scene(grid, counts_by_direction):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
