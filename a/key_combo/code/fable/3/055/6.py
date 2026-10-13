from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
DIVIDER = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ROTATION_STEPS = (0, 1, 3)  # np.rot90: 0, ccw 90, cw 90

FRAME_H_RANGE = (10, 14)
FRAME_W_RANGE = (12, 17)
OBJECT_COUNT_OPTIONS = (3, 4)
SIDE_OPTIONS = (3, 4)
PLACEMENT_MARGIN = 1
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values = list(map(int, grid.ravel()))
    return int(Counter(values).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _extract_objects(frame: np.ndarray, bg: int) -> list[dict]:
    height, width = frame.shape
    seen = np.zeros((height, width), dtype=bool)
    objects: list[dict] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(frame[r, c]) == int(bg):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(frame[nr, nc]) != int(bg):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            bbox = _bbox_from_coords(arr)
            patch = _crop_bbox(frame, bbox)
            objects.append(
                {
                    "coords": arr,
                    "bbox": bbox,
                    "patch": patch,
                    "mask": patch != int(bg),
                }
            )

    return objects


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _canonical_mask_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _infer_rotation_steps(top_patch: np.ndarray, bottom_patch: np.ndarray, bg: int) -> int:
    exact = [
        k
        for k in ROTATION_STEPS
        if np.rot90(top_patch, k).shape == bottom_patch.shape and np.array_equal(np.rot90(top_patch, k), bottom_patch)
    ]
    if len(exact) == 1:
        return int(exact[0])

    top_mask = top_patch != int(bg)
    bottom_mask = bottom_patch != int(bg)
    mask_only = [
        k
        for k in ROTATION_STEPS
        if np.rot90(top_mask, k).shape == bottom_mask.shape and np.array_equal(np.rot90(top_mask, k), bottom_mask)
    ]
    assert len(mask_only) == 1, "each object must reveal a unique 0/90-degree step"
    return int(mask_only[0])


def _top_left_from_center2(center2: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    height, width = map(int, shape)
    r_num = int(center2[0]) - (height - 1)
    c_num = int(center2[1]) - (width - 1)
    assert r_num % 2 == 0 and c_num % 2 == 0
    return int(r_num // 2), int(c_num // 2)


def _paste_non_bg(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    height, width = patch.shape
    assert 0 <= top and 0 <= left and top + height <= out.shape[0] and left + width <= out.shape[1]
    dest = out[top : top + height, left : left + width]
    mask = patch != int(bg)
    assert not bool(((dest != int(bg)) & mask).any())
    dest[mask] = patch[mask]


def _find_divider_row(grid: np.ndarray) -> tuple[int, int]:
    height = int(grid.shape[0])
    candidates = [
        r
        for r in range(1, height - 1)
        if r == height - r - 1 and np.all(grid[r] == int(grid[r, 0]))
    ]
    assert len(candidates) == 1, "expected a single full-width divider row splitting equal-height frames"
    row = int(candidates[0])
    return row, int(grid[row, 0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each object's frame-to-frame rigid motion and extrapolate the next frame.

    Input structure:
    - A full-width monochrome divider row splits the grid into two equal-height canvases.
    - The upper and lower canvases contain the same set of disconnected foreground objects on a shared background.
    - Object colors may repeat, so cross-frame pairing must rely on each object's tight-bbox mask up to rotation.

    Rule:
    - Match every upper-frame object with the lower-frame object that has the same mask up to rotation.
    - For each pair, infer whether the object kept its orientation or rotated 90 degrees clockwise/counterclockwise, plus its translation vector between frames.
    - Apply that same per-object transform once more to the lower-frame object and draw all results on a blank canvas of single-frame size.

    Key insight:
    - The two halves are consecutive moments of the same scene; the answer is the unshown third frame obtained by letting every object continue its own motion rule one more step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider_row, divider_color = _find_divider_row(g)
    top = g[:divider_row]
    bottom = g[divider_row + 1 :]
    assert top.shape == bottom.shape and top.size > 0

    bg = _most_common_color(np.vstack([top, bottom]))
    assert int(bg) != int(divider_color)

    top_objects = _extract_objects(top, bg)
    bottom_objects = _extract_objects(bottom, bg)
    assert len(top_objects) == len(bottom_objects) >= 1

    top_by_key: dict[tuple[tuple[int, int], ...], dict] = {}
    for obj in top_objects:
        key = _canonical_mask_key(obj["mask"])
        assert key not in top_by_key, "upper-frame shapes must be unique up to rotation"
        top_by_key[key] = obj

    out = np.full(top.shape, int(bg), dtype=int)
    matched = 0
    seen_keys: set[tuple[tuple[int, int], ...]] = set()
    for obj in bottom_objects:
        key = _canonical_mask_key(obj["mask"])
        assert key in top_by_key, "every lower-frame object must match an upper-frame shape"
        assert key not in seen_keys, "lower-frame shapes must also be unique up to rotation"
        top_obj = top_by_key[key]

        steps = _infer_rotation_steps(top_obj["patch"], obj["patch"], bg)
        tr0, tc0, tr1, tc1 = map(int, top_obj["bbox"])
        br0, bc0, br1, bc1 = map(int, obj["bbox"])
        delta_center2 = (int(br0 + br1 - tr0 - tr1), int(bc0 + bc1 - tc0 - tc1))

        next_patch = np.rot90(obj["patch"], steps)
        next_center2 = (int(br0 + br1 + delta_center2[0]), int(bc0 + bc1 + delta_center2[1]))
        nr0, nc0 = _top_left_from_center2(next_center2, next_patch.shape)
        _paste_non_bg(out, next_patch, nr0, nc0, bg)
        seen_keys.add(key)
        matched += 1

    assert matched == len(top_objects)
    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _random_connected_mask(side: int, area: int) -> np.ndarray | None:
    for _ in range(120):
        cells = {(random.randrange(side), random.randrange(side))}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != area:
            continue

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1 - r0 + 1) != side or int(c1 - c0 + 1) != side:
            continue

        rotations = {_mask_key(np.rot90(mask, k)) for k in range(4)}
        if len(rotations) != 4:
            continue
        return mask
    return None


def _generate_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen_keys: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(count):
        placed = False
        for _ in range(200):
            side = int(random.choice(SIDE_OPTIONS))
            area_low = side + 1
            area_high = min(side * side - 2, side + 4)
            area = random.randint(area_low, area_high)
            mask = _random_connected_mask(side, area)
            if mask is None:
                continue
            key = _canonical_mask_key(mask)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            shapes.append(mask)
            placed = True
            break
        if not placed:
            return None
    return shapes


def _rotation_plan(count: int) -> list[int]:
    plan = [0, 1, 3]
    while len(plan) < count:
        plan.append(int(random.choice(ROTATION_STEPS)))
    random.shuffle(plan)
    return plan[:count]


def _vector_candidates() -> list[tuple[int, int]]:
    candidates = [
        (dr, dc)
        for dr in (-2, -1, 0, 1, 2)
        for dc in (-2, -1, 0, 1, 2)
        if (dr, dc) != (0, 0)
    ]
    random.shuffle(candidates)
    return candidates


def _valid_axis_starts(limit: int, side: int, delta: int) -> list[int]:
    starts = []
    low = PLACEMENT_MARGIN
    high = limit - side - PLACEMENT_MARGIN
    for start in range(low, high + 1):
        if all(low <= start + step * delta <= high for step in (0, 1, 2)):
            starts.append(start)
    return starts


def _render_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BACKGROUND, dtype=int)
    patch[mask] = int(color)
    return patch


def _gen_attempt() -> np.ndarray | None:
    frame_h = random.randint(*FRAME_H_RANGE)
    frame_w = random.randint(*FRAME_W_RANGE)
    count = int(random.choice(OBJECT_COUNT_OPTIONS))
    shapes = _generate_shapes(count)
    if shapes is None:
        return None

    colors_used = random.randint(2, min(4, count))
    palette = random.sample(list(OBJECT_COLORS), colors_used)
    colors = [int(random.choice(palette)) for _ in range(count)]
    rotations = _rotation_plan(count)

    order = sorted(range(count), key=lambda idx: shapes[idx].shape[0], reverse=True)
    placed_boxes: dict[int, list[tuple[int, int, int, int]]] = {0: [], 1: [], 2: []}
    placements: dict[int, tuple[int, int, int, int]] = {}

    for idx in order:
        side = int(shapes[idx].shape[0])
        found = False
        for dr, dc in _vector_candidates():
            rows = _valid_axis_starts(frame_h, side, dr)
            cols = _valid_axis_starts(frame_w, side, dc)
            if not rows or not cols:
                continue

            row_order = rows[:]
            col_order = cols[:]
            random.shuffle(row_order)
            random.shuffle(col_order)
            for r0 in row_order:
                for c0 in col_order:
                    boxes = [
                        (r0 + step * dr, c0 + step * dc, r0 + step * dr + side - 1, c0 + step * dc + side - 1)
                        for step in (0, 1, 2)
                    ]
                    if all(
                        all(_boxes_separated(boxes[step], other, PLACEMENT_GAP) for other in placed_boxes[step])
                        for step in (0, 1, 2)
                    ):
                        for step in (0, 1, 2):
                            placed_boxes[step].append(boxes[step])
                        placements[idx] = (r0, c0, dr, dc)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return None

    top = np.full((frame_h, frame_w), BACKGROUND, dtype=int)
    bottom = np.full((frame_h, frame_w), BACKGROUND, dtype=int)
    third = np.full((frame_h, frame_w), BACKGROUND, dtype=int)

    for idx, mask in enumerate(shapes):
        color = int(colors[idx])
        steps = int(rotations[idx])
        r0, c0, dr, dc = placements[idx]
        patch0 = _render_patch(mask, color)
        patch1 = np.rot90(patch0, steps)
        patch2 = np.rot90(patch1, steps)
        _paste_non_bg(top, patch0, r0, c0, BACKGROUND)
        _paste_non_bg(bottom, patch1, r0 + dr, c0 + dc, BACKGROUND)
        _paste_non_bg(third, patch2, r0 + 2 * dr, c0 + 2 * dc, BACKGROUND)

    inp = np.full((frame_h * 2 + 1, frame_w), BACKGROUND, dtype=int)
    inp[:frame_h] = top
    inp[frame_h] = DIVIDER
    inp[frame_h + 1 :] = bottom

    try:
        out = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(out, third):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid third-frame puzzle")
