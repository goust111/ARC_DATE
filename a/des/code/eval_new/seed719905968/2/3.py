from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
SIDES = ("top", "bottom", "left", "right")
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GENERATION_RETRIES = 400
SHAPE_RETRIES = 200
PLACEMENT_PADDING = 1
OUTER_MARGIN = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0)


def _flip_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side in ("left", "right"):
        return np.fliplr(mask)
    return np.flipud(mask)


def _lock_offsets(side: str, inner_shape: tuple[int, int], mask_shape: tuple[int, int]) -> tuple[int, int]:
    inner_h, inner_w = inner_shape
    mask_h, mask_w = mask_shape
    assert mask_h <= inner_h and mask_w <= inner_w
    if side == "top":
        return 0, (inner_w - mask_w) // 2
    if side == "bottom":
        return inner_h - mask_h, (inner_w - mask_w) // 2
    if side == "left":
        return (inner_h - mask_h) // 2, 0
    return (inner_h - mask_h) // 2, inner_w - mask_w


def _paint_lock(
    out: np.ndarray,
    color: int,
    mask: np.ndarray,
    side: str,
    inner_origin: tuple[int, int],
    inner_shape: tuple[int, int],
) -> None:
    flipped = _flip_mask(mask, side)
    dr, dc = _lock_offsets(side, inner_shape, flipped.shape)
    r0 = inner_origin[0] + dr
    c0 = inner_origin[1] + dc
    region = out[r0 : r0 + flipped.shape[0], c0 : c0 + flipped.shape[1]]
    region[flipped] = color


def _find_separator(g: np.ndarray) -> int:
    h, w = g.shape
    candidates = [
        c
        for c in range(w)
        if np.all(g[:, c] == GREY) and np.any(g[:, :c] != GREY) and np.any(g[:, c + 1 :] != GREY)
    ]
    assert len(candidates) == 1, f"expected one grey separator column, got {candidates}"
    return candidates[0]


def _extract_keys(left: np.ndarray) -> dict[int, np.ndarray]:
    keys: dict[int, np.ndarray] = {}
    for comp in _components((left != BG) & (left != GREY)):
        colors = {int(left[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = colors.pop()
        assert color not in keys
        mask, _ = _tight_mask(comp)
        keys[color] = mask
    assert keys
    return keys


def _find_lock_marker(
    right: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    if r0 > 0:
        for c in range(c0 + 1, c1):
            value = int(right[r0 - 1, c])
            if value not in (BG, GREY):
                candidates.append(("top", value))
    if r1 + 1 < right.shape[0]:
        for c in range(c0 + 1, c1):
            value = int(right[r1 + 1, c])
            if value not in (BG, GREY):
                candidates.append(("bottom", value))
    if c0 > 0:
        for r in range(r0 + 1, r1):
            value = int(right[r, c0 - 1])
            if value not in (BG, GREY):
                candidates.append(("left", value))
    if c1 + 1 < right.shape[1]:
        for r in range(r0 + 1, r1):
            value = int(right[r, c1 + 1])
            if value not in (BG, GREY):
                candidates.append(("right", value))
    assert len(candidates) == 1, f"expected one colored lock marker, got {candidates}"
    return candidates[0]


def _extract_locks(right: np.ndarray) -> dict[int, tuple[str, tuple[int, int], tuple[int, int]]]:
    locks: dict[int, tuple[str, tuple[int, int], tuple[int, int]]] = {}
    for comp in _components(right == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        local = right[r0 : r1 + 1, c0 : c1 + 1]
        h, w = local.shape
        assert h >= 3 and w >= 3

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.array_equal(local == GREY, border)
        assert np.all(local[1:-1, 1:-1] == BG)

        side, color = _find_lock_marker(right, r0, c0, r1, c1)
        assert color not in locks
        locks[color] = (side, (r0 + 1, c0 + 1), (h - 2, w - 2))
    assert locks
    return locks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored key from the left side into the matching right-side lock after mirroring it by the marker-defined insertion direction.

    Input structure:
    - The background is black, and one full grey column separates a left key area from a right lock area.
    - The left area contains several connected monochrome key sprites, one per color.
    - The right area contains several hollow grey rectangular locks with black interiors, and each lock has one colored single-pixel marker just outside one side.
    - Each marker color matches exactly one key color.

    Rule:
    - Split the grid at the grey separator column.
    - Extract every colored key on the left, and extract every grey lock plus its marker side on the right.
    - Pair keys to locks by color, then mirror the key across the axis perpendicular to the marked insertion side.
    - Place the mirrored key inside the lock, aligned flush with the marked side, erase the original left-side key, and keep the lock frame, separator, and marker unchanged.

    Key insight:
    - The marker color identifies which key belongs in which lock, while the marker side simultaneously determines the mirror axis and the side of the lock that the transformed key must touch.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    keys = _extract_keys(left)
    locks = _extract_locks(right)
    assert set(keys) == set(locks)

    out = g.copy()
    left_region = out[:, :sep]
    left_region[(left != BG) & (left != GREY)] = BG

    for color, key_mask in keys.items():
        side, inner_origin_rel, inner_shape = locks[color]
        inner_origin = (int(inner_origin_rel[0]), int(inner_origin_rel[1] + sep + 1))
        _paint_lock(out, color, key_mask, side, inner_origin, inner_shape)

    return out


def _random_key_mask(rng: random.Random) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        canvas_h = rng.randint(2, 5)
        canvas_w = rng.randint(2, 5)
        max_area = canvas_h * canvas_w - 1
        min_area = min(max_area, max(3, (canvas_h * canvas_w) // 2))
        target_area = rng.randint(min_area, max_area)

        cells = {(rng.randrange(canvas_h), rng.randrange(canvas_w))}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        tight = mask[r0 : r1 + 1, c0 : c1 + 1]

        if tight.shape[0] < 2 or tight.shape[1] < 2:
            continue
        if int(tight.sum()) == int(tight.size):
            continue
        if np.array_equal(tight, np.fliplr(tight)) or np.array_equal(tight, np.flipud(tight)):
            continue
        if not tight[0, :].any() or not tight[-1, :].any() or not tight[:, 0].any() or not tight[:, -1].any():
            continue
        return tight
    raise RuntimeError("failed to generate an asymmetric connected key")


def _key_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return patch


def _lock_patch(
    inner_shape: tuple[int, int], side: str, color: int, rng: random.Random
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    inner_h, inner_w = inner_shape
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    if side == "top":
        patch = np.zeros((frame_h + 1, frame_w), dtype=int)
        frame_r0 = 1
        marker_r = 0
        marker_c = rng.randint(1, frame_w - 2)
        inner_origin = (2, 1)
    elif side == "bottom":
        patch = np.zeros((frame_h + 1, frame_w), dtype=int)
        frame_r0 = 0
        marker_r = frame_h
        marker_c = rng.randint(1, frame_w - 2)
        inner_origin = (1, 1)
    elif side == "left":
        patch = np.zeros((frame_h, frame_w + 1), dtype=int)
        frame_r0 = 0
        marker_r = rng.randint(1, frame_h - 2)
        marker_c = 0
        inner_origin = (1, 2)
    else:
        patch = np.zeros((frame_h, frame_w + 1), dtype=int)
        frame_r0 = 0
        marker_r = rng.randint(1, frame_h - 2)
        marker_c = frame_w
        inner_origin = (1, 1)

    if side in ("top", "bottom"):
        patch[frame_r0 : frame_r0 + frame_h, 0] = GREY
        patch[frame_r0 : frame_r0 + frame_h, frame_w - 1] = GREY
        patch[frame_r0, :] = GREY
        patch[frame_r0 + frame_h - 1, :] = GREY
    else:
        frame_c0 = 1 if side == "left" else 0
        patch[:, frame_c0] = np.where(
            patch[:, frame_c0] == BG,
            GREY,
            patch[:, frame_c0],
        )
        patch[:, frame_c0 + frame_w - 1] = np.where(
            patch[:, frame_c0 + frame_w - 1] == BG,
            GREY,
            patch[:, frame_c0 + frame_w - 1],
        )
        patch[0, frame_c0 : frame_c0 + frame_w] = GREY
        patch[frame_h - 1, frame_c0 : frame_c0 + frame_w] = GREY

    patch[marker_r, marker_c] = color
    return patch, inner_origin, (inner_h, inner_w)


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = patch.shape
    if row_hi < row_lo or col_hi < col_lo:
        return None

    positions = [
        (r, c)
        for r in range(row_lo, row_hi + 1)
        for c in range(col_lo, col_hi + 1)
        if not reserved[
            max(0, r - PLACEMENT_PADDING) : min(grid.shape[0], r + h + PLACEMENT_PADDING),
            max(0, c - PLACEMENT_PADDING) : min(grid.shape[1], c + w + PLACEMENT_PADDING),
        ].any()
    ]
    if not positions:
        return None

    r0, c0 = rng.choice(positions)
    mask = patch != BG
    region = grid[r0 : r0 + h, c0 : c0 + w]
    region[mask] = patch[mask]
    reserved[
        max(0, r0 - PLACEMENT_PADDING) : min(grid.shape[0], r0 + h + PLACEMENT_PADDING),
        max(0, c0 - PLACEMENT_PADDING) : min(grid.shape[1], c0 + w + PLACEMENT_PADDING),
    ] = True
    return int(r0), int(c0)


def _build_expected(
    inp: np.ndarray,
    placements: list[dict[str, object]],
    sep: int,
) -> np.ndarray:
    out = inp.copy()
    left_region = out[:, :sep]
    left_region[(left_region != BG) & (left_region != GREY)] = BG
    for item in placements:
        color = int(item["color"])
        key_mask = np.asarray(item["key_mask"], dtype=bool)
        lock_top, lock_left = item["lock_pos"]
        inner_r, inner_c = item["inner_origin"]
        side = str(item["side"])
        inner_shape = tuple(map(int, item["inner_shape"]))
        inner_origin = (int(lock_top + inner_r), int(lock_left + inner_c))
        _paint_lock(out, color, key_mask, side, inner_origin, inner_shape)
    return out


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(2, 4)
    colors = rng.sample(COLOR_POOL, pair_count)
    sides = rng.sample(SIDES, pair_count)

    items: list[dict[str, object]] = []
    for color, side in zip(colors, sides):
        key_mask = _random_key_mask(rng)
        key_patch = _key_patch(key_mask, color)
        lock_patch, inner_origin, inner_shape = _lock_patch(key_mask.shape, side, color, rng)
        items.append(
            {
                "color": color,
                "side": side,
                "key_mask": key_mask,
                "key_patch": key_patch,
                "lock_patch": lock_patch,
                "inner_origin": inner_origin,
                "inner_shape": inner_shape,
            }
        )

    max_key_h = max(int(np.asarray(item["key_patch"]).shape[0]) for item in items)
    max_key_w = max(int(np.asarray(item["key_patch"]).shape[1]) for item in items)
    max_lock_h = max(int(np.asarray(item["lock_patch"]).shape[0]) for item in items)
    max_lock_w = max(int(np.asarray(item["lock_patch"]).shape[1]) for item in items)

    left_min = max(max_key_w + 4, 9)
    right_min = max(max_lock_w + 4, 11)
    total_min_w = left_min + 1 + right_min
    if total_min_w > 30:
        return None

    min_h = max(max(max_key_h, max_lock_h) + 4, 10 + 2 * pair_count)
    if min_h > 30:
        return None

    height = rng.randint(min_h, 30)
    width = rng.randint(total_min_w, 30)
    sep = rng.randint(left_min, width - 1 - right_min)

    grid = np.zeros((height, width), dtype=int)
    grid[:, sep] = GREY
    reserved = np.zeros((height, width), dtype=bool)
    reserved[:, sep] = True

    for item in sorted(items, key=lambda x: np.asarray(x["key_patch"]).size, reverse=True):
        patch = np.asarray(item["key_patch"], dtype=int)
        pos = _place_patch(
            grid,
            reserved,
            patch,
            OUTER_MARGIN,
            height - patch.shape[0] - OUTER_MARGIN,
            OUTER_MARGIN,
            sep - patch.shape[1] - OUTER_MARGIN,
            rng,
        )
        if pos is None:
            return None
        item["key_pos"] = pos

    for item in sorted(items, key=lambda x: np.asarray(x["lock_patch"]).size, reverse=True):
        patch = np.asarray(item["lock_patch"], dtype=int)
        pos = _place_patch(
            grid,
            reserved,
            patch,
            OUTER_MARGIN,
            height - patch.shape[0] - OUTER_MARGIN,
            sep + 1 + OUTER_MARGIN,
            width - patch.shape[1] - OUTER_MARGIN,
            rng,
        )
        if pos is None:
            return None
        item["lock_pos"] = pos

    expected = _build_expected(grid, items, sep)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock puzzle")
