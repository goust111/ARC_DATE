from __future__ import annotations

import itertools
import random

import numpy as np


BG_COLOR = 0
GRAY_NOISE = 5
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components_for_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return components


def _bbox_center2(comp: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    return int(r0 + r1), int(c0 + c1)


def _center_key(comp: dict) -> tuple[int, int]:
    return _bbox_center2(comp)


def _rounded_midpoint(center2_a: tuple[int, int], center2_b: tuple[int, int]) -> tuple[int, int]:
    return (
        (int(center2_a[0]) + int(center2_b[0]) + 2) // 4,
        (int(center2_a[1]) + int(center2_b[1]) + 2) // 4,
    )


def _rotate_coords_180(coords: np.ndarray, midpoint: tuple[int, int]) -> np.ndarray:
    out = np.asarray(coords, dtype=int).copy()
    out[:, 0] = 2 * int(midpoint[0]) - out[:, 0]
    out[:, 1] = 2 * int(midpoint[1]) - out[:, 1]
    return out


def _draw_coords(canvas: np.ndarray, coords: np.ndarray, color: int, bg: int) -> None:
    h, w = canvas.shape
    for r, c in np.asarray(coords, dtype=int):
        rr = int(r)
        cc = int(c)
        assert 0 <= rr < h and 0 <= cc < w
        assert canvas[rr, cc] in (int(bg), int(color))
        canvas[rr, cc] = int(color)


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(np.asarray(mask, dtype=bool))
    if len(pts) == 0:
        return False

    coords = {tuple(map(int, pt)) for pt in pts}
    stack = [next(iter(coords))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS8:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _touches_full_bbox(mask: np.ndarray) -> bool:
    m = np.asarray(mask, dtype=bool)
    return bool(m[0, :].any() and m[-1, :].any() and m[:, 0].any() and m[:, -1].any())


def _mask_asymmetric(mask: np.ndarray) -> bool:
    m = np.asarray(mask, dtype=bool)
    return not np.array_equal(m, np.rot90(m, 2))


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(np.asarray(mask, dtype=bool)) + np.array([int(r0), int(c0)], dtype=int)


def _box_from_top_left(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return (
        int(a[2]) + int(pad) < int(b[0])
        or int(b[2]) + int(pad) < int(a[0])
        or int(a[3]) + int(pad) < int(b[1])
        or int(b[3]) + int(pad) < int(a[1])
    )


def _point_in_box(point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    return int(box[0]) <= int(point[0]) <= int(box[2]) and int(box[1]) <= int(point[1]) <= int(box[3])


def _box_is_free(reserve: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = reserve.shape
    r0 = max(0, int(box[0]) - int(pad))
    c0 = max(0, int(box[1]) - int(pad))
    r1 = min(h - 1, int(box[2]) + int(pad))
    c1 = min(w - 1, int(box[3]) + int(pad))
    return not bool(reserve[r0 : r1 + 1, c0 : c1 + 1].any())


def _reserve_box(reserve: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = reserve.shape
    r0 = max(0, int(box[0]) - int(pad))
    c0 = max(0, int(box[1]) - int(pad))
    r1 = min(h - 1, int(box[2]) + int(pad))
    c1 = min(w - 1, int(box[3]) + int(pad))
    reserve[r0 : r1 + 1, c0 : c1 + 1] = True


def main(grid: np.ndarray) -> np.ndarray:
    """Pair same-colored fragments, rotate the earlier one 180 degrees about the pair midpoint, and keep only the merged result.

    Input structure:
    - The background is black.
    - Non-gray foreground objects are monochrome 8-connected fragments of size 3-6 pixels.
    - Each non-gray color appears in exactly two separate fragments, forming one pair.
    - Extra gray pixels, when present, are isolated one-cell noise points on the background.

    Rule:
    - Ignore all gray single-pixel noise and clear it in the output.
    - For each non-gray color pair, order the two fragments by bbox-center position, top-to-bottom then left-to-right; call the earlier one A and the later one B.
    - Compute the midpoint of A and B's bbox centers, rounding to the nearest cell, then rotate every pixel of A by 180 degrees about that midpoint.
    - Draw the rotated A together with B in the pair color on a clean black canvas; erase both original fragments.

    Key insight:
    - Color identifies the pairing, but the actual move is a point reflection of one fragment around the midpoint between the two bbox centers, not a direct translation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _dominant_color(g)

    grouped: dict[int, list[dict]] = {}
    for color in map(int, np.unique(g)):
        if color == int(bg):
            continue
        comps = _components_for_color(g, color)
        if color == GRAY_NOISE:
            assert all(int(comp["area"]) == 1 for comp in comps)
            continue
        assert len(comps) == 2
        assert all(3 <= int(comp["area"]) <= 6 for comp in comps)
        grouped[int(color)] = comps

    assert grouped
    out = np.full((h, w), int(bg), dtype=int)

    for color, comps in grouped.items():
        a_comp, b_comp = sorted(comps, key=_center_key)
        midpoint = _rounded_midpoint(_bbox_center2(a_comp), _bbox_center2(b_comp))
        rotated = _rotate_coords_180(a_comp["coords"], midpoint)
        _draw_coords(out, rotated, int(color), int(bg))
        _draw_coords(out, b_comp["coords"], int(color), int(bg))

    return out


# --------------- generator ---------------

PAIR_COUNT_CHOICES = (2, 2, 3, 3, 4)
GRID_LIMIT = 30
BBOX_OPTIONS = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3))
NOISE_COUNT_CHOICES = (0, 0, 1, 2, 3, 4)
GENERATION_ATTEMPTS = 400
MASK_PAIR_ATTEMPTS = 200
PAIR_PLACEMENT_ATTEMPTS = 80


def _enumerate_masks(h: int, w: int) -> tuple[np.ndarray, ...]:
    masks = []
    cell_count = int(h * w)
    for area in range(3, min(6, cell_count) + 1):
        for combo in itertools.combinations(range(cell_count), area):
            mask = np.zeros((h, w), dtype=bool)
            for idx in combo:
                mask[idx // w, idx % w] = True
            if not _touches_full_bbox(mask):
                continue
            if not _mask_connected(mask):
                continue
            if not _mask_asymmetric(mask):
                continue
            masks.append(mask)
    assert masks
    return tuple(masks)


MASK_LIBRARY = {dims: _enumerate_masks(*dims) for dims in BBOX_OPTIONS}


def _sample_mask_pair(dims: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    masks = MASK_LIBRARY[dims]
    for _ in range(MASK_PAIR_ATTEMPTS):
        a_out = random.choice(masks)
        b_mask = random.choice(masks)
        if np.array_equal(a_out, b_mask):
            continue
        union = np.asarray(a_out | b_mask, dtype=bool)
        if not _mask_connected(union):
            continue
        if int(union.sum()) <= max(int(a_out.sum()), int(b_mask.sum())):
            continue
        if not bool((a_out & ~b_mask).any()):
            continue
        if not bool((b_mask & ~a_out).any()):
            continue
        return a_out.copy(), b_mask.copy()
    return None


def _sample_offset(h: int, w: int) -> tuple[int, int]:
    row_values = list(range(h + 2, h + 8))
    col_values = list(range(w + 2, w + 8))
    branch = random.choice(("horizontal", "vertical", "diag_right", "diag_left"))
    if branch == "horizontal":
        return 0, random.choice(col_values)
    if branch == "vertical":
        return random.choice(row_values), 0
    if branch == "diag_right":
        return random.choice(row_values), random.choice(col_values)
    return random.choice(row_values), -random.choice(col_values)


def _candidate_b_positions(
    grid_shape: tuple[int, int], h: int, w: int, dr: int, dc: int
) -> list[tuple[int, int]]:
    gh, gw = grid_shape
    r_lo = max(0, int(dr))
    r_hi = min(int(gh - h), int(gh - h + dr))
    c_lo = max(0, int(dc))
    c_hi = min(int(gw - w), int(gw - w + dc))
    if r_lo > r_hi or c_lo > c_hi:
        return []
    positions = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    random.shuffle(positions)
    return positions


def _draw_mask(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    _draw_coords(canvas, _coords_from_mask(mask, r0, c0), int(color), BG_COLOR)


def _place_pair(
    inp: np.ndarray,
    expected: np.ndarray,
    reserve: np.ndarray,
    color: int,
) -> bool:
    gh, gw = inp.shape
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        h, w = random.choice(BBOX_OPTIONS)
        pair = _sample_mask_pair((h, w))
        if pair is None:
            continue
        a_out, b_mask = pair
        a_in = np.rot90(a_out, 2)
        dr, dc = _sample_offset(h, w)
        positions = _candidate_b_positions((gh, gw), h, w, dr, dc)

        for b_r0, b_c0 in positions:
            a_r0 = int(b_r0 - dr)
            a_c0 = int(b_c0 - dc)
            a_box = _box_from_top_left(a_r0, a_c0, h, w)
            b_box = _box_from_top_left(b_r0, b_c0, h, w)
            if not _boxes_separated(a_box, b_box, pad=1):
                continue
            if not _box_is_free(reserve, a_box, pad=1):
                continue
            if not _box_is_free(reserve, b_box, pad=1):
                continue

            a_center2 = (2 * a_r0 + h - 1, 2 * a_c0 + w - 1)
            b_center2 = (2 * b_r0 + h - 1, 2 * b_c0 + w - 1)
            midpoint = _rounded_midpoint(a_center2, b_center2)
            if _point_in_box(midpoint, a_box) or _point_in_box(midpoint, b_box):
                continue
            if reserve[midpoint]:
                continue

            _draw_mask(inp, a_in, a_r0, a_c0, color)
            _draw_mask(inp, b_mask, b_r0, b_c0, color)
            _draw_mask(expected, a_out, b_r0, b_c0, color)
            _draw_mask(expected, b_mask, b_r0, b_c0, color)
            _reserve_box(reserve, a_box, pad=1)
            _reserve_box(reserve, b_box, pad=1)
            reserve[midpoint] = True
            return True

    return False


def _place_noise(inp: np.ndarray, reserve: np.ndarray) -> None:
    gh, gw = inp.shape
    target = random.choice(NOISE_COUNT_CHOICES)
    placed = 0
    cells = [(r, c) for r in range(gh) for c in range(gw)]
    random.shuffle(cells)

    for r, c in cells:
        if placed >= target:
            break
        if reserve[r, c] or inp[r, c] != BG_COLOR:
            continue
        inp[r, c] = GRAY_NOISE
        reserve[max(0, r - 1) : min(gh, r + 2), max(0, c - 1) : min(gw, c + 2)] = True
        placed += 1


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    min_side = 10 + 4 * int(pair_count)
    height = random.randint(min_side, GRID_LIMIT)
    width = random.randint(min_side, GRID_LIMIT)

    inp = np.full((height, width), BG_COLOR, dtype=int)
    expected = np.full_like(inp, BG_COLOR)
    reserve = np.zeros((height, width), dtype=bool)
    colors = random.sample(PAIR_COLORS, int(pair_count))

    for color in colors:
        if not _place_pair(inp, expected, reserve, int(color)):
            return None

    _place_noise(inp, reserve)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid midpoint-rotation pairing puzzle")
