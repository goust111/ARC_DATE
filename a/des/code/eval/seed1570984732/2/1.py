from __future__ import annotations

import random
from collections import Counter

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
FRAME_SIZE_CHOICES = [8, 9, 10, 11, 12, 13]
KEYHOLE_COUNT_CHOICES = [1, 2, 3]
DISTRACTOR_COUNT_CHOICES = [0, 1, 2]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _rect_border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if key not in seen:
            seen.add(key)
            variants.append(rot)
    return variants


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = []
    for rot in _rotation_variants(mask):
        keys.append((tuple(map(int, rot.shape)), rot.tobytes()))
    return min(keys)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int]:
    candidates: list[tuple[int, tuple[int, int, int, int], int]] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != int(bg)):
        for coords in _components(grid == int(color)):
            if len(coords) < 8:
                continue
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            expected = _rect_border_mask(h, w)
            actual = np.zeros((h, w), dtype=bool)
            rel = coords - np.array([r0, c0], dtype=int)
            actual[rel[:, 0], rel[:, 1]] = True
            if np.array_equal(actual, expected):
                candidates.append((h * w, (r0, c0, r1, c1), color))
    assert candidates, "main: no rectangular frame found"
    _, bbox, color = max(candidates, key=lambda item: item[0])
    return bbox, int(color)


def _orbit_positions(n: int, r: int, c: int) -> list[tuple[int, int]]:
    pts = [
        (r, c),
        (c, n - 1 - r),
        (n - 1 - r, n - 1 - c),
        (n - 1 - c, r),
    ]
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pt in pts:
        if pt not in seen:
            seen.add(pt)
            out.append(pt)
    return out


def _find_keyhole_mask(inner: np.ndarray, bg: int) -> np.ndarray:
    n, m = map(int, inner.shape)
    assert n == m, "main: inner pattern must be square for 90-degree symmetry"
    keyhole = np.zeros((n, n), dtype=bool)
    seen_orbits: set[tuple[tuple[int, int], ...]] = set()
    for r in range(n):
        for c in range(n):
            orbit = tuple(sorted(_orbit_positions(n, r, c)))
            if orbit in seen_orbits:
                continue
            seen_orbits.add(orbit)
            values = [int(inner[rr, cc]) for rr, cc in orbit]
            non_bg = {v for v in values if v != int(bg)}
            if int(bg) not in values or not non_bg:
                continue
            if len(non_bg) == 1:
                for rr, cc in orbit:
                    if int(inner[rr, cc]) == int(bg):
                        keyhole[rr, cc] = True
    return keyhole


def _extract_component_masks(mask: np.ndarray) -> list[np.ndarray]:
    return [_coords_to_mask(coords) for coords in _components(mask)]


def _piece_shape_counts(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int) -> Counter:
    r0, c0, r1, c1 = frame_bbox
    outside = grid != int(bg)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    counts: Counter = Counter()
    for coords in _components(outside):
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        counts[_canonical_rotation_key(_coords_to_mask(coords))] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Locate bg symmetry defects inside a square frame, verify matching outside pieces,
    and fill the defects with the frame color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_bbox, frame_color = _find_frame(g, bg)
    r0, c0, r1, c1 = frame_bbox
    inner = g[r0 + 1 : r1, c0 + 1 : c1].copy()

    keyhole_mask = _find_keyhole_mask(inner, bg)
    keyhole_masks = _extract_component_masks(keyhole_mask)
    assert keyhole_masks, "main: no keyhole detected"

    piece_counts = _piece_shape_counts(g, frame_bbox, bg)
    needed = Counter(_canonical_rotation_key(mask) for mask in keyhole_masks)
    for key, count in needed.items():
        assert piece_counts[key] >= count, "main: missing matching outside piece"

    out = inner.copy()
    out[keyhole_mask] = int(frame_color)
    return out


def _random_connected_shape(
    rng: random.Random,
    min_area: int,
    max_area: int,
    max_h: int,
    max_w: int,
) -> np.ndarray:
    target = rng.randint(min_area, max_area)
    box_h = rng.randint(1, max_h)
    box_w = rng.randint(1, max_w)
    cells = {(rng.randrange(box_h), rng.randrange(box_w))}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, box_h, box_w):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(tuple(frontier)))
    coords = np.asarray(sorted(cells), dtype=int)
    return _coords_to_mask(coords)


def _place_shape_in_quadrant(
    occ: np.ndarray, shape: np.ndarray, rng: random.Random
) -> tuple[int, int] | None:
    qh, qw = map(int, occ.shape)
    sh, sw = map(int, shape.shape)
    candidates = []
    for r0 in range(qh - sh + 1):
        for c0 in range(qw - sw + 1):
            ok = True
            for rr, cc in np.argwhere(shape):
                r = int(r0 + rr)
                c = int(c0 + cc)
                if occ[r, c]:
                    ok = False
                    break
                for nr, nc in _neighbors4(r, c, qh, qw):
                    if occ[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r0, c0))
    if not candidates:
        return None
    return rng.choice(candidates)


def _generate_keyhole_mask(
    n: int, count: int, rng: random.Random
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    q = n // 2
    occ = np.zeros((q, q), dtype=bool)
    placed: list[tuple[np.ndarray, int, int]] = []
    keys: set[tuple[tuple[int, int], bytes]] = set()
    max_side = min(4, q)
    max_area = min(5, q * q - 1)
    if max_area < 1:
        return None

    for _ in range(count):
        shape = None
        loc = None
        for _attempt in range(80):
            cand = _random_connected_shape(rng, 1, max_area, max_side, max_side)
            key = _canonical_rotation_key(cand)
            if key in keys:
                continue
            loc = _place_shape_in_quadrant(occ, cand, rng)
            if loc is None:
                continue
            shape = cand
            keys.add(key)
            break
        if shape is None or loc is None:
            return None
        r0, c0 = loc
        sh, sw = map(int, shape.shape)
        occ[r0 : r0 + sh, c0 : c0 + sw] |= shape
        placed.append((shape, r0, c0))

    mask = np.zeros((n, n), dtype=bool)
    for shape, r0, c0 in placed:
        sh, sw = map(int, shape.shape)
        mask[r0 : r0 + sh, c0 : c0 + sw] |= shape
    mask = np.rot90(mask, rng.randrange(4))
    return mask, [shape for shape, _r0, _c0 in placed]


def _build_symmetric_pattern(n: int, colors: list[int], rng: random.Random) -> np.ndarray:
    inner = np.zeros((n, n), dtype=int)
    seen_orbits: set[tuple[tuple[int, int], ...]] = set()
    used_colors: set[int] = set()
    for r in range(n):
        for c in range(n):
            orbit = tuple(sorted(_orbit_positions(n, r, c)))
            if orbit in seen_orbits:
                continue
            seen_orbits.add(orbit)
            color = int(rng.choice(colors))
            used_colors.add(color)
            for rr, cc in orbit:
                inner[rr, cc] = color
    if len(used_colors) == 1 and len(colors) > 1:
        orbit = next(iter(seen_orbits))
        alt = next(color for color in colors if color != inner[orbit[0]])
        for rr, cc in orbit:
            inner[rr, cc] = int(alt)
    return inner


def _shape_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top + r), int(left + c)) for r, c in np.argwhere(mask)]


def _can_place_piece(mask: np.ndarray, top: int, left: int, blocked: np.ndarray) -> bool:
    h, w = map(int, blocked.shape)
    cells = _shape_cells(mask, top, left)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = map(int, blocked.shape)
    for r, c in cells:
        blocked[r, c] = True
        for nr, nc in _neighbors8(r, c, h, w):
            blocked[nr, nc] = True


def _place_piece_anywhere(
    grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, color: int, rng: random.Random
) -> bool:
    gh, gw = map(int, grid.shape)
    ph, pw = map(int, mask.shape)
    candidates = []
    for r0 in range(gh - ph + 1):
        for c0 in range(gw - pw + 1):
            if _can_place_piece(mask, r0, c0, blocked):
                candidates.append((r0, c0))
    if not candidates:
        return False
    top, left = rng.choice(candidates)
    cells = _shape_cells(mask, top, left)
    for r, c in cells:
        grid[r, c] = int(color)
    _mark_blocked(blocked, cells)
    return True


def _generate_distractor_shape(
    hole_keys: set[tuple[tuple[int, int], bytes]],
    used_keys: set[tuple[tuple[int, int], bytes]],
    max_side: int,
    rng: random.Random,
) -> np.ndarray | None:
    for _ in range(80):
        shape = _random_connected_shape(rng, 1, 5, max_side, max_side)
        key = _canonical_rotation_key(shape)
        if key in hole_keys or key in used_keys:
            continue
        return shape
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inner_size = int(rng.choice(FRAME_SIZE_CHOICES))
    keyhole_count = int(rng.choice(KEYHOLE_COUNT_CHOICES))
    if inner_size // 2 <= 4:
        keyhole_count = min(keyhole_count, 2)

    colors = list(range(1, 10))
    rng.shuffle(colors)
    frame_color = colors.pop()
    pattern_color_count = rng.randint(2, 4)
    pattern_colors = [colors.pop() for _ in range(pattern_color_count)]

    keyhole_result = _generate_keyhole_mask(inner_size, keyhole_count, rng)
    if keyhole_result is None:
        return None
    keyhole_mask, _ = keyhole_result
    keyhole_components = _extract_component_masks(keyhole_mask)
    if len(keyhole_components) != keyhole_count:
        return None

    inner = _build_symmetric_pattern(inner_size, pattern_colors, rng)
    inner_with_holes = inner.copy()
    inner_with_holes[keyhole_mask] = 0

    expected = inner_with_holes.copy()
    expected[keyhole_mask] = int(frame_color)

    frame_size = inner_size + 2
    matched_piece_masks = [rng.choice(_rotation_variants(mask)) for mask in keyhole_components]
    hole_keys = {_canonical_rotation_key(mask) for mask in keyhole_components}

    distractor_count = int(rng.choice(DISTRACTOR_COUNT_CHOICES))
    max_extra = min(len(colors), len(matched_piece_masks) + distractor_count)
    if max_extra < len(matched_piece_masks):
        return None

    min_side = max(frame_size + 8, 18)
    max_side = max(min_side, 28)
    grid_h = rng.randint(min_side, max_side)
    grid_w = rng.randint(min_side, max_side)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    top_limit = grid_h - frame_size - 3
    left_limit = grid_w - frame_size - 3
    if top_limit < 2 or left_limit < 2:
        return None
    frame_top = rng.randint(2, top_limit)
    frame_left = rng.randint(2, left_limit)
    frame_bottom = frame_top + frame_size - 1
    frame_right = frame_left + frame_size - 1

    grid[frame_top, frame_left : frame_right + 1] = int(frame_color)
    grid[frame_bottom, frame_left : frame_right + 1] = int(frame_color)
    grid[frame_top : frame_bottom + 1, frame_left] = int(frame_color)
    grid[frame_top : frame_bottom + 1, frame_right] = int(frame_color)
    grid[frame_top + 1 : frame_bottom, frame_left + 1 : frame_right] = inner_with_holes

    blocked = np.zeros_like(grid, dtype=bool)
    r0 = max(0, frame_top - 1)
    c0 = max(0, frame_left - 1)
    r1 = min(grid_h, frame_bottom + 2)
    c1 = min(grid_w, frame_right + 2)
    blocked[r0:r1, c0:c1] = True

    piece_colors: list[int] = []
    while len(piece_colors) < len(matched_piece_masks) + distractor_count:
        if colors:
            piece_colors.append(colors.pop())
        else:
            piece_colors.append(rng.choice(pattern_colors))

    for piece_mask, piece_color in zip(matched_piece_masks, piece_colors[: len(matched_piece_masks)]):
        if not _place_piece_anywhere(grid, blocked, piece_mask, piece_color, rng):
            return None

    used_distractor_keys: set[tuple[tuple[int, int], bytes]] = set()
    for piece_color in piece_colors[len(matched_piece_masks) :]:
        shape = _generate_distractor_shape(hole_keys, used_distractor_keys, 4, rng)
        if shape is None:
            return None
        used_distractor_keys.add(_canonical_rotation_key(shape))
        piece_mask = rng.choice(_rotation_variants(shape))
        if not _place_piece_anywhere(grid, blocked, piece_mask, piece_color, rng):
            return None

    if _most_common_color(grid) != 0:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid instance")
