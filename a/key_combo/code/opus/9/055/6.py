from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
AXIS_COLOR = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GENERATION_ATTEMPTS = 400

SHAPE_LIBRARY = [
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[0, 1], [1, 1], [0, 1]], dtype=int),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neigh = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, ignored: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in ignored or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": _bbox(coords_arr),
                    "area": int(coords_arr.shape[0]),
                }
            )
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_patch(grid: np.ndarray, coords: np.ndarray, fill: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(fill), dtype=int)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    patch[rr, cc] = grid[coords[:, 0], coords[:, 1]]
    return patch, (r0, c0, r1, c1)


def _paint_patch(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = out.shape
    ph, pw = patch.shape
    for r in range(ph):
        for c in range(pw):
            value = int(patch[r, c])
            if value == bg:
                continue
            rr = top + r
            cc = left + c
            assert 0 <= rr < h and 0 <= cc < w, "patch paint out of bounds"
            existing = int(out[rr, cc])
            assert existing in {bg, AXIS_COLOR, value}, "unexpected sprite collision"
            out[rr, cc] = value


def _stamp_scaled_pair(
    out: np.ndarray,
    patch: np.ndarray,
    bbox: tuple[int, int, int, int],
    axis_col: int,
    side: str,
    bg: int,
) -> None:
    r0, c0, r1, c1 = bbox
    del r1
    distance = axis_col - c1 if side == "left" else c0 - axis_col
    assert distance >= 1, "sprite must be strictly on one side of the axis"
    factor = 1 << (distance - 1)
    scaled = np.kron(patch, np.ones((factor, factor), dtype=int))
    top = r0
    if side == "left":
        left = c1 - scaled.shape[1] + 1
    else:
        left = c0
    _paint_patch(out, scaled, top, left, bg)

    for lr, lc in np.argwhere(scaled != bg):
        rr = top + int(lr)
        cc = left + int(lc)
        mc = 2 * axis_col - cc
        assert 0 <= rr < out.shape[0] and 0 <= mc < out.shape[1], "mirror paint out of bounds"
        existing = int(out[rr, mc])
        value = int(scaled[lr, lc])
        assert existing in {bg, AXIS_COLOR, value}, "unexpected mirror collision"
        out[rr, mc] = value


def _find_axis_body(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for coords in _components_of_mask(g == AXIS_COLOR, connectivity=8):
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local = coords - np.array([r0, c0], dtype=int)
        patch[local[:, 0], local[:, 1]] = True
        if patch.shape[1] % 2 == 1 and np.array_equal(patch, np.fliplr(patch)):
            candidates.append((coords, (r0, c0, r1, c1), (c0 + c1) // 2))
    assert len(candidates) == 1, "expected exactly one grey vertically symmetric body"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each side sprite by its distance to the grey body's borrowed vertical axis, then mirror it across that axis.

    Input structure:
    - The background is black.
    - A single grey connected object near the middle is mirror-symmetric left-to-right and therefore defines a vertical symmetry axis.
    - Two or three other connected monochrome sprites of distinct colors lie strictly on the same side of that axis.

    Rule:
    - Detect the grey symmetric object and read its vertical axis from the center column of its symmetric bbox.
    - For each colored sprite, measure the horizontal distance D from the axis to the sprite's nearest occupied column.
    - Replace that sprite by a pixel-scaled copy with factor 2^(D-1), keeping its top row and its axis-nearest edge fixed so it grows away from the axis.
    - Reflect the scaled sprite across the same axis onto the opposite side; grey pixels remain unless covered by a colored scaled pixel.

    Key insight:
    - The same borrowed axis controls both operations: it is the mirror line and also the hidden lookup that converts position into scaling strength.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    body_coords, body_bbox, axis_col = _find_axis_body(g)
    del body_bbox

    sprites = _monochrome_components(g, ignored={bg, AXIS_COLOR})
    assert 2 <= len(sprites) <= 3, "expected two or three colored sprites"

    side = None
    for comp in sprites:
        _patch, bbox = _extract_patch(g, comp["coords"], bg)
        r0, c0, r1, c1 = bbox
        del r0, r1
        if c1 < axis_col:
            current = "left"
        elif c0 > axis_col:
            current = "right"
        else:
            raise AssertionError("sprite must not cross the axis")
        if side is None:
            side = current
        else:
            assert side == current, "all sprites must be on the same side"
    assert side is not None

    out = np.full_like(g, bg)
    out[body_coords[:, 0], body_coords[:, 1]] = AXIS_COLOR
    for comp in sorted(sprites, key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"])):
        patch, bbox = _extract_patch(g, comp["coords"], bg)
        _stamp_scaled_pair(out, patch, bbox, axis_col, side, bg)
    return out


def _walk_toward(mask: np.ndarray, start: tuple[int, int], target: tuple[int, int]) -> None:
    r, c = start
    tr, tc = target
    mask[r, c] = True
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((r + 1, c))
        elif r > tr:
            moves.append((r - 1, c))
        if c < tc:
            moves.append((r, c + 1))
        elif c > tc:
            moves.append((r, c - 1))
        r, c = random.choice(moves)
        mask[r, c] = True


def _grow_connected(mask: np.ndarray, target_cells: int) -> None:
    h, w = mask.shape
    while int(mask.sum()) < target_cells:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in DIR4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True


def _make_body_mask(height: int, width: int) -> np.ndarray:
    half_width = (width + 1) // 2
    axis_local = half_width - 1
    for _ in range(120):
        half = np.zeros((height, half_width), dtype=bool)
        left_anchor = (random.randrange(height), 0)
        axis_anchor = (random.randrange(height), axis_local)
        _walk_toward(half, left_anchor, axis_anchor)
        second_anchor = (random.randrange(height), random.randrange(half_width))
        _walk_toward(half, second_anchor, axis_anchor)
        target = random.randint(max(half.sum() + 1, height + 1), height * half_width - 1)
        _grow_connected(half, target)
        if not half[:, 0].any() or not half[:, axis_local].any():
            continue
        full = np.zeros((height, width), dtype=bool)
        full[:, :half_width] |= half
        full[:, width - half_width :] |= np.fliplr(half)
        if len(_components_of_mask(full, connectivity=4)) != 1:
            continue
        if full.all():
            continue
        return full
    raise RuntimeError("failed to generate a symmetric body")


def _choose_shapes(distances: list[int]) -> list[np.ndarray] | None:
    candidates_per_distance: list[list[np.ndarray]] = []
    for distance in distances:
        factor = 1 << (distance - 1)
        options = [shape.copy() for shape in SHAPE_LIBRARY if shape.shape[0] * factor <= 12 and shape.shape[1] * factor <= 12]
        if not options:
            return None
        candidates_per_distance.append(options)

    if len(distances) >= 2 and random.random() < 0.55:
        i, j = random.sample(range(len(distances)), 2)
        shared = [
            shape.copy()
            for shape in SHAPE_LIBRARY
            if shape.shape[0] * (1 << (distances[i] - 1)) <= 12
            and shape.shape[1] * (1 << (distances[i] - 1)) <= 12
            and shape.shape[0] * (1 << (distances[j] - 1)) <= 12
            and shape.shape[1] * (1 << (distances[j] - 1)) <= 12
        ]
        if shared:
            common = random.choice(shared)
            shapes = [random.choice(options).copy() for options in candidates_per_distance]
            shapes[i] = common.copy()
            shapes[j] = common.copy()
            return shapes

    return [random.choice(options).copy() for options in candidates_per_distance]


def _areas_not_monotone(distances: list[int], shapes: list[np.ndarray]) -> bool:
    pairs = sorted((dist, int(shape.sum())) for dist, shape in zip(distances, shapes))
    return any(pairs[i][1] >= pairs[i + 1][1] for i in range(len(pairs) - 1))


def _assign_top_rows(scaled_heights: list[int], height: int) -> list[int]:
    order = list(range(len(scaled_heights)))
    random.shuffle(order)
    min_gap = 1
    base = sum(scaled_heights[i] for i in order) + min_gap * (len(order) - 1)
    extra = height - base
    slack = [0] * (len(order) + 1)
    for _ in range(extra):
        slack[random.randrange(len(slack))] += 1
    tops = [0] * len(order)
    cursor = slack[0]
    for idx, sprite_index in enumerate(order):
        tops[sprite_index] = cursor
        cursor += scaled_heights[sprite_index]
        if idx < len(order) - 1:
            cursor += min_gap + slack[idx + 1]
    return tops


def _paint_bool_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    grid[top + coords[:, 0], left + coords[:, 1]] = int(color)


def _oracle_output(
    shape: tuple[int, int],
    body_mask: np.ndarray,
    body_top: int,
    body_left: int,
    sprites: list[dict],
    axis_col: int,
    side: str,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    _paint_bool_mask(out, body_mask, body_top, body_left, AXIS_COLOR)

    for sprite in sprites:
        color = int(sprite["color"])
        patch = np.where(sprite["shape"] != 0, color, BG).astype(int)
        factor = 1 << (sprite["distance"] - 1)
        scaled = np.kron(patch, np.ones((factor, factor), dtype=int))
        top = int(sprite["top"])
        if side == "left":
            right = int(sprite["right"])
            left = right - scaled.shape[1] + 1
        else:
            left = int(sprite["left"])
        _paint_patch(out, scaled, top, left, BG)
        for lr, lc in np.argwhere(scaled != BG):
            rr = top + int(lr)
            cc = left + int(lc)
            out[rr, 2 * axis_col - cc] = color
    return out


def _input_rows(sprite: dict) -> tuple[int, int]:
    return int(sprite["top"]), int(sprite["top"] + sprite["shape"].shape[0] - 1)


def _scaled_rows(sprite: dict) -> tuple[int, int]:
    factor = 1 << (int(sprite["distance"]) - 1)
    return int(sprite["top"]), int(sprite["top"] + sprite["shape"].shape[0] * factor - 1)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _choose_body_top(height: int, body_height: int, sprites: list[dict], half_width: int) -> int | None:
    candidate_starts = list(range(0, height - body_height + 1))
    valid = []
    overlap_pref = []

    for start in candidate_starts:
        end = start + body_height - 1
        if any(_intervals_overlap(start, end, *_input_rows(sprite)) for sprite in sprites):
            continue
        valid.append(start)
        if any(sprite["distance"] <= half_width and _intervals_overlap(start, end, *_scaled_rows(sprite)) for sprite in sprites):
            overlap_pref.append(start)

    if not valid:
        return None
    if overlap_pref and random.random() < 0.65:
        return random.choice(overlap_pref)
    return random.choice(valid)


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    side = random.choice(["left", "right"])
    sprite_count = 3 if random.random() < 0.6 else 2
    distances = sorted(random.sample([1, 2, 3], sprite_count))
    shapes = _choose_shapes(distances)
    if shapes is None:
        return None
    if not _areas_not_monotone(distances, shapes):
        return None

    colors = random.sample(list(SPRITE_COLORS), sprite_count)
    sprites = []
    scaled_heights = []
    scaled_widths = []
    for distance, shape, color in zip(distances, shapes, colors):
        factor = 1 << (distance - 1)
        scaled_heights.append(int(shape.shape[0] * factor))
        scaled_widths.append(int(shape.shape[1] * factor))
        sprites.append({"distance": distance, "shape": shape.copy(), "color": color})

    body_height = random.randint(5, 8)
    body_width = random.choice([5, 7])

    min_height = max(18, sum(scaled_heights) + (sprite_count - 1) + 2, body_height + 2)
    if min_height > 30:
        return None
    height = random.randint(min_height, min(30, min_height + 5))
    tops = _assign_top_rows(scaled_heights, height)
    for sprite, top in zip(sprites, tops):
        sprite["top"] = top

    half_span = max(max(distance + width - 1 for distance, width in zip(distances, scaled_widths)), body_width // 2)
    min_width = max(18, 2 * half_span + 1)
    if min_width > 30:
        return None
    width = random.randint(min_width, min(30, min_width + 5))
    extra = width - (2 * half_span + 1)
    left_extra = random.randint(0, extra)
    axis_col = half_span + left_extra

    body_mask = _make_body_mask(body_height, body_width)
    body_top = _choose_body_top(height, body_height, sprites, body_width // 2)
    if body_top is None:
        return None
    body_left = axis_col - body_width // 2

    for sprite in sprites:
        shape = sprite["shape"]
        distance = int(sprite["distance"])
        if side == "left":
            right = axis_col - distance
            left = right - shape.shape[1] + 1
            sprite["left"] = left
            sprite["right"] = right
        else:
            left = axis_col + distance
            right = left + shape.shape[1] - 1
            sprite["left"] = left
            sprite["right"] = right

    inp = np.full((height, width), BG, dtype=int)
    _paint_bool_mask(inp, body_mask, body_top, body_left, AXIS_COLOR)
    for sprite in sprites:
        color = int(sprite["color"])
        patch = np.where(sprite["shape"] != 0, color, BG).astype(int)
        top = int(sprite["top"])
        left = int(sprite["left"])
        right = int(sprite["right"])
        assert right < width and left >= 0
        for pr, pc in np.argwhere(patch != BG):
            rr = top + int(pr)
            cc = left + int(pc)
            if int(inp[rr, cc]) != BG:
                return None
            inp[rr, cc] = color

    expected = _oracle_output(inp.shape, body_mask, body_top, body_left, sprites, axis_col, side)

    if len(_components_of_mask(inp == AXIS_COLOR, connectivity=8)) != 1:
        return None
    detected_sprites = _monochrome_components(inp, ignored={BG, AXIS_COLOR})
    if len(detected_sprites) != sprite_count:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    distances_detected = []
    for comp in detected_sprites:
        r0, c0, r1, c1 = comp["bbox"]
        del r0, r1
        if side == "left":
            distances_detected.append(axis_col - c1)
        else:
            distances_detected.append(c0 - axis_col)
    if sorted(distances_detected) != distances:
        return None

    out_components = _monochrome_components(expected, ignored={BG, AXIS_COLOR})
    color_counts = Counter(comp["color"] for comp in out_components)
    if any(count < 2 for count in color_counts.values()):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input for puzzle 6")
