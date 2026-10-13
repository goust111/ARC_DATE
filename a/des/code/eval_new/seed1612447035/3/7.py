from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np


_RNG = np.random.default_rng()
_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_COLORS = tuple(range(10))
_MAX_GENERATION_ATTEMPTS = 400
_PAIR_COUNT_OPTIONS = (2, 3)
_PAIR_AREAS = (2, 3, 4, 5, 6, 7, 8)
_BAND_HEIGHT = 5
_BAND_GAP = 1
_MIN_VISIBLE_GAP = 4


def _mask(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


_SHAPE_BANK: dict[int, tuple[np.ndarray, ...]] = {
    2: (_mask(("##",)),),
    3: (_mask(("###",)), _mask(("#.", "##"))),
    4: (_mask(("##", "##")), _mask(("###", ".#.")), _mask(("#.", "#.", "##"))),
    5: (_mask((".#.", "###", ".#.")), _mask(("#..", "#..", "###")), _mask(("#.#", "###"))),
    6: (_mask(("###", "###")), _mask(("###", "##.")), _mask(("##.", ".##", ".#."))),
    7: (_mask(("###", "###", "#..")), _mask(("###", "##.", "##."))),
    8: (_mask(("###", "###", "##.")), _mask(("###", "###", ".##"))),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "area": int(arr.shape[0]),
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best: tuple[tuple[int, int], bytes] | None = None
    cur = np.asarray(mask, dtype=bool)
    for k in range(4):
        rot = np.rot90(cur, k)
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _detect_palette_bar(components: list[dict]) -> tuple[str, list[int], set[tuple[int, int]]]:
    top = [comp for comp in components if int(comp["area"]) == 1 and int(comp["bbox"][0]) == 0 and int(comp["bbox"][2]) == 0]
    left = [comp for comp in components if int(comp["area"]) == 1 and int(comp["bbox"][1]) == 0 and int(comp["bbox"][3]) == 0]

    top_valid = len(top) >= 2 and len({int(comp["color"]) for comp in top}) == len(top)
    left_valid = len(left) >= 2 and len({int(comp["color"]) for comp in left}) == len(left)
    assert top_valid != left_valid, "expected exactly one palette bar on the top row or left column"

    if top_valid:
        top.sort(key=lambda comp: int(comp["bbox"][1]))
        colors = [int(comp["color"]) for comp in top]
        coords = {(0, int(comp["bbox"][1])) for comp in top}
        return "top", colors, coords

    left.sort(key=lambda comp: int(comp["bbox"][0]))
    colors = [int(comp["color"]) for comp in left]
    coords = {(int(comp["bbox"][0]), 0) for comp in left}
    return "left", colors, coords


def _bbox_center(component: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, component["bbox"])
    return ((r0 + r1) // 2, (c0 + c1) // 2)


def _inclusive_range(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _ordered_centers(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if (a[1], a[0]) <= (b[1], b[0]) else (b, a)


def _l_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    cells = [(sr, c) for c in _inclusive_range(sc, ec)]
    cells.extend((r, ec) for r in _inclusive_range(sr, er))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each rotated-shape sprite pair with the palette color assigned by pair area.

    Input structure:
    - bg = most common color.
    - Several small monochrome 4-connected sprites (area 2-8) are scattered away from a border palette bar.
    - Exactly one palette bar made of distinct single pixels lies either on the top row or on the left column.
    - Every non-bar sprite has exactly one partner with the same bbox mask up to rotation and a different color.

    Rule:
    - Identify the palette bar and remove its single-pixel components from the sprite set.
    - Convert each remaining sprite to a bbox-relative binary mask and pair sprites whose masks match under 0/90/180/270 degree rotation.
    - Sort the sprite pairs by area from small to large and assign palette colors in bar order to those sorted pairs.
    - For each pair, take the two bbox centers, start from the leftmost center, draw a one-pixel-wide L path horizontal first and then vertical, and only paint background cells.

    Key insight:
    - Sprite color is a distractor: the real pairing signal is rotation-invariant shape, while the border palette bar provides the visible area-to-line-color mapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected non-background components"

    _bar_side, palette_colors, bar_coords = _detect_palette_bar(components)
    sprites = [
        comp
        for comp in components
        if tuple(map(int, comp["coords"][0])) not in bar_coords
    ]
    assert sprites and len(sprites) % 2 == 0
    assert all(2 <= int(comp["area"]) <= 8 for comp in sprites)

    grouped: defaultdict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    for comp in sprites:
        key = _canonical_rotation_key(_component_mask(comp))
        grouped[key].append(comp)

    pairs: list[tuple[int, list[dict]]] = []
    for members in grouped.values():
        assert len(members) == 2, "each shape must appear exactly twice"
        assert int(members[0]["color"]) != int(members[1]["color"])
        area = int(members[0]["area"])
        assert int(members[1]["area"]) == area
        pairs.append((area, members))

    areas = [area for area, _members in pairs]
    assert len(set(areas)) == len(areas), "pair areas must be unique to match the palette order"
    assert len(pairs) == len(palette_colors), "palette length must equal pair count"
    pairs.sort(key=lambda item: int(item[0]))

    out = g.copy()
    for line_color, (_area, members) in zip(palette_colors, pairs):
        center_a = _bbox_center(members[0])
        center_b = _bbox_center(members[1])
        start, end = _ordered_centers(center_a, center_b)
        for r, c in _l_path(start, end):
            if int(out[r, c]) == bg:
                out[r, c] = int(line_color)

    return out


def _rotated(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, int(_RNG.integers(0, 4)))


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {
        (top + int(r), left + int(c))
        for r, c in np.argwhere(mask)
    }


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _band_ranges(height: int, start_row: int, pair_count: int) -> list[tuple[int, int]] | None:
    min_used = pair_count * _BAND_HEIGHT + (pair_count - 1) * _BAND_GAP + 1
    slack = height - start_row - min_used
    if slack < 0:
        return None
    extras = _RNG.multinomial(int(slack), np.ones(pair_count + 1) / (pair_count + 1))
    bands: list[tuple[int, int]] = []
    row = start_row + int(extras[0])
    for i in range(pair_count):
        bands.append((row, row + _BAND_HEIGHT))
        row += _BAND_HEIGHT
        if i < pair_count - 1:
            row += _BAND_GAP + int(extras[i + 1])
    return bands


def _random_colors(pair_count: int) -> tuple[int, list[int], list[tuple[int, int]]]:
    bg = int(_RNG.choice(np.array(_COLORS, dtype=int)))
    pool = [color for color in _COLORS if color != bg]
    _RNG.shuffle(pool)
    palette = [int(color) for color in pool[:pair_count]]
    sprite_colors = [int(color) for color in pool[pair_count : pair_count + 2 * pair_count]]
    pairs = [(sprite_colors[2 * i], sprite_colors[2 * i + 1]) for i in range(pair_count)]
    return bg, palette, pairs


def _bar_positions(orientation: str, shape: tuple[int, int], pair_count: int) -> list[tuple[int, int]]:
    h, w = shape
    if orientation == "top":
        start = int(_RNG.integers(1, w - pair_count))
        return [(0, start + i) for i in range(pair_count)]
    start = int(_RNG.integers(1, h - pair_count))
    return [(start + i, 0) for i in range(pair_count)]


def _place_pair(
    band: tuple[int, int],
    template: np.ndarray,
    occupied: np.ndarray,
    reserved_paths: np.ndarray,
    inner_left: int,
    inner_right: int,
) -> tuple[np.ndarray, int, int, np.ndarray, int, int, set[tuple[int, int]]] | None:
    band_top, band_bottom = band
    mid = (inner_left + inner_right) // 2

    for _ in range(250):
        mask_a = _rotated(template)
        mask_b = _rotated(template)
        ha, wa = map(int, mask_a.shape)
        hb, wb = map(int, mask_b.shape)

        left_hi = mid - wa - _MIN_VISIBLE_GAP
        right_lo = mid + 1
        right_min = right_lo
        if left_hi < inner_left or inner_right - wb < right_min:
            continue

        left = int(_RNG.integers(inner_left, left_hi + 1))
        min_right = max(right_min, left + wa + _MIN_VISIBLE_GAP)
        max_right = inner_right - wb + 1
        if min_right > max_right:
            continue
        right = int(_RNG.integers(min_right, max_right + 1))

        top_a = int(_RNG.integers(band_top, band_bottom - ha + 1))
        top_b = int(_RNG.integers(band_top, band_bottom - hb + 1))

        coords_a = _coords_from_mask(mask_a, top_a, left)
        coords_b = _coords_from_mask(mask_b, top_b, right)
        if coords_a & coords_b:
            continue

        dilated = _dilate8(occupied)
        if any(dilated[r, c] for r, c in coords_a | coords_b):
            continue
        if any(reserved_paths[r, c] for r, c in coords_a | coords_b):
            continue

        center_a = (top_a + (ha - 1) // 2, left + (wa - 1) // 2)
        center_b = (top_b + (hb - 1) // 2, right + (wb - 1) // 2)
        start, end = _ordered_centers(center_a, center_b)
        path = set(_l_path(start, end))
        visible_path = path - coords_a - coords_b
        if any(occupied[r, c] or reserved_paths[r, c] for r, c in visible_path):
            continue

        return mask_a, top_a, left, mask_b, top_b, right, visible_path

    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.choice(np.array(_PAIR_COUNT_OPTIONS, dtype=int)))
    orientation = "top" if bool(_RNG.integers(0, 2)) else "left"

    min_height = (2 if orientation == "top" else 1) + pair_count * _BAND_HEIGHT + (pair_count - 1) * _BAND_GAP + 1
    min_width = 19 if orientation == "top" else 20
    height = int(_RNG.integers(min_height, 31))
    width = int(_RNG.integers(min_width, 31))

    bands = _band_ranges(height, 2 if orientation == "top" else 1, pair_count)
    if bands is None:
        return None

    bg, palette_colors, sprite_color_pairs = _random_colors(pair_count)
    chosen_areas = [int(v) for v in _RNG.choice(np.array(_PAIR_AREAS, dtype=int), size=pair_count, replace=False)]
    pair_specs = []
    for area, color_pair in zip(chosen_areas, sprite_color_pairs):
        templates = _SHAPE_BANK[int(area)]
        template = templates[int(_RNG.integers(0, len(templates)))]
        pair_specs.append({"area": int(area), "template": template, "colors": color_pair})
    _RNG.shuffle(pair_specs)

    grid = np.full((height, width), bg, dtype=int)
    bar_positions = _bar_positions(orientation, grid.shape, pair_count)
    for color, (r, c) in zip(palette_colors, bar_positions):
        grid[r, c] = int(color)

    occupied = np.zeros_like(grid, dtype=bool)
    for r, c in bar_positions:
        occupied[r, c] = True
    reserved_paths = np.zeros_like(grid, dtype=bool)

    inner_left = 2 if orientation == "top" else 3
    inner_right = width - 3
    if inner_right - inner_left + 1 < 14:
        return None

    for band, spec in zip(bands, pair_specs):
        placement = _place_pair(band, spec["template"], occupied, reserved_paths, inner_left, inner_right)
        if placement is None:
            return None
        mask_a, top_a, left_a, mask_b, top_b, left_b, visible_path = placement
        color_a, color_b = spec["colors"]
        _paint_mask(grid, mask_a, top_a, left_a, int(color_a))
        _paint_mask(grid, mask_b, top_b, left_b, int(color_b))
        occupied[top_a : top_a + mask_a.shape[0], left_a : left_a + mask_a.shape[1]][mask_a] = True
        occupied[top_b : top_b + mask_b.shape[0], left_b : left_b + mask_b.shape[1]][mask_b] = True
        for r, c in visible_path:
            reserved_paths[r, c] = True

    color_counts = Counter(map(int, grid.flat))
    if int(color_counts[bg]) <= max(count for color, count in color_counts.items() if int(color) != bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
