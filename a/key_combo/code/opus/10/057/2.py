import numpy as np
from collections import Counter, deque


_RNG = np.random.default_rng()
_BG = 0
_BAR_COLOR_POOL = (1, 2, 3, 4)
_MAX_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(cells)),
                }
            )
    return out


def _extract_bars(grid: np.ndarray, bg: int) -> list[dict]:
    h, _ = grid.shape
    bars = []
    for comp in _components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        if c0 != c1 or r1 != h - 1:
            continue
        height = r1 - r0 + 1
        if not (1 <= height <= 4 and comp["area"] == height):
            continue
        bars.append(
            {
                "color": int(comp["color"]),
                "col": int(c0),
                "height": int(height),
                "coords": comp["coords"],
            }
        )
    bars.sort(key=lambda bar: bar["col"])
    return bars


def _extract_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    h, _ = grid.shape
    sprites = []
    for comp in _components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        bbox_h = r1 - r0 + 1
        bbox_w = c1 - c0 + 1
        if r1 == h - 1 and bbox_w == 1 and comp["area"] == bbox_h:
            continue
        sprites.append(comp)
    return sprites


def _rotate_coords(coords: np.ndarray, k: int) -> np.ndarray:
    k %= 4
    if k == 0:
        return coords.copy()

    r0, c0, r1, c1 = _bbox(coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    rotated = []
    for r, c in coords:
        dr = float(r) - center_r
        dc = float(c) - center_c
        if k == 1:
            ndr, ndc = dc, -dr
        elif k == 2:
            ndr, ndc = -dr, -dc
        else:
            ndr, ndc = -dc, dr
        nr = center_r + ndr
        nc = center_c + ndc
        nri = int(round(nr))
        nci = int(round(nc))
        assert abs(nr - nri) < 1e-9 and abs(nc - nci) < 1e-9
        rotated.append((nri, nci))

    out = np.array(sorted(set(rotated)), dtype=int)
    assert len(out) == len(coords)
    return out


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    arr = np.array(coords, dtype=int)
    r0, c0, _, _ = _bbox(arr)
    arr = arr - np.array([r0, c0], dtype=int)
    pts = sorted((int(r), int(c)) for r, c in arr)
    return tuple(pts)


def _patch_from_coords(coords) -> np.ndarray:
    norm = np.array(_normalize(coords), dtype=int)
    r1 = int(norm[:, 0].max())
    c1 = int(norm[:, 1].max())
    patch = np.zeros((r1 + 1, c1 + 1), dtype=int)
    patch[norm[:, 0], norm[:, 1]] = 1
    return patch


def _rotated_shape(coords, k: int) -> tuple[tuple[int, int], ...]:
    patch = _patch_from_coords(coords)
    rot = np.rot90(patch, -k)
    pts = np.argwhere(rot == 1)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _canonical_shape_key(coords) -> tuple[tuple[int, int], ...]:
    return min(_rotated_shape(coords, k) for k in range(4))


def _embed_in_support(coords) -> tuple[tuple[int, int], ...]:
    norm = np.array(_normalize(coords), dtype=int)
    r0, c0, r1, c1 = _bbox(norm)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    roff = (3 - h) // 2
    coff = (3 - w) // 2
    shifted = norm + np.array([roff, coff], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_library() -> list[tuple[tuple[int, int], ...]]:
    raw = [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0)),
    ]
    shapes = []
    for coords in raw:
        key = _canonical_shape_key(coords)
        assert _rotated_shape(key, 1) != key
        support = _embed_in_support(key)
        support_arr = np.array(support, dtype=int)
        for k in range(4):
            rotated = _rotate_coords(support_arr, k)
            assert rotated[:, 0].min() >= 0 and rotated[:, 0].max() <= 2
            assert rotated[:, 1].min() >= 0 and rotated[:, 1].max() <= 2
        shapes.append(support)
    return shapes


_SHAPES = _shape_library()
_RECTANGULAR_SHAPES = [shape for shape in _SHAPES if len({pt[0] for pt in shape}) != len({pt[1] for pt in shape})]


def _expected_output_from_parts(
    grid: np.ndarray, sprite_specs: list[dict], bar_specs: list[dict]
) -> np.ndarray:
    out = grid.copy()
    for sprite in sprite_specs:
        for r, c in sprite["coords"]:
            out[r, c] = _BG
    for bar in bar_specs:
        for r, c in bar["coords"]:
            out[r, c] = _BG
    for sprite in sprite_specs:
        k = sprite["rotation"]
        rotated = _rotate_coords(np.array(sprite["coords"], dtype=int), k)
        for r, c in rotated:
            out[r, c] = sprite["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each upper sprite by the quarter-turn encoded in the height of the matching-color bottom bar, then erase all bars.

    Input structure:
    - The grid has a black background with 3-4 monochrome bottom-aligned vertical bars in the lower region.
    - Each bar is one cell wide, uses a distinct color, and its color also appears on exactly one separate 4-connected sprite in the upper region.
    - The upper sprites are small non-background components whose bounding boxes are at most 3x3 and are separated from the bars by blank rows.

    Rule:
    - Read each bottom bar's height h and convert it to a clockwise rotation count k = h mod 4.
    - Find the upper sprite of the same color and rotate that sprite in place around the center of its own bounding box by k quarter-turns.
    - After all matching-color sprite rotations are applied, erase every bottom bar to background.

    Key insight:
    - The bottom bars are not objects to preserve; they are color-addressed instructions whose heights encode the sprite rotations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    h, w = g.shape

    bars = _extract_bars(g, bg)
    assert 3 <= len(bars) <= 4
    assert len({bar["color"] for bar in bars}) == len(bars)
    assert len({bar["height"] for bar in bars}) == len(bars)
    assert all(i == 0 or bars[i]["col"] - bars[i - 1]["col"] >= 3 for i in range(len(bars)))

    sprites = _extract_sprites(g, bg)
    assert len(sprites) == len(bars)
    assert all(3 <= sprite["area"] <= 6 for sprite in sprites)
    assert all((sprite["bbox"][2] - sprite["bbox"][0] + 1) <= 3 for sprite in sprites)
    assert all((sprite["bbox"][3] - sprite["bbox"][1] + 1) <= 3 for sprite in sprites)
    assert all(sprite["bbox"][2] <= h - 4 for sprite in sprites)

    bars_by_color = {bar["color"]: bar for bar in bars}
    sprites_by_color = {int(sprite["color"]): sprite for sprite in sprites}
    assert set(bars_by_color) == set(sprites_by_color)

    out = g.copy()
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = bg
    for bar in bars:
        for r, c in bar["coords"]:
            out[r, c] = bg

    occupied = set()
    for color, sprite in sprites_by_color.items():
        k = bars_by_color[color]["height"] % 4
        rotated = _rotate_coords(sprite["coords"], k)
        for r, c in rotated:
            assert 0 <= r < h and 0 <= c < w
            occupied.add((int(r), int(c)))
        for r, c in rotated:
            out[r, c] = color

    total_area = sum(sprite["area"] for sprite in sprites)
    assert len(occupied) == total_area
    return out


def _slot_positions(height: int, width: int, max_row: int) -> list[tuple[int, int]]:
    slots = [(r, c) for r in range(max_row + 1) for c in range(width - 2)]
    _RNG.shuffle(slots)
    return slots


def _can_use_slot(slot: tuple[int, int], used: list[tuple[int, int]]) -> bool:
    r, c = slot
    for ur, uc in used:
        if not (r + 3 <= ur - 1 or ur + 3 <= r - 1 or c + 3 <= uc - 1 or uc + 3 <= c - 1):
            return False
    return True


def _place_sprites(
    height: int, width: int, colors: list[int], heights: list[int]
) -> list[dict] | None:
    max_bar_height = max(heights)
    max_sprite_row = height - max_bar_height - 3
    slot_row_max = max_sprite_row - 2
    if slot_row_max < 0:
        return None

    canonical_choices = [_canonical_shape_key(shape) for shape in _SHAPES]
    rect_choices = [_canonical_shape_key(shape) for shape in _RECTANGULAR_SHAPES]
    chosen_keys = []
    chosen_keys.append(rect_choices[int(_RNG.integers(len(rect_choices)))])
    pool = [key for key in canonical_choices if key != chosen_keys[0]]
    _RNG.shuffle(pool)
    chosen_keys.extend(pool[: len(colors) - 1])
    _RNG.shuffle(chosen_keys)

    rotating_colors = [color for color, bar_h in zip(colors, heights) if bar_h % 4 in (1, 3)]
    if rotating_colors:
        rectangular_color = rotating_colors[int(_RNG.integers(len(rotating_colors)))]
    else:
        rectangular_color = colors[0]
    rect_index = colors.index(rectangular_color)
    rect_key = chosen_keys[0]
    chosen_keys[0], chosen_keys[rect_index] = chosen_keys[rect_index], chosen_keys[0]
    chosen_keys[rect_index] = rect_key

    used_slots: list[tuple[int, int]] = []
    sprites = []
    for color, key in zip(colors, chosen_keys):
        support_shape = _embed_in_support(_rotated_shape(key, int(_RNG.integers(4))))
        coords_local = np.array(support_shape, dtype=int)
        candidates = _slot_positions(height, width, slot_row_max)
        placed = None
        for slot in candidates:
            if not _can_use_slot(slot, used_slots):
                continue
            sr, sc = slot
            coords = coords_local + np.array([sr, sc], dtype=int)
            all_ok = True
            for k in range(4):
                rot = _rotate_coords(coords, k)
                if rot[:, 0].min() < 0 or rot[:, 0].max() >= height or rot[:, 1].min() < 0 or rot[:, 1].max() >= width:
                    all_ok = False
                    break
            if all_ok:
                placed = {"color": color, "coords": [tuple(map(int, p)) for p in coords]}
                used_slots.append(slot)
                sprites.append(placed)
                break
        if placed is None:
            return None

    sprites.sort(key=lambda sprite: min(c for _, c in sprite["coords"]))
    return sprites


def _place_bars(height: int, width: int, colors: list[int], heights: list[int]) -> list[dict] | None:
    candidate_cols = list(range(1, width - 1))
    _RNG.shuffle(candidate_cols)
    bars = []
    for color, bar_h in zip(colors, heights):
        for col in candidate_cols:
            if any(abs(col - other["col"]) < 3 for other in bars):
                continue
            coords = [(r, col) for r in range(height - bar_h, height)]
            bars.append({"color": color, "height": bar_h, "col": col, "coords": coords})
            break
        else:
            return None
    bars.sort(key=lambda bar: bar["col"])
    return bars


def _build_grid(height: int, width: int, sprites: list[dict], bars: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for sprite in sprites:
        for r, c in sprite["coords"]:
            grid[r, c] = sprite["color"]
    for bar in bars:
        for r, c in bar["coords"]:
            assert grid[r, c] == _BG
            grid[r, c] = bar["color"]
    return grid


def _validate_instance(grid: np.ndarray, sprites: list[dict], bars: list[dict], expected: np.ndarray) -> bool:
    bg = _get_bg(grid)
    parsed_bars = _extract_bars(grid, bg)
    parsed_sprites = _extract_sprites(grid, bg)
    if len(parsed_bars) != len(bars) or len(parsed_sprites) != len(sprites):
        return False
    if len({bar["height"] for bar in parsed_bars}) != len(parsed_bars):
        return False
    if len({bar["color"] for bar in parsed_bars}) != len(parsed_bars):
        return False

    sprite_order = [sprite["color"] for sprite in sorted(parsed_sprites, key=lambda s: s["bbox"][1])]
    bar_order = [bar["color"] for bar in parsed_bars]
    if sprite_order == bar_order:
        return False

    height_map = {bar["color"]: bar["height"] % 4 for bar in parsed_bars}
    if len(set(height_map.values())) < min(3, len(parsed_bars)):
        return False
    if not any(k == 0 for k in height_map.values()):
        return False
    if not any(k in (1, 3) for k in height_map.values()):
        return False

    rectangular_swap = False
    for sprite in parsed_sprites:
        coords = sprite["coords"]
        r0, c0, r1, c1 = sprite["bbox"]
        bbox_h = r1 - r0 + 1
        bbox_w = c1 - c0 + 1
        k = height_map[sprite["color"]]
        rotated = _rotate_coords(coords, k)
        if k in (1, 3) and bbox_h != bbox_w:
            rr0, cc0, rr1, cc1 = _bbox(rotated)
            if (rr1 - rr0 + 1) == bbox_w and (cc1 - cc0 + 1) == bbox_h:
                rectangular_swap = True
    if not rectangular_swap:
        return False

    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(11, 15))
    width = int(_RNG.integers(14, 19))
    count = 4 if _RNG.random() < 0.65 else 3

    colors = list(_RNG.choice(_BAR_COLOR_POOL, size=count, replace=False))
    if count == 4:
        heights = [1, 2, 3, 4]
        _RNG.shuffle(heights)
    else:
        heights = list(_RNG.choice([1, 2, 3, 4], size=3, replace=False))
        _RNG.shuffle(heights)

    bars = _place_bars(height, width, colors, heights)
    if bars is None:
        return None

    sprites = _place_sprites(height, width, colors, heights)
    if sprites is None:
        return None

    sprite_specs = []
    for sprite in sprites:
        rotation = next(bar["height"] % 4 for bar in bars if bar["color"] == sprite["color"])
        sprite_specs.append({"color": sprite["color"], "coords": sprite["coords"], "rotation": rotation})

    grid = _build_grid(height, width, sprites, bars)
    expected = _expected_output_from_parts(grid, sprite_specs, bars)
    if not _validate_instance(grid, sprite_specs, bars, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
