from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


# Every dictionary shape has the same bbox and the same area, so matching by
# size or pixel count is insufficient; the exact normalized mask is required.
TOP_SHAPES = [
    _mask(["###", ".#.", ".#."]),
    _mask([".#.", "###", ".#."]),
    _mask(["#..", "###", "#.."]),
    _mask(["..#", "###", "..#"]),
    _mask(["#..", "#..", "###"]),
    _mask(["..#", "..#", "###"]),
    _mask(["###", "#..", "#.."]),
    _mask(["###", "..#", "..#"]),
    _mask(["##.", ".#.", ".##"]),
    _mask([".##", ".#.", "##."]),
    _mask(["##.", ".##", ".#."]),
    _mask([".##", "##.", ".#."]),
]

BOTTOM_SHAPES = [
    _mask(["##", "##"]),
    _mask(["###"]),
    _mask(["#", "#", "#"]),
    _mask(["##", "#."]),
    _mask(["###", ".#."]),
    _mask(["##.", ".##"]),
    _mask([".#.", "###"]),
    _mask(["#.", "##"]),
    _mask(["###", "###"]),
    _mask([".#.", "###", ".#."]),
]


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return objects


def _shape_key(coords: np.ndarray) -> tuple[int, int, bytes]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.uint8)
    shifted = coords - np.array([[r0, c0]])
    mask[shifted[:, 0], shifted[:, 1]] = 1
    return int(mask.shape[0]), int(mask.shape[1]), mask.tobytes()


def _find_separator_rows(grid: np.ndarray) -> tuple[int, tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, tuple[int, int]]] = []
    for color in np.unique(grid):
        rows = [r for r in range(h) if np.all(grid[r] == color)]
        if len(rows) == 2 and int(np.count_nonzero(grid == color)) == 2 * w:
            candidates.append((int(color), (int(rows[0]), int(rows[1]))))
    assert len(candidates) == 1, f"expected one separator color, got {candidates}"
    return candidates[0]


def _panel_shape_colors(panel: np.ndarray, bg: int) -> tuple[dict, dict]:
    shape_to_color: dict[tuple[int, int, bytes], int] = {}
    shape_to_count: dict[tuple[int, int, bytes], int] = {}
    for obj in _find_components(panel, bg):
        key = _shape_key(obj["coords"])
        shape_to_count[key] = shape_to_count.get(key, 0) + 1
        color = int(obj["color"])
        if key in shape_to_color:
            assert shape_to_color[key] == color, "duplicate shapes must keep one color per panel"
        else:
            shape_to_color[key] = color
    return shape_to_color, shape_to_count


def _derive_color_map(top: np.ndarray, middle: np.ndarray, bg: int) -> dict[int, int]:
    top_shapes, top_counts = _panel_shape_colors(top, bg)
    mid_shapes, mid_counts = _panel_shape_colors(middle, bg)
    assert top_counts == mid_counts, "top and middle panels must share the same shape multiset"

    color_map: dict[int, int] = {}
    inverse: dict[int, int] = {}
    for key, top_color in top_shapes.items():
        assert key in mid_shapes, "missing partner shape in middle panel"
        mid_color = int(mid_shapes[key])
        if top_color in color_map:
            assert color_map[top_color] == mid_color, "a top color mapped to multiple middle colors"
        else:
            color_map[top_color] = mid_color
        if mid_color in inverse:
            assert inverse[mid_color] == top_color, "a middle color matched multiple top colors"
        else:
            inverse[mid_color] = top_color
    return color_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the lower panel with the color dictionary hidden in top-middle shape matches.

    Input structure:
    - Two full-width grey separator rows split the grid into top, middle, and bottom panels.
    - The top panel contains several separated monochrome shapes.
    - The middle panel contains the same normalized shape multiset in different positions and different colors.
    - The bottom panel is a separate drawing that uses the top-panel colors.

    Rule:
    - Match each top-panel object with the middle-panel object that has the same bbox-normalized pixel mask.
    - Read each matched pair as one entry of a top-color to middle-color dictionary.
    - Apply that color dictionary cellwise to the bottom panel and output only the recolored bottom panel.

    Key insight:
    - The useful information is not position or size; the hidden recoloring rule is encoded only by exact shape pairing across the first two panels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_color, (sep1, sep2) = _find_separator_rows(g)
    bg = _most_common(g[g != sep_color])

    top = g[:sep1]
    middle = g[sep1 + 1 : sep2]
    bottom = g[sep2 + 1 :]
    assert top.size > 0 and middle.size > 0 and bottom.size > 0, "all three panels must be non-empty"

    color_map = _derive_color_map(top, middle, bg)
    out = bottom.copy()
    for src, dst in color_map.items():
        out[bottom == src] = dst
    return out


def _paint_sprite(canvas: np.ndarray, sprite: np.ndarray, color: int, r0: int, c0: int) -> None:
    view = canvas[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]]
    view[sprite] = int(color)


def _place_specs(
    panel_shape: tuple[int, int],
    specs: list[tuple[np.ndarray, int, int]],
    rng: np.random.Generator,
    *,
    padding: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    h, w = panel_shape
    panel = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int, int]] = []
    order = list(range(len(specs)))
    rng.shuffle(order)

    for idx in order:
        sprite, color, sprite_id = specs[idx]
        sh, sw = sprite.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - sh + 1):
            for c0 in range(w - sw + 1):
                rr0 = max(0, r0 - padding)
                cc0 = max(0, c0 - padding)
                rr1 = min(h, r0 + sh + padding)
                cc1 = min(w, c0 + sw + padding)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        _paint_sprite(panel, sprite, color, r0, c0)
        occupied[r0 : r0 + sh, c0 : c0 + sw] |= sprite
        placements.append((int(sprite_id), int(r0), int(c0)))
    return panel, placements


def _mapping_with_swap(colors: list[int], rng: np.random.Generator) -> dict[int, int]:
    shuffled = list(colors)
    rng.shuffle(shuffled)
    if len(shuffled) == 4:
        a, b, c, d = shuffled
        return {a: b, b: a, c: d, d: c}

    assert len(shuffled) == 5
    a, b, c, d, e = shuffled
    if int(rng.integers(2)) == 0:
        cycle = {c: d, d: e, e: c}
    else:
        cycle = {c: e, e: d, d: c}
    return {a: b, b: a, **cycle}


def _row_major_ids(placements: list[tuple[int, int, int]]) -> list[int]:
    return [sprite_id for sprite_id, _r, _c in sorted(placements, key=lambda item: (item[1], item[2], item[0]))]


def _bottom_panel(
    panel_shape: tuple[int, int],
    colors: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    object_count = len(colors) + int(rng.integers(2, 5))
    chosen_colors = list(colors)
    while len(chosen_colors) < object_count:
        chosen_colors.append(int(rng.choice(colors)))
    rng.shuffle(chosen_colors)

    specs: list[tuple[np.ndarray, int, int]] = []
    for idx, color in enumerate(chosen_colors):
        sprite = BOTTOM_SHAPES[int(rng.integers(len(BOTTOM_SHAPES)))]
        specs.append((sprite, int(color), idx))
    placed = _place_specs(panel_shape, specs, rng, padding=1)
    if placed is None:
        return None
    return placed[0]


def _expected_bottom(bottom: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    out = bottom.copy()
    for src, dst in color_map.items():
        out[bottom == src] = dst
    return out


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_colors = int(rng.integers(4, 6))
    colors = [int(c) for c in rng.choice(PALETTE, size=n_colors, replace=False)]
    color_map = _mapping_with_swap(colors, rng)

    shape_ids = [int(i) for i in rng.choice(len(TOP_SHAPES), size=n_colors, replace=False)]
    top_specs = [(TOP_SHAPES[sid], colors[i], sid) for i, sid in enumerate(shape_ids)]
    mid_specs = [(TOP_SHAPES[sid], color_map[colors[i]], sid) for i, sid in enumerate(shape_ids)]

    width = int(rng.integers(15, 23))
    top_h = int(rng.integers(7, 10))
    mid_h = int(rng.integers(7, 10))
    bottom_h = int(rng.integers(7, 11))

    top_placed = _place_specs((top_h, width), top_specs, rng, padding=1)
    if top_placed is None:
        return None
    mid_placed = _place_specs((mid_h, width), mid_specs, rng, padding=1)
    if mid_placed is None:
        return None
    bottom = _bottom_panel((bottom_h, width), colors, rng)
    if bottom is None:
        return None

    top_panel, top_places = top_placed
    mid_panel, mid_places = mid_placed
    if _row_major_ids(top_places) == _row_major_ids(mid_places):
        return None
    if any((tr, tc) == (mr, mc) for (_, tr, tc), (_, mr, mc) in zip(sorted(top_places), sorted(mid_places))):
        return None

    grid = np.full((top_h + mid_h + bottom_h + 2, width), BG, dtype=int)
    grid[:top_h] = top_panel
    grid[top_h] = SEP
    grid[top_h + 1 : top_h + 1 + mid_h] = mid_panel
    grid[top_h + 1 + mid_h] = SEP
    grid[top_h + 2 + mid_h :] = bottom
    expected = _expected_bottom(bottom, color_map)
    return grid, expected


def _semantic_ok(grid: np.ndarray, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if expected.ndim != 2 or max(expected.shape) > 30:
        return False
    if not np.array_equal(main(grid), expected):
        return False

    sep_color, (sep1, sep2) = _find_separator_rows(grid)
    if sep_color != SEP:
        return False
    bg = _most_common(grid[grid != sep_color])
    top = grid[:sep1]
    middle = grid[sep1 + 1 : sep2]
    bottom = grid[sep2 + 1 :]
    top_objects = _find_components(top, bg)
    middle_objects = _find_components(middle, bg)
    if len(top_objects) < 4 or len(top_objects) != len(middle_objects):
        return False

    top_keys = [_shape_key(obj["coords"]) for obj in top_objects]
    mid_keys = [_shape_key(obj["coords"]) for obj in middle_objects]
    if len(set(top_keys)) != len(top_keys):
        return False
    if sorted(top_keys) != sorted(mid_keys):
        return False

    features = {(obj["area"], obj["bbox"][2] - obj["bbox"][0] + 1, obj["bbox"][3] - obj["bbox"][1] + 1) for obj in top_objects}
    if len(features) != 1:
        return False

    color_map = _derive_color_map(top, middle, bg)
    if any(src == dst for src, dst in color_map.items()):
        return False
    if not any(color_map.get(dst) == src for src, dst in color_map.items()):
        return False

    top_order = [_shape_key(obj["coords"]) for obj in sorted(top_objects, key=lambda o: (o["bbox"][0], o["bbox"][1]))]
    mid_order = [_shape_key(obj["coords"]) for obj in sorted(middle_objects, key=lambda o: (o["bbox"][0], o["bbox"][1]))]
    if top_order == mid_order and len(top_order) > 1:
        return False

    bottom_colors = set(int(v) for v in np.unique(bottom) if int(v) != bg)
    if bottom_colors != set(color_map):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        result = _gen_attempt(rng)
        if result is None:
            continue
        grid, expected = result
        if _semantic_ok(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid shape-pairing dictionary input")
