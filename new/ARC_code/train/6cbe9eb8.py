from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, paste


def _rect_item(g: np.ndarray, color: int) -> dict:
    coords = np.argwhere(g == int(color))
    assert coords.size, "expected a present color"
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = h * w

    rel = coords - np.array([r0, c0])
    on_border = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == h - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == w - 1)
    )
    if int(coords.shape[0]) == area:
        kind = "solid"
    else:
        assert bool(np.all(on_border)), "foreground colors must form rectangles or rectangular frames"
        kind = "frame"

    return {
        "color": int(color),
        "kind": kind,
        "height": h,
        "width": w,
        "area": area,
    }


def _foreground_rectangles(g: np.ndarray) -> list[dict]:
    items = []
    for color in np.unique(g):
        comps = components_mask(g == int(color), connectivity=4)
        sizes = [int(comp.shape[0]) for comp in comps]
        assert sizes, "unique colors must have at least one component"

        # Wallpaper colors appear as many tiny disconnected islands; overlaid
        # rectangle colors have one large component, sometimes split by occlusion.
        if max(sizes) <= 3:
            continue
        assert len(comps) <= 4, "foreground rectangle colors have only a few visible components"
        items.append(_rect_item(g, int(color)))

    assert len(items) >= 2, "expected at least two nested foreground rectangles"
    areas = [item["area"] for item in items]
    assert len(areas) == len(set(areas)), "rectangle areas must define a unique nesting order"
    items.sort(key=lambda item: int(item["area"]), reverse=True)
    return items


def _paint_item(out: np.ndarray, item: dict, r: int, c: int) -> tuple[int, int, int, int]:
    color = int(item["color"])
    h = int(item["height"])
    w = int(item["width"])
    assert 0 <= r and 0 <= c and r + h <= out.shape[0] and c + w <= out.shape[1]

    if item["kind"] == "solid":
        out[r : r + h, c : c + w] = color
        return r, c, h, w

    assert item["kind"] == "frame"
    assert h >= 3 and w >= 3, "frames must have an interior"
    out[r, c : c + w] = color
    out[r + h - 1, c : c + w] = color
    out[r : r + h, c] = color
    out[r : r + h, c + w - 1] = color
    return r + 1, c + 1, h - 2, w - 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the foreground rectangles as a largest-to-smallest lower-left nesting.

    Input structure
    - A periodic wallpaper background is overlaid by foreground colors that each form a rectangular frame or solid rectangle.
    - The largest foreground rectangle is the output frame; smaller foreground rectangles are the objects to nest inside it.
    - A frame may be partially hidden by a smaller rectangle, but its visible pixels still determine its bbox.

    Rule
    - Ignore the wallpaper and crop the result to the largest foreground rectangle's bbox size.
    - Sort foreground rectangles by bbox area from largest to smallest.
    - Reconstruct each frame or solid rectangle, placing every next smaller rectangle in the lower-left of the current rectangle's usable region.

    Key insight
    - The scattered background colors are not objects; the meaningful order is the unique descending rectangle size chain, and each step nests into the previous rectangle's lower-left interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    items = _foreground_rectangles(g)
    outer = items[0]
    out = np.zeros((int(outer["height"]), int(outer["width"])), dtype=int)

    region = _paint_item(out, outer, 0, 0)
    for item in items[1:]:
        rr, cc, rh, rw = region
        h = int(item["height"])
        w = int(item["width"])
        assert h <= rh and w <= rw, "nested rectangle must fit in parent usable region"
        r = rr + rh - h
        c = cc
        region = _paint_item(out, item, r, c)

    return out

# --------------- generator (att2) ---------------

_RNG = np.random.default_rng()

# Colors and sentinels.
COLOR_VALUES, FOREGROUND_COLORS = tuple(range(10)), tuple(range(1, 10))
TRANSPARENT = -1
# Rectangle chain counts.
ITEM_COUNT_RANGE, THREE_ITEM_FRAME_OPTIONS = (3, 4), (1, 2)
FOUR_ITEM_FRAME_COUNT = 2
# Solid rectangle dimensions.
INNER_SOLID_HEIGHT_RANGE, INNER_SOLID_WIDTH_RANGE = (2, 3), (2, 3)
SECOND_SOLID_MAX_HEIGHT, SECOND_SOLID_MAX_WIDTH = 9, 9
# Frame growth and bounds.
FRAME_MIN_GROWTH, INNER_FRAME_MAX_GROWTH = 2, 5
INNER_FRAME_MAX_HEIGHT, INNER_FRAME_MAX_WIDTH = 13, 11
OUTER_FRAME_MIN_HEIGHT, OUTER_FRAME_MIN_WIDTH = 8, 8
OUTER_FRAME_MAX_HEIGHT, OUTER_FRAME_MAX_WIDTH = 15, 13
# Wallpaper canvas bounds and margin around the outer rectangle.
CANVAS_HEIGHT_RANGE, CANVAS_WIDTH_RANGE = (13, 19), (22, 27)
CANVAS_BORDER_MARGIN = 1
# Wallpaper tile/color sampling.
WALLPAPER_COLOR_COUNT_RANGE = (2, 4)
ZERO_WALLPAPER_ROLL_RANGE = (0, 4)
ZERO_WALLPAPER_THRESHOLD = 3
TILE_SIZE_2 = 2
TILE_SIZE_OPTIONS, TILE3_SIGN_OPTIONS = (2, 3), (1, 2)
# Visibility/oracle retry limits.
VISIBLE_OUTER_MIN_COMPONENT, MAX_VISIBLE_OUTER_COMPONENTS = 3, 4
GENERATION_RETRY_LIMIT = 300


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _sample(seq, k: int) -> list[int]:
    idx = _RNG.choice(len(seq), size=int(k), replace=False)
    return [int(seq[int(i)]) for i in idx]


def _shuffled(seq) -> list[int]:
    vals = list(seq)
    _RNG.shuffle(vals)
    return [int(v) for v in vals]


def _choose_dims(
    h_min: int,
    h_max: int,
    w_min: int,
    w_max: int,
    used_areas: set[int],
    *,
    min_area: int = 0,
) -> tuple[int, int] | None:
    pairs = [
        (h, w)
        for h in range(int(h_min), int(h_max) + 1)
        for w in range(int(w_min), int(w_max) + 1)
        if h * w > int(min_area) and h * w not in used_areas
    ]
    return _choice(pairs) if pairs else None


def _build_items() -> list[dict]:
    """Build a largest-to-smallest rectangle chain with unique bbox areas."""
    n_items = _randint(*ITEM_COUNT_RANGE)
    if n_items == ITEM_COUNT_RANGE[1]:
        n_frames = FOUR_ITEM_FRAME_COUNT
    else:
        n_frames = _choice(THREE_ITEM_FRAME_OPTIONS)
    n_solids = n_items - n_frames

    colors = _sample(FOREGROUND_COLORS, n_items)
    chain = []
    used_areas: set[int] = set()

    h = _randint(*INNER_SOLID_HEIGHT_RANGE)
    w = _randint(*INNER_SOLID_WIDTH_RANGE)
    chain.append({"kind": "solid", "height": h, "width": w})
    used_areas.add(h * w)

    if n_solids == 2:
        dims = _choose_dims(h, SECOND_SOLID_MAX_HEIGHT, w, SECOND_SOLID_MAX_WIDTH, used_areas, min_area=h * w)
        if dims is None:
            return []
        h, w = dims
        chain.append({"kind": "solid", "height": h, "width": w})
        used_areas.add(h * w)

    if n_frames == 2:
        dims = _choose_dims(
            h + FRAME_MIN_GROWTH,
            min(INNER_FRAME_MAX_HEIGHT, h + INNER_FRAME_MAX_GROWTH),
            w + FRAME_MIN_GROWTH,
            min(INNER_FRAME_MAX_WIDTH, w + INNER_FRAME_MAX_GROWTH),
            used_areas,
        )
        if dims is None:
            return []
        h, w = dims
        chain.append({"kind": "frame", "height": h, "width": w})
        used_areas.add(h * w)

    dims = _choose_dims(
        max(OUTER_FRAME_MIN_HEIGHT, h + FRAME_MIN_GROWTH),
        OUTER_FRAME_MAX_HEIGHT,
        max(OUTER_FRAME_MIN_WIDTH, w + FRAME_MIN_GROWTH),
        OUTER_FRAME_MAX_WIDTH,
        used_areas,
    )
    if dims is None:
        return []
    h, w = dims
    chain.append({"kind": "frame", "height": h, "width": w})

    items = list(reversed(chain))
    for item, color in zip(items, colors):
        item["color"] = int(color)
        item["area"] = int(item["height"] * item["width"])
    return items


def _item_size(item: dict) -> tuple[int, int]:
    return int(item["height"]), int(item["width"])


def _paint_item_view(view: np.ndarray, item: dict) -> None:
    color = int(item["color"])
    if item["kind"] == "solid":
        view[:, :] = color
    else:
        view[0, :] = color
        view[-1, :] = color
        view[:, 0] = color
        view[:, -1] = color


def _gen_paint_item(out: np.ndarray, item: dict, r: int, c: int) -> tuple[int, int, int, int]:
    h, w = _item_size(item)
    _paint_item_view(out[r : r + h, c : c + w], item)
    if item["kind"] == "solid":
        return r, c, h, w

    return r + 1, c + 1, h - 2, w - 2


def _render_output(items: list[dict]) -> np.ndarray:
    outer = items[0]
    out = np.zeros((int(outer["height"]), int(outer["width"])), dtype=int)
    region = _gen_paint_item(out, outer, 0, 0)
    for item in items[1:]:
        rr, cc, rh, rw = region
        h = int(item["height"])
        w = int(item["width"])
        region = _gen_paint_item(out, item, rr + rh - h, cc)
    return out


def _rect_patch(item: dict) -> np.ndarray:
    h, w = _item_size(item)
    patch = np.full((h, w), TRANSPARENT, dtype=int)
    _paint_item_view(patch, item)
    return patch


def _tile2(colors: list[int]) -> np.ndarray:
    colors = _shuffled(colors)
    if len(colors) == 2:
        a, b = colors
        return np.array([[a, b], [b, a]], dtype=int)
    if len(colors) == 3:
        a, b, c = colors
        if _randint(0, 1):
            return np.array([[a, b], [c, a]], dtype=int)
        return np.array([[b, a], [a, c]], dtype=int)
    return np.array(colors, dtype=int).reshape(2, 2)


def _tile3(colors: list[int]) -> np.ndarray:
    colors = _shuffled(colors)
    sign = _choice(TILE3_SIGN_OPTIONS)
    tile = np.array(
        [[colors[(r + sign * c) % 3] for c in range(3)] for r in range(3)],
        dtype=int,
    )
    if len(colors) == 4:
        tile[_randint(0, TILE_SIZE_2), _randint(0, TILE_SIZE_2)] = colors[3]
    return tile


def _wallpaper(rows: int, cols: int, fg_colors: set[int]) -> np.ndarray:
    available = [c for c in COLOR_VALUES if c not in fg_colors]
    k = _randint(*WALLPAPER_COLOR_COUNT_RANGE)
    if 0 in available and _randint(*ZERO_WALLPAPER_ROLL_RANGE) < ZERO_WALLPAPER_THRESHOLD:
        colors = [0] + _sample([c for c in available if c != 0], k - 1)
    else:
        colors = _sample(available, k)

    tile_size = TILE_SIZE_2 if k == WALLPAPER_COLOR_COUNT_RANGE[0] else _choice(TILE_SIZE_OPTIONS)
    tile = _tile2(colors) if tile_size == TILE_SIZE_2 else _tile3(colors)
    reps = (rows // tile.shape[0] + 1, cols // tile.shape[1] + 1)
    return np.tile(tile, reps)[:rows, :cols].copy()


def _placed_mask(mask: np.ndarray, rows: int, cols: int, r: int, c: int) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=bool)
    h, w = mask.shape
    out[r : r + h, c : c + w] = mask
    return out


def _outer_survives(outer_mask: np.ndarray, item_mask: np.ndarray, r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    visible = outer_mask.copy()
    h, w = item_mask.shape
    visible[r : r + h, c : c + w] &= ~item_mask
    r0, c0, rh, rw = box
    if not visible[r0, c0 : c0 + rw].any():
        return False
    if not visible[r0 + rh - 1, c0 : c0 + rw].any():
        return False
    if not visible[r0 : r0 + rh, c0].any():
        return False
    if not visible[r0 : r0 + rh, c0 + rw - 1].any():
        return False
    sizes = [int(comp.shape[0]) for comp in components_mask(visible, connectivity=4)]
    return bool(sizes) and max(sizes) > VISIBLE_OUTER_MIN_COMPONENT and len(sizes) <= MAX_VISIBLE_OUTER_COMPONENTS


def _overlap_positions(mask: np.ndarray, outer_mask: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    rows, cols = outer_mask.shape
    h, w = mask.shape
    positions = []
    for r in range(rows - h + 1):
        for c in range(cols - w + 1):
            if not np.any(outer_mask[r : r + h, c : c + w] & mask):
                continue
            if _outer_survives(outer_mask, mask, r, c, box):
                positions.append((r, c))
    return positions


def _free_positions(mask: np.ndarray, occupied: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = occupied.shape
    h, w = mask.shape
    positions = []
    for r in range(rows - h + 1):
        for c in range(cols - w + 1):
            if not np.any(occupied[r : r + h, c : c + w] & mask):
                positions.append((r, c))
    return positions


def _build_canvas(items: list[dict]) -> np.ndarray:
    """Create a wallpaper canvas large enough to leave a border around the outer frame."""
    outer_h, outer_w = _item_size(items[0])
    rows = _randint(max(CANVAS_HEIGHT_RANGE[0], outer_h + 2 * CANVAS_BORDER_MARGIN), CANVAS_HEIGHT_RANGE[1])
    cols = _randint(max(CANVAS_WIDTH_RANGE[0], outer_w + 2 * CANVAS_BORDER_MARGIN), CANVAS_WIDTH_RANGE[1])
    fg_colors = {int(item["color"]) for item in items}
    return _wallpaper(rows, cols, fg_colors)


def _choose_placements(items: list[dict], masks: list[np.ndarray], rows: int, cols: int) -> list[tuple[int, int, int]] | None:
    """Scatter rectangles, forcing one valid overlap with the outer frame and keeping the rest disjoint."""
    outer_h, outer_w = _item_size(items[0])
    outer_r = _randint(CANVAS_BORDER_MARGIN, rows - outer_h - CANVAS_BORDER_MARGIN)
    outer_c = _randint(CANVAS_BORDER_MARGIN, cols - outer_w - CANVAS_BORDER_MARGIN)
    outer_mask = _placed_mask(masks[0], rows, cols, outer_r, outer_c)
    occupied = outer_mask.copy()
    placements = [(0, outer_r, outer_c)]

    overlap_i = _choice(list(range(1, len(items))))
    choices = _overlap_positions(masks[overlap_i], outer_mask, (outer_r, outer_c, outer_h, outer_w))
    if not choices:
        return None
    r, c = _choice(choices)
    placements.append((overlap_i, r, c))
    occupied |= _placed_mask(masks[overlap_i], rows, cols, r, c)

    remaining = [i for i in range(1, len(items)) if i != overlap_i]
    remaining.sort(key=lambda i: int(items[i]["area"]), reverse=True)
    for i in remaining:
        choices = _free_positions(masks[i], occupied)
        if not choices:
            return None
        r, c = _choice(choices)
        placements.append((i, r, c))
        occupied |= _placed_mask(masks[i], rows, cols, r, c)

    return placements


def _passes_oracle(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    items = _build_items()
    if not items:
        return None

    expected = _render_output(items)
    grid = _build_canvas(items)
    patches = [_rect_patch(item) for item in items]
    masks = [patch != TRANSPARENT for patch in patches]
    placements = _choose_placements(items, masks, *grid.shape)
    if placements is None:
        return None
    for i, r, c in placements:
        paste(grid, patches[i], r, c, transparent=TRANSPARENT)
    if not _passes_oracle(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
