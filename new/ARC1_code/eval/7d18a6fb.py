import numpy as np

from arc_utils import bbox_coords, bbox_h, bbox_w, components_mask, find_components, get_bg, paste
import random


def _find_selector(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    selectors = [
        comp
        for comp in comps
        if comp["area"] == 45 and bbox_h(comp["bbox"]) == 7 and bbox_w(comp["bbox"]) == 7
    ]
    assert len(selectors) == 1
    selector = selectors[0]
    return int(selector["color"]), selector["bbox"]


def _external_sprite(g: np.ndarray, color: int, selector_bbox: tuple[int, int, int, int], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = selector_bbox
    outside_selector = np.ones(g.shape, dtype=bool)
    outside_selector[r0 : r1 + 1, c0 : c1 + 1] = False

    comps = components_mask((g == int(color)) & outside_selector, connectivity=8)
    assert len(comps) == 1
    coords = comps[0]
    sprite_bbox = bbox_coords(coords)
    assert bbox_h(sprite_bbox) == 3 and bbox_w(sprite_bbox) == 3

    sr0, sc0, _sr1, _sc1 = sprite_bbox
    sprite = np.full((3, 3), bg, dtype=int)
    for r, c in coords:
        sprite[int(r) - sr0, int(c) - sc0] = int(color)
    return sprite


def main(grid: np.ndarray) -> np.ndarray:
    """Place the selected colored sprites into a 7x7 summary using the marker colors in the selector block.

    Input structure
    A dominant background surrounds one 7x7 selector block: its main color fills all but four marker pixels at the
    centers of the four 3x3 quadrants. Elsewhere, each marker color has a unique same-color 8-connected sprite with a
    3x3 bounding box.

    Rule
    Build a 7x7 background canvas split into four 3x3 quadrants by a blank middle row and column. For each marker,
    copy the unique external sprite of the same color into the corresponding quadrant.

    Key insight
    The selector block is a positional legend: marker position chooses the output quadrant, and marker color chooses
    which outside sprite is copied there.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    selector_color, selector_bbox = _find_selector(g, bg)
    r0, c0, r1, c1 = selector_bbox

    selector = g[r0 : r1 + 1, c0 : c1 + 1]
    assert selector.shape == (7, 7)
    assert np.all(selector != bg)

    markers = np.argwhere((selector != selector_color) & (selector != bg))
    assert markers.shape == (4, 2)

    out = np.full((7, 7), bg, dtype=int)
    quadrant_anchor = {1: 0, 5: 4}
    for rr, cc in markers:
        rr = int(rr)
        cc = int(cc)
        assert rr in quadrant_anchor and cc in quadrant_anchor
        color = int(selector[rr, cc])
        sprite = _external_sprite(g, color, selector_bbox, bg)
        paste(out, sprite, quadrant_anchor[rr], quadrant_anchor[cc], transparent=bg)

    return out

# --------------- generator (o5x) ---------------

# Fixed colors and geometry.
GEN_BG = 0
GEN_SELECTOR_COLOR = 1
GEN_SELECTOR_SIZE = 7
GEN_SPRITE_SIZE = 3
GEN_MARKER_POSITIONS = [(1, 1), (1, 5), (5, 1), (5, 5)]
GEN_QUADRANT_ANCHORS = [(0, 0), (0, 4), (4, 0), (4, 4)]

# Sampling ranges and palette.
GEN_COLOR_POOL = list(range(2, 10))
GEN_MARKER_COUNT = 4
GEN_DISTRACTOR_COUNT_RANGE = (0, 2)
GEN_SPRITE_AREA_RANGE = (4, 7)

# Canvas placement and retry limits.
GEN_ROW_RANGE = (16, 18)
GEN_COL_RANGE = (13, 17)
GEN_PLACEMENT_GAP = 1
GEN_MAX_ATTEMPTS = 300


def _gen_valid_sprite_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate all 8-connected 3x3 masks with a full 3x3 bbox."""
    min_area, max_area = GEN_SPRITE_AREA_RANGE
    masks_by_area: dict[int, list[np.ndarray]] = {area: [] for area in range(min_area, max_area + 1)}
    cell_count = GEN_SPRITE_SIZE * GEN_SPRITE_SIZE
    for bits in range(1, 1 << cell_count):
        mask = np.array([(bits >> k) & 1 for k in range(cell_count)], dtype=bool).reshape(
            GEN_SPRITE_SIZE, GEN_SPRITE_SIZE
        )
        area = int(mask.sum())
        if area not in masks_by_area:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if len(components_mask(mask, connectivity=8)) != 1:
            continue
        masks_by_area[area].append(mask)

    assert all(masks_by_area.values())
    return masks_by_area


_GEN_SPRITE_MASKS = _gen_valid_sprite_masks()


def _gen_sprite(color: int) -> np.ndarray:
    area = random.randint(*GEN_SPRITE_AREA_RANGE)
    mask = random.choice(_GEN_SPRITE_MASKS[area])
    sprite = np.zeros((GEN_SPRITE_SIZE, GEN_SPRITE_SIZE), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _gen_bbox_at(r: int, c: int) -> tuple[int, int, int, int]:
    return int(r), int(c), int(r) + GEN_SPRITE_SIZE - 1, int(c) + GEN_SPRITE_SIZE - 1


def _gen_bboxes_have_gap(a, b, gap=GEN_PLACEMENT_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    g = int(gap)
    return ar1 + g < br0 or br1 + g < ar0 or ac1 + g < bc0 or bc1 + g < ac0


def _gen_place_regions(rows, cols, selector_bbox, count) -> list[tuple[int, int]] | None:
    """Backtrack shuffled anchors until all requested sprite gaps hold."""
    candidates = []
    for r in range(int(rows) - GEN_SPRITE_SIZE + 1):
        for c in range(int(cols) - GEN_SPRITE_SIZE + 1):
            region_bbox = _gen_bbox_at(r, c)
            if _gen_bboxes_have_gap(region_bbox, selector_bbox):
                candidates.append((int(r), int(c)))
    random.shuffle(candidates)

    def search(chosen, chosen_bboxes, pool):
        if len(chosen) == int(count):
            return chosen
        if len(pool) < int(count) - len(chosen):
            return None

        for i, (r, c) in enumerate(pool):
            bbox = _gen_bbox_at(r, c)
            if not all(_gen_bboxes_have_gap(bbox, prev) for prev in chosen_bboxes):
                continue
            next_pool = [
                loc
                for loc in pool[i + 1 :]
                if _gen_bboxes_have_gap(_gen_bbox_at(loc[0], loc[1]), bbox)
            ]
            found = search(chosen + [(r, c)], chosen_bboxes + [bbox], next_pool)
            if found is not None:
                return found
        return None

    return search([], [], candidates)


def _gen_selector_bbox(rows: int, cols: int) -> tuple[int, int, int, int]:
    max_r = int(rows) - GEN_SELECTOR_SIZE
    max_c = int(cols) - GEN_SELECTOR_SIZE
    corner = random.choice(((0, 0), (0, max_c), (max_r, 0), (max_r, max_c)))
    r0, c0 = corner
    return int(r0), int(c0), int(r0) + GEN_SELECTOR_SIZE - 1, int(c0) + GEN_SELECTOR_SIZE - 1


def _gen_palette() -> tuple[list[int], list[int]]:
    distractor_count = random.randint(*GEN_DISTRACTOR_COUNT_RANGE)
    colors = random.sample(GEN_COLOR_POOL, GEN_MARKER_COUNT + distractor_count)
    return colors[:GEN_MARKER_COUNT], colors[GEN_MARKER_COUNT:]


def _gen_expected_and_selector(marker_colors, marker_sprites) -> tuple[np.ndarray, np.ndarray]:
    """Render paired 7x7 oracle output and selector legend for the same marker order."""
    expected = np.full((GEN_SELECTOR_SIZE, GEN_SELECTOR_SIZE), GEN_BG, dtype=int)
    for sprite, (r, c) in zip(marker_sprites, GEN_QUADRANT_ANCHORS):
        paste(expected, sprite, r, c, transparent=GEN_BG)

    selector = np.full((GEN_SELECTOR_SIZE, GEN_SELECTOR_SIZE), GEN_SELECTOR_COLOR, dtype=int)
    for (r, c), color in zip(GEN_MARKER_POSITIONS, marker_colors):
        selector[r, c] = int(color)
    return expected, selector


def _gen_canvas_layout() -> tuple[int, int, tuple[int, int, int, int]]:
    rows = random.randint(*GEN_ROW_RANGE)
    cols = random.randint(*GEN_COL_RANGE)
    return rows, cols, _gen_selector_bbox(rows, cols)


def _gen_render_grid(rows, cols, selector, selector_bbox, marker_sprites, distractor_sprites) -> np.ndarray | None:
    """Place selector and shuffled sprites, returning None if gap search fails."""
    sr0, sc0, _sr1, _sc1 = selector_bbox

    sprites_for_input = marker_sprites + distractor_sprites
    random.shuffle(sprites_for_input)
    regions = _gen_place_regions(rows, cols, selector_bbox, len(sprites_for_input))
    if regions is None:
        return None

    grid = np.full((rows, cols), GEN_BG, dtype=int)
    paste(grid, selector, sr0, sc0)
    for sprite, (r, c) in zip(sprites_for_input, regions):
        paste(grid, sprite, r, c, transparent=GEN_BG)
    return grid


def _gen_solver_accepts(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    marker_colors, distractor_colors = _gen_palette()
    marker_sprites = [_gen_sprite(color) for color in marker_colors]
    distractor_sprites = [_gen_sprite(color) for color in distractor_colors]
    expected, selector = _gen_expected_and_selector(marker_colors, marker_sprites)
    rows, cols, selector_bbox = _gen_canvas_layout()
    grid = _gen_render_grid(rows, cols, selector, selector_bbox, marker_sprites, distractor_sprites)
    return grid if grid is not None and _gen_solver_accepts(grid, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 7d18a6fb input")
