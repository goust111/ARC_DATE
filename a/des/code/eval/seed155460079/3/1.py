import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

ORIENTATION_TO_COLOR = {
    0: BLUE,
    1: GREEN,
    2: YELLOW,
    3: ORANGE,
}

GEN_RETRIES = 400
PLACEMENT_ATTEMPTS = 400
REGION_COUNT_RANGE = (4, 7)
REGION_SIZE_RANGE = (7, 9)
GRID_H_BASE_RANGE = (24, 34)
GRID_W_BASE_RANGE = (30, 40)

TEMPLATE_FRAME_SIZE = 5
TEMPLATE_GAP = 1
TEMPLATE_PADDING = 1

BASE_L = ((0, 0), (1, 0), (2, 0), (2, 1))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _normalize(coords):
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _rotate(coords, quarter_turns):
    out = list(coords)
    for _ in range(quarter_turns % 4):
        out = [(c, -r) for r, c in out]
    return _normalize(out)


L_SHAPES = tuple(_rotate(BASE_L, k) for k in range(4))
L_SHAPE_TO_ORIENTATION = {shape: idx for idx, shape in enumerate(L_SHAPES)}


def _shape_size(shape):
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _color_components(grid, color):
    mask = np.asarray(grid == color)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _frame_from_component(coords):
    if len(coords) < 8:
        return None
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(coords) != 2 * (h + w) - 4:
        return None
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    if set(coords) != border:
        return None
    return (r0, c0, r1, c1)


def _find_frames(grid):
    frames = []
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        for comp in _color_components(grid, color):
            bbox = _frame_from_component(comp)
            if bbox is not None:
                frames.append({"color": color, "bbox": bbox})
    return frames


def _extract_l_orientation(patch):
    if patch.size == 0:
        return None
    if np.any((patch != BG) & (patch != RED)):
        return None
    coords = [tuple(map(int, cell)) for cell in np.argwhere(patch == RED)]
    if len(coords) != 4:
        return None
    coord_set = set(coords)
    stack = [coords[0]]
    seen = {coords[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in coord_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    if len(seen) != 4:
        return None
    return L_SHAPE_TO_ORIENTATION.get(_normalize(coord_set))


def _frame_interior(grid, bbox):
    r0, c0, r1, c1 = bbox
    return grid[r0 + 1 : r1, c0 + 1 : c1]


def _extract_template_mapping(grid, frames):
    mapping = {}
    for frame in frames:
        if frame["color"] == GREY:
            continue
        orientation = _extract_l_orientation(_frame_interior(grid, frame["bbox"]))
        if orientation is None:
            continue
        assert orientation not in mapping, "duplicate template orientation"
        mapping[orientation] = frame["color"]
    assert len(mapping) == 4, "expected four colored template frames"
    return mapping


def _target_frames(grid, frames):
    targets = []
    for frame in frames:
        if frame["color"] != GREY:
            continue
        interior = _frame_interior(grid, frame["bbox"])
        if np.any((interior != BG) & (interior != RED)):
            continue
        orientation = _extract_l_orientation(interior)
        if orientation is None:
            continue
        targets.append((frame["bbox"], orientation))
    return targets


def _draw_frame(grid, bbox, color):
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _template_wrapper_size():
    h = TEMPLATE_FRAME_SIZE + 2 * TEMPLATE_PADDING + 2
    w = 4 * TEMPLATE_FRAME_SIZE + 3 * TEMPLATE_GAP + 2 * TEMPLATE_PADDING + 2
    return h, w


def _stamp_l(grid, bbox, orientation, rng, require_margin):
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    shape = L_SHAPES[orientation]
    shape_h, shape_w = _shape_size(shape)

    slack_h = inner_h - shape_h
    slack_w = inner_w - shape_w
    assert slack_h >= 0 and slack_w >= 0

    min_r = 1 if require_margin and slack_h >= 2 else 0
    max_r = slack_h - 1 if require_margin and slack_h >= 2 else slack_h
    min_c = 1 if require_margin and slack_w >= 2 else 0
    max_c = slack_w - 1 if require_margin and slack_w >= 2 else slack_w

    off_r = rng.randint(min_r, max_r)
    off_c = rng.randint(min_c, max_c)
    for dr, dc in shape:
        grid[r0 + 1 + off_r + dr, c0 + 1 + off_c + dc] = RED


def _mark_occupied(occupied, bbox, moat):
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - moat)
    cc0 = max(0, c0 - moat)
    rr1 = min(occupied.shape[0] - 1, r1 + moat)
    cc1 = min(occupied.shape[1] - 1, c1 + moat)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _bbox_is_free(occupied, bbox, moat):
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - moat)
    cc0 = max(0, c0 - moat)
    rr1 = min(occupied.shape[0] - 1, r1 + moat)
    cc1 = min(occupied.shape[1] - 1, c1 + moat)
    return not np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _place_template(grid, rng, occupied):
    h, w = grid.shape
    wrapper_h, wrapper_w = _template_wrapper_size()
    corners = [
        (0, 0),
        (0, w - wrapper_w),
        (h - wrapper_h, 0),
        (h - wrapper_h, w - wrapper_w),
    ]
    wrapper_r0, wrapper_c0 = rng.choice(corners)
    wrapper_bbox = (
        wrapper_r0,
        wrapper_c0,
        wrapper_r0 + wrapper_h - 1,
        wrapper_c0 + wrapper_w - 1,
    )
    _draw_frame(grid, wrapper_bbox, GREY)

    order = list(range(4))
    rng.shuffle(order)
    frame_r0 = wrapper_r0 + 1 + TEMPLATE_PADDING
    frame_c0 = wrapper_c0 + 1 + TEMPLATE_PADDING
    for idx, orientation in enumerate(order):
        left = frame_c0 + idx * (TEMPLATE_FRAME_SIZE + TEMPLATE_GAP)
        bbox = (
            frame_r0,
            left,
            frame_r0 + TEMPLATE_FRAME_SIZE - 1,
            left + TEMPLATE_FRAME_SIZE - 1,
        )
        _draw_frame(grid, bbox, ORIENTATION_TO_COLOR[orientation])
        _stamp_l(grid, bbox, orientation, rng, require_margin=False)

    _mark_occupied(occupied, wrapper_bbox, moat=1)
    return wrapper_bbox


def _place_regions(grid, rng, occupied, count):
    h, w = grid.shape
    orientations = list(range(4)) + [rng.randrange(4) for _ in range(count - 4)]
    rng.shuffle(orientations)
    placed = []

    for orientation in orientations:
        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            region_h = rng.randint(*REGION_SIZE_RANGE)
            region_w = rng.randint(*REGION_SIZE_RANGE)
            if region_h >= h - 2 or region_w >= w - 2:
                continue
            r0 = rng.randint(1, h - region_h - 1)
            c0 = rng.randint(1, w - region_w - 1)
            bbox = (r0, c0, r0 + region_h - 1, c0 + region_w - 1)
            if not _bbox_is_free(occupied, bbox, moat=1):
                continue
            _draw_frame(grid, bbox, GREY)
            _stamp_l(grid, bbox, orientation, rng, require_margin=True)
            _mark_occupied(occupied, bbox, moat=1)
            placed.append((bbox, orientation))
            success = True
            break
        if not success:
            return None
    return placed


def _expected_output(grid, regions):
    out = grid.copy()
    for bbox, orientation in regions:
        _draw_frame(out, bbox, ORIENTATION_TO_COLOR[orientation])
        interior = _frame_interior(out, bbox)
        interior[interior == RED] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the template L orientations to recover orientation->color, then recolor each grey
    rectangular frame by the matching template color and erase its red seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    frames = _find_frames(g)
    mapping = _extract_template_mapping(g, frames)

    out = g.copy()
    for bbox, orientation in _target_frames(g, frames):
        _draw_frame(out, bbox, mapping[orientation])
        interior = _frame_interior(out, bbox)
        interior[interior == RED] = BG
    return out


def _gen_attempt(rng):
    n_regions = rng.randint(*REGION_COUNT_RANGE)
    extra = n_regions - REGION_COUNT_RANGE[0]
    min_h = max(GRID_H_BASE_RANGE[0], 24 + 2 * extra)
    min_w = max(GRID_W_BASE_RANGE[0], 30 + 2 * extra)
    h = rng.randint(min_h, GRID_H_BASE_RANGE[1])
    w = rng.randint(min_w, GRID_W_BASE_RANGE[1])

    wrapper_h, wrapper_w = _template_wrapper_size()
    if h < wrapper_h + 8 or w < wrapper_w + 3:
        return None

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    _place_template(grid, rng, occupied)
    regions = _place_regions(grid, rng, occupied, n_regions)
    if regions is None:
        return None

    expected = _expected_output(grid, regions)

    frames = _find_frames(grid)
    mapping = _extract_template_mapping(grid, frames)
    targets = _target_frames(grid, frames)
    if len(mapping) != 4 or len(targets) != len(regions):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
