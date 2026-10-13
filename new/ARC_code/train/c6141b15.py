from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, find_components, get_bg, random_sprite
import random


Coord = tuple[int, int]


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _straight_line_cells(a: Coord, b: Coord) -> set[Coord] | None:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return None
    sr = _sgn(dr)
    sc = _sgn(dc)
    return {(r0 + i * sr, c0 + i * sc) for i in range(max(abs(dr), abs(dc)) + 1)}


def _segment_endpoints(coords: np.ndarray) -> tuple[Coord, Coord] | None:
    cells = {tuple(map(int, rc)) for rc in coords}
    if len(cells) < 2:
        return None

    endpoints: list[Coord] = []
    for r, c in cells:
        degree = sum(
            (r + dr, c + dc) in cells
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )
        if degree == 1:
            endpoints.append((r, c))
        elif degree != 2:
            return None

    if len(endpoints) != 2:
        return None
    line = _straight_line_cells(endpoints[0], endpoints[1])
    if line != cells:
        return None
    return endpoints[0], endpoints[1]


def _normalized_shape(coords: np.ndarray) -> frozenset[Coord]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _center_anchor(coords: np.ndarray) -> Coord | None:
    r0, c0, r1, c1 = bbox_coords(coords)
    if (r0 + r1) % 2 or (c0 + c1) % 2:
        return None
    anchor = ((r0 + r1) // 2, (c0 + c1) // 2)
    if anchor not in {tuple(map(int, rc)) for rc in coords}:
        return None
    return anchor


def _are_repeated_centered_sprites(components: list[np.ndarray]) -> bool:
    if len(components) < 2:
        return False
    shapes = {_normalized_shape(coords) for coords in components}
    return len(shapes) == 1 and all(_center_anchor(coords) is not None for coords in components)


def _non_collinear_three(points: list[Coord]) -> bool:
    if len(points) != 3:
        return True
    (r0, c0), (r1, c1), (r2, c2) = points
    return (r1 - r0) * (c2 - c0) != (r2 - r0) * (c1 - c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap a connector segment with repeated marker sprites by connecting the marker anchors and stamping the sprite at the segment endpoints.

    Input structure
    The grid has one background color and exactly two foreground colors. One foreground color forms a single straight 8-connected segment. The other foreground color forms two or more identical centered sprites.

    Rule
    Build a background canvas of the same size. Draw straight connector-color segments between every pair of original sprite anchors. Then stamp the marker sprite, using its centered anchor, onto the two endpoints of the original connector segment.

    Key insight
    The original segment endpoints are the destinations for the repeated sprite, while the original sprite centers become the vertices of the new connector-color complete graph.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2

    components_by_color = {
        color: [
            component["coords"]
            for component in find_components(g, bg=bg, colors=[color], connectivity=8)
        ]
        for color in colors
    }

    connector_colors = [
        color
        for color, components in components_by_color.items()
        if len(components) == 1 and _segment_endpoints(components[0]) is not None
    ]
    sprite_colors = [
        color
        for color, components in components_by_color.items()
        if _are_repeated_centered_sprites(components)
    ]
    assert len(connector_colors) == 1
    assert len(sprite_colors) == 1

    connector_color = connector_colors[0]
    sprite_color = sprite_colors[0]
    assert connector_color != sprite_color

    endpoint_pair = _segment_endpoints(components_by_color[connector_color][0])
    assert endpoint_pair is not None

    sprite_components = components_by_color[sprite_color]
    anchors: list[Coord] = []
    for coords in sprite_components:
        anchor = _center_anchor(coords)
        assert anchor is not None
        anchors.append(anchor)
    assert _non_collinear_three(anchors), "three repeated sprites must form a non-collinear triangle"

    reference_anchor = _center_anchor(sprite_components[0])
    assert reference_anchor is not None
    sprite_offsets = [
        (int(r - reference_anchor[0]), int(c - reference_anchor[1]))
        for r, c in sprite_components[0]
    ]

    out = np.full_like(g, bg)
    for a, b in combinations(anchors, 2):
        assert _straight_line_cells(a, b) is not None
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=connector_color)

    height, width = out.shape
    for endpoint_r, endpoint_c in endpoint_pair:
        for dr, dc in sprite_offsets:
            r = endpoint_r + dr
            c = endpoint_c + dc
            assert 0 <= r < height and 0 <= c < width
            out[r, c] = sprite_color

    return out

# --------------- generator (att1) ---------------

# Canvas and colors.
GRID_SIZE_MIN = 11
GRID_SIZE_MAX = 16
BACKGROUND_COLOR = 7
FOREGROUND_POOL = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# Symmetric sprite shape.
SPRITE_HEIGHT_CHOICES = [1, 3, 3, 5, 5]
SPRITE_WIDTH_CHOICES = [1, 3, 3, 5]
SPRITE_DIM_SWAP_PROB = 0.5
SPRITE_QUADRANT_FILL_PROB = 0.5
SPRITE_BUILD_RETRIES = 50

# Anchor layout.
THREE_SPRITE_PROB = 0.85
ANCHOR_LAYOUT_RETRIES = 100
ANCHOR_DIRECTIONS = ['h', 'v', 'd1', 'd2']
TWO_ANCHOR_DIST_MIN = 3
TWO_ANCHOR_DIST_MAX = 8
TRIANGLE_DELTA_MIN = 3
TRIANGLE_DELTA_MAX = 7
TRIANGLE_ROTATIONS = [0, 1, 2, 3]

# Connector segment.
SEGMENT_RETRIES = 100
SEGMENT_LENGTH_MIN = 6
SEGMENT_LENGTH_MAX = 10

# Overall retry budget.
GENERATE_RETRIES = 300


def _gen2_symmetric_sprite(color):
    """Generate one connected LR+UD symmetric sprite with a filled center."""
    h = random.choice(SPRITE_HEIGHT_CHOICES)
    w = random.choice(SPRITE_WIDTH_CHOICES)
    if random.random() < SPRITE_DIM_SWAP_PROB:
        h, w = w, h

    if h == 1 and w == 1:
        return np.array([[color]])

    half_h = h // 2 + 1
    half_w = w // 2 + 1
    sprite = np.zeros((h, w), dtype=int)
    cr, cc = h // 2, w // 2
    sprite[cr, cc] = color

    for r in range(half_h):
        for c in range(half_w):
            if random.random() < SPRITE_QUADRANT_FILL_PROB:
                sprite[r, c] = color
                sprite[r, w - 1 - c] = color
                sprite[h - 1 - r, c] = color
                sprite[h - 1 - r, w - 1 - c] = color

    sprite[cr, cc] = color

    from arc_utils import is_contiguous
    mask = (sprite != 0)
    if not mask.any():
        sprite[cr, cc] = color
        mask = (sprite != 0)
    if not is_contiguous(mask, background=0, connectivity=8):
        return None

    if not np.any(sprite[0, :]) or not np.any(sprite[-1, :]):
        return None
    if not np.any(sprite[:, 0]) or not np.any(sprite[:, -1]):
        return None

    return sprite


def _gen_sprite_offsets(sprite):
    h, w = sprite.shape
    cr, cc = h // 2, w // 2
    offsets = []
    for r in range(h):
        for c in range(w):
            if sprite[r, c] != 0:
                offsets.append((r - cr, c - cc))
    return offsets


def _gen_sprite_cells_at(offsets, anchor_r, anchor_c):
    return {(anchor_r + dr, anchor_c + dc) for dr, dc in offsets}


def _fits_in_grid(offsets, anchor_r, anchor_c, h, w):
    for dr, dc in offsets:
        r, c = anchor_r + dr, anchor_c + dc
        if not (0 <= r < h and 0 <= c < w):
            return False
    return True


def _choose_sprite(sprite_color):
    for _ in range(SPRITE_BUILD_RETRIES):
        sprite = _gen2_symmetric_sprite(sprite_color)
        if sprite is not None:
            return sprite
    return None


def _step_from_direction(direction, dist):
    if direction == 'h':
        return 0, dist
    if direction == 'v':
        return dist, 0
    if direction == 'd1':
        return dist, dist
    return dist, -dist


def _sample_two_anchor_layout(size, radius_r, radius_c):
    direction = random.choice(ANCHOR_DIRECTIONS)
    a1_r = random.randint(radius_r + 1, size - 1 - radius_r - 1)
    a1_c = random.randint(radius_c + 1, size - 1 - radius_c - 1)
    dist = random.randint(TWO_ANCHOR_DIST_MIN, TWO_ANCHOR_DIST_MAX)
    dr, dc = _step_from_direction(direction, dist)
    return [(a1_r, a1_c), (a1_r + dr, a1_c + dc)]


def _sample_three_anchor_layout(size, radius_r, radius_c):
    dx = random.randint(TRIANGLE_DELTA_MIN, TRIANGLE_DELTA_MAX)
    dy = dx * random.choice([-1, 1])
    rot = random.choice(TRIANGLE_ROTATIONS)
    a_r = random.randint(radius_r + 1, size - 1 - radius_r - 1)
    a_c = random.randint(radius_c + 1, size - 1 - radius_c - 1)

    if rot == 0:
        return [(a_r, a_c), (a_r, a_c + dx), (a_r + dy, a_c)]
    if rot == 1:
        return [(a_r, a_c), (a_r + dx, a_c), (a_r, a_c + dy)]
    if rot == 2:
        return [(a_r, a_c), (a_r, a_c - dx), (a_r + dy, a_c)]
    return [(a_r, a_c), (a_r - dx, a_c), (a_r, a_c + dy)]


def _anchors_are_valid(anchors, offsets, size):
    if len(set(anchors)) != len(anchors):
        return False
    if not _non_collinear_three(anchors):
        return False
    if not all(_fits_in_grid(offsets, ar, ac, size, size) for ar, ac in anchors):
        return False
    return all(_straight_line_cells(a, b) is not None for a, b in combinations(anchors, 2))


def _choose_anchors(offsets, size, radius_r, radius_c):
    num_sprites = 3 if random.random() < THREE_SPRITE_PROB else 2
    for _ in range(ANCHOR_LAYOUT_RETRIES):
        if num_sprites == 2:
            anchors = _sample_two_anchor_layout(size, radius_r, radius_c)
        else:
            anchors = _sample_three_anchor_layout(size, radius_r, radius_c)
        if _anchors_are_valid(anchors, offsets, size):
            return anchors
    return None


def _line_cells_between_pairs(anchors):
    cells = set()
    for a, b in combinations(anchors, 2):
        cells.update(_straight_line_cells(a, b))
    return cells


def _sprite_cells_for_anchors(offsets, anchors):
    cells = set()
    for ar, ac in anchors:
        cells.update(_gen_sprite_cells_at(offsets, ar, ac))
    return cells


def _has_pairwise_sprite_overlap(offsets, anchors):
    sprite_cells = [_gen_sprite_cells_at(offsets, ar, ac) for ar, ac in anchors]
    for i in range(len(sprite_cells)):
        for j in range(i + 1, len(sprite_cells)):
            if sprite_cells[i] & sprite_cells[j]:
                return True
    return False


def _sample_segment(size, radius_r, radius_c):
    seg_dir = random.choice(ANCHOR_DIRECTIONS)
    seg_len = random.randint(SEGMENT_LENGTH_MIN, SEGMENT_LENGTH_MAX)
    e1_r = random.randint(radius_r, size - 1 - radius_r)
    e1_c = random.randint(radius_c, size - 1 - radius_c)
    dr, dc = _step_from_direction(seg_dir, seg_len - 1)
    e2 = (e1_r + dr, e1_c + dc)
    return (e1_r, e1_c), e2


def _choose_verified_input(size, bg, connector_color, sprite_color, offsets, anchors, radius_r, radius_c):
    input_sprite_cells = _sprite_cells_for_anchors(offsets, anchors)
    output_connector_cells = _line_cells_between_pairs(anchors)

    for _ in range(SEGMENT_RETRIES):
        e1, e2 = _sample_segment(size, radius_r, radius_c)
        if not _fits_in_grid(offsets, e1[0], e1[1], size, size):
            continue
        if not _fits_in_grid(offsets, e2[0], e2[1], size, size):
            continue

        seg_cells = _straight_line_cells(e1, e2)
        if seg_cells is None:
            continue
        if any(not (0 <= r < size and 0 <= c < size) for r, c in seg_cells):
            continue

        endpoint_sprite_cells = _sprite_cells_for_anchors(offsets, [e1, e2])
        if output_connector_cells & endpoint_sprite_cells:
            continue
        if input_sprite_cells & seg_cells:
            continue
        if _has_pairwise_sprite_overlap(offsets, anchors):
            continue
        if _gen_sprite_cells_at(offsets, e1[0], e1[1]) & _gen_sprite_cells_at(offsets, e2[0], e2[1]):
            continue

        inp = _render_input(size, bg, connector_color, sprite_color, offsets, anchors, seg_cells)
        try:
            out = main(inp)
        except Exception:
            continue

        expected_out = _render_expected_output(
            size, bg, connector_color, sprite_color, offsets, [e1, e2], output_connector_cells
        )
        if np.array_equal(out, expected_out):
            return inp
    return None


def _render_input(size, bg, connector_color, sprite_color, offsets, anchors, seg_cells):
    grid = np.full((size, size), bg, dtype=int)
    for ar, ac in anchors:
        for dr, dc in offsets:
            grid[ar + dr, ac + dc] = sprite_color
    for r, c in seg_cells:
        grid[r, c] = connector_color
    return grid


def _render_expected_output(size, bg, connector_color, sprite_color, offsets, endpoints, output_connector_cells):
    grid = np.full((size, size), bg, dtype=int)
    for r, c in output_connector_cells:
        grid[r, c] = connector_color
    for er, ec in endpoints:
        for dr, dc in offsets:
            grid[er + dr, ec + dc] = sprite_color
    return grid


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    bg = BACKGROUND_COLOR
    connector_color, sprite_color = random.sample(FOREGROUND_POOL, 2)

    sprite = _choose_sprite(sprite_color)
    if sprite is None:
        return None

    offsets = _gen_sprite_offsets(sprite)
    radius_r = sprite.shape[0] // 2
    radius_c = sprite.shape[1] // 2

    anchors = _choose_anchors(offsets, size, radius_r, radius_c)
    if anchors is None:
        return None

    return _choose_verified_input(
        size, bg, connector_color, sprite_color, offsets, anchors, radius_r, radius_c
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATE_RETRIES} attempts")
