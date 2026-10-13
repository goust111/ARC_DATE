import itertools
import random
from collections import deque

import numpy as np


BG = 0
GUIDE = 6
SPRITE_SIZE = 3
RING_COORDS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
FRAME_SIDES = {
    "top": 1,
    "left": 2,
    "right": 3,
    "bottom": 4,
}
FRAME_COLORS = set(FRAME_SIDES.values())
GEN_ATTEMPTS = 400


def _bbox_from_coords(coords):
    coords = list(coords)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _solid_color(values):
    uniq = {int(v) for v in values}
    assert len(uniq) == 1
    return next(iter(uniq))


def _bbox_contains(outer, inner):
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 <= ir0 and oc0 <= ic0 and ir1 <= or1 and ic1 <= oc1


def _expanded_overlap(box_a, box_b, pad_each):
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    ar0 -= pad_each
    ac0 -= pad_each
    ar1 += pad_each
    ac1 += pad_each
    br0 -= pad_each
    bc0 -= pad_each
    br1 += pad_each
    bc1 += pad_each
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _mask_components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _is_hollow_sprite(patch, frame_colors):
    if patch.shape != (3, 3) or int(patch[1, 1]) != BG:
        return False
    return all(int(patch[r, c]) in frame_colors for r, c in RING_COORDS)


def _find_sprites(grid):
    h, w = grid.shape
    top = _solid_color(grid[0, 1:-1])
    bottom = _solid_color(grid[-1, 1:-1])
    left = _solid_color(grid[1:-1, 0])
    right = _solid_color(grid[1:-1, -1])
    frame_colors = {top, bottom, left, right}

    sprites = []
    for r0 in range(1, h - 3):
        for c0 in range(1, w - 3):
            patch = grid[r0 : r0 + 3, c0 : c0 + 3]
            if _is_hollow_sprite(patch, frame_colors):
                coords = [(r0 + dr, c0 + dc) for dr, dc in RING_COORDS]
                sprites.append({"bbox": (r0, c0, r0 + 2, c0 + 2), "coords": coords})
    return sprites


def _move_target_sprite(grid, sprite):
    top = _solid_color(grid[0, 1:-1])
    bottom = _solid_color(grid[-1, 1:-1])
    left = _solid_color(grid[1:-1, 0])
    right = _solid_color(grid[1:-1, -1])
    directions = {
        top: (-1, 0),
        bottom: (1, 0),
        left: (0, -1),
        right: (0, 1),
    }

    claims = {}
    for idx, (r, c) in enumerate(sorted(sprite["coords"])):
        color = int(grid[r, c])
        dr, dc = directions[color]
        nr, nc = r + dr, c + dc
        claims.setdefault((nr, nc), []).append((1 if dr != 0 else 0, -idx, color))

    moved = {}
    for cell, options in claims.items():
        moved[cell] = max(options)[2]
    box = _bbox_from_coords(moved.keys())
    out = np.full((box[2] - box[0] + 1, box[3] - box[1] + 1), BG, dtype=int)
    for (r, c), color in moved.items():
        out[r - box[0], c - box[1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Select the hollow 3x3 sprite framed by the guide mask, then move its pixels by frame-matched directions.

    Input structure:
    - The grid has a one-cell border whose corners are black and whose four sides are solid, distinct colors.
    - Inside the border sit 3-4 hollow 3x3 square sprites with colored outer rings and black centers.
    - Sparse magenta guide pixels are separate from the sprites, and their bounding box encloses exactly one sprite.

    Rule:
    - Find every 3x3 hollow-square sprite inside the border.
    - Take all magenta guide pixels, compute their bounding box, and choose the unique sprite fully contained in that box.
    - Move each pixel of the selected sprite one cell toward the border side with the same color: top-color up, bottom-color down, left-color left, right-color right.
    - Resolve simultaneous landing conflicts by keeping a vertically moving pixel over a horizontally moving one, then return the tight crop of the moved result.

    Key insight:
    - The guide mask decides which sprite to operate on, while the frame colors independently decode the per-pixel motion directions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 7 and g.shape[1] >= 7
    assert int(g[0, 0]) == int(g[0, -1]) == int(g[-1, 0]) == int(g[-1, -1]) == BG

    sprites = _find_sprites(g)
    assert sprites

    guide_mask = g == GUIDE
    assert np.any(guide_mask)
    guide_box = _bbox_from_coords(map(tuple, np.argwhere(guide_mask)))

    selected = [sprite for sprite in sprites if _bbox_contains(guide_box, sprite["bbox"])]
    assert len(selected) == 1
    return _move_target_sprite(g, selected[0])


def _random_ring_pattern(allowed_subset):
    colors = list(allowed_subset)
    ring = [random.choice(colors) for _ in range(8)]
    for color in colors:
        if color not in ring:
            ring[random.randrange(8)] = color
    return ring


def _ring_destinations(ring):
    directions = {
        FRAME_SIDES["top"]: (-1, 0),
        FRAME_SIDES["bottom"]: (1, 0),
        FRAME_SIDES["left"]: (0, -1),
        FRAME_SIDES["right"]: (0, 1),
    }
    claims = {}
    for idx, ((r, c), color) in enumerate(zip(RING_COORDS, ring)):
        dr, dc = directions[color]
        claims.setdefault((r + dr, c + dc), []).append((idx, dr != 0))
    return claims


def _sample_selected_ring():
    want_collision = random.random() < 0.5
    subset_size = random.choices([2, 3, 4], weights=[1, 3, 4])[0]
    for _ in range(300):
        subset = frozenset(random.sample(sorted(FRAME_COLORS), subset_size))
        ring = _random_ring_pattern(subset)
        claims = _ring_destinations(ring)
        mixed_collision = False
        same_axis_collision = False
        for options in claims.values():
            has_vertical = any(is_vertical for _, is_vertical in options)
            has_horizontal = any(not is_vertical for _, is_vertical in options)
            if has_vertical and has_horizontal:
                mixed_collision = True
            vertical_count = sum(1 for _, is_vertical in options if is_vertical)
            horizontal_count = len(options) - vertical_count
            if vertical_count > 1 or horizontal_count > 1:
                same_axis_collision = True
                break
        if same_axis_collision:
            continue
        if want_collision != mixed_collision:
            continue
        return ring, subset
    raise RuntimeError("failed to sample selected ring")


def _sample_other_ring(subset):
    for _ in range(200):
        ring = _random_ring_pattern(subset)
        claims = _ring_destinations(ring)
        if all(len(options) == 1 for options in claims.values()):
            return ring
    return _random_ring_pattern(subset)


def _draw_frame(grid):
    grid[0, 1:-1] = FRAME_SIDES["top"]
    grid[-1, 1:-1] = FRAME_SIDES["bottom"]
    grid[1:-1, 0] = FRAME_SIDES["left"]
    grid[1:-1, -1] = FRAME_SIDES["right"]


def _paint_sprite(grid, top_left, ring):
    r0, c0 = top_left
    for (dr, dc), color in zip(RING_COORDS, ring):
        grid[r0 + dr, c0 + dc] = color
    grid[r0 + 1, c0 + 1] = BG


def _guide_perimeter_cells(box):
    r0, c0, r1, c1 = box
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return sorted(set(cells))


def _cells_by_side(box):
    r0, c0, r1, c1 = box
    return {
        "top": [(r0, c) for c in range(c0, c1 + 1)],
        "bottom": [(r1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }


def _make_guide_mask(shape, box):
    side_cells = _cells_by_side(box)
    perimeter = _guide_perimeter_cells(box)
    for _ in range(200):
        target = random.randint(max(6, int(len(perimeter) * 0.4)), max(7, int(len(perimeter) * 0.65)))
        chosen = {random.choice(side_cells["top"]), random.choice(side_cells["bottom"])}
        chosen.add(random.choice(side_cells["left"]))
        chosen.add(random.choice(side_cells["right"]))
        remainder = [cell for cell in perimeter if cell not in chosen]
        random.shuffle(remainder)
        for cell in remainder[: max(0, target - len(chosen))]:
            chosen.add(cell)

        mask = np.zeros(shape, dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        comp_count = len(_mask_components(mask))
        if 2 <= comp_count <= 6:
            return chosen
    return set(perimeter[::2])


def _candidate_positions(h, w, min_margin):
    return [
        (r, c)
        for r in range(min_margin, h - min_margin - 2)
        for c in range(min_margin, w - min_margin - 2)
    ]


def _guide_box_for_selected(top_left, h, w):
    sr, sc = top_left
    max_up = min(2, sr - 1)
    max_left = min(2, sc - 1)
    max_down = min(2, h - (sr + 3) - 1)
    max_right = min(2, w - (sc + 3) - 1)
    if min(max_up, max_left, max_down, max_right) < 1:
        return None
    margins = [
        random.randint(1, max_up),
        random.randint(1, max_left),
        random.randint(1, max_down),
        random.randint(1, max_right),
    ]
    if all(m == 1 for m in margins) and random.random() < 0.7:
        margins[random.randrange(4)] = 2
    return (sr - margins[0], sc - margins[1], sr + 2 + margins[2], sc + 2 + margins[3])


def _generate_attempt():
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    grid = np.full((h, w), BG, dtype=int)
    _draw_frame(grid)

    sprite_count = random.randint(3, 4)
    selected_pos = random.choice(_candidate_positions(h, w, min_margin=3))
    guide_box = _guide_box_for_selected(selected_pos, h, w)
    if guide_box is None:
        return None

    selected_ring, selected_subset = _sample_selected_ring()
    sprites = [{"pos": selected_pos, "ring": selected_ring, "subset": selected_subset}]

    subset_choices = [frozenset(s) for i in range(1, 5) for s in itertools.combinations(sorted(FRAME_COLORS), i)]
    random.shuffle(subset_choices)
    subset_choices = [s for s in subset_choices if s != selected_subset]

    placed_boxes = [(*selected_pos, selected_pos[0] + 2, selected_pos[1] + 2)]
    available = _candidate_positions(h, w, min_margin=2)
    random.shuffle(available)
    for pos in available:
        if len(sprites) == sprite_count:
            break
        box = (pos[0], pos[1], pos[0] + 2, pos[1] + 2)
        if _bbox_contains(guide_box, box) or _expanded_overlap(box, guide_box, 0):
            continue
        if any(_expanded_overlap(box, other, 1) for other in placed_boxes):
            continue
        if not subset_choices:
            break
        subset = subset_choices.pop()
        sprites.append({"pos": pos, "ring": _sample_other_ring(subset), "subset": subset})
        placed_boxes.append(box)

    if len(sprites) != sprite_count:
        return None

    for sprite in sprites:
        _paint_sprite(grid, sprite["pos"], sprite["ring"])

    guide_pixels = _make_guide_mask(grid.shape, guide_box)
    for r, c in guide_pixels:
        if grid[r, c] != BG:
            return None
        grid[r, c] = GUIDE

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape[0] > 5 or out.shape[1] > 5:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
