import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_MIN = 18
GRID_MAX = 24
PAIR_COUNT_CHOICES = (1, 2, 3)
MAX_SHAPE_ATTEMPTS = 200
MAX_PLACEMENT_ATTEMPTS = 200
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _normalize(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    return norm, height, width


def _mirror_horizontal(coords, width: int):
    return tuple(sorted((r, width - 1 - c) for r, c in coords))


def _translate(coords, top: int, left: int):
    return [(top + r, left + c) for r, c in coords]


def _find_components(grid: np.ndarray, skip: np.ndarray):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if skip[r, c] or visited[r, c] or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if skip[nr, nc] or visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))
            norm, box_h, box_w = _normalize(coords)
            components.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "norm": norm,
                    "height": box_h,
                    "width": box_w,
                    "size": len(coords),
                    "anchor": coords[0],
                }
            )
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair identical sprites by shape, move them to their anchor, and color overlap
    with the legend's overlap color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2
    h, w = g.shape

    overlap_color = int(g[0, 1])
    skip = np.zeros((h, w), dtype=bool)
    skip[0, 0] = True
    skip[0, 1] = True

    components = _find_components(g, skip)
    anchors = [comp for comp in components if comp["size"] == 1]
    sprites = [comp for comp in components if 3 <= comp["size"] <= 8]
    assert len(anchors) >= 1 and len(sprites) >= 2
    assert len(anchors) * 2 == len(sprites)
    assert len(anchors) + len(sprites) == len(components)

    sprite_groups = defaultdict(list)
    for comp in sprites:
        signature = (comp["height"], comp["width"], comp["norm"])
        sprite_groups[signature].append(comp)

    out = np.full_like(g, BG)
    out[0, 0] = int(g[0, 0])
    out[0, 1] = overlap_color

    placed_anchors = set()
    for signature, pair in sprite_groups.items():
        assert len(pair) == 2, "each shape must appear exactly twice"
        first, second = pair
        assert first["color"] != second["color"]

        matching_anchors = [a for a in anchors if a["color"] in {first["color"], second["color"]}]
        assert len(matching_anchors) == 1, "pair must have exactly one matching anchor"
        anchor_comp = matching_anchors[0]
        placed_anchors.add(anchor_comp["anchor"])
        anchor_r, anchor_c = anchor_comp["anchor"]

        if anchor_comp["color"] == first["color"]:
            fixed, flipped = first, second
        else:
            fixed, flipped = second, first

        box_h, box_w, _ = signature
        fixed_cells = set(_translate(fixed["norm"], anchor_r, anchor_c))
        flipped_cells = set(_translate(_mirror_horizontal(flipped["norm"], box_w), anchor_r, anchor_c))

        for rr, cc in fixed_cells | flipped_cells:
            assert 0 <= rr < h and 0 <= cc < w
            assert not (rr == 0 and cc in (0, 1))
            current = int(out[rr, cc])
            assert current == BG, "placed composites must not overlap each other"
            if (rr, cc) in fixed_cells and (rr, cc) in flipped_cells:
                out[rr, cc] = overlap_color
            elif (rr, cc) in fixed_cells:
                out[rr, cc] = int(fixed["color"])
            else:
                out[rr, cc] = int(flipped["color"])

    assert len(placed_anchors) == len(anchors)
    return out


def _random_connected_shape(rng: random.Random):
    for _ in range(MAX_SHAPE_ATTEMPTS):
        box_h = rng.randint(2, 4)
        box_w = rng.randint(2, 4)
        max_cells = min(8, box_h * box_w)
        target = rng.randint(3, max_cells)
        start = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue

        norm, height, width = _normalize(cells)
        if len(norm) < 3 or width < 2:
            continue
        if len(norm) == height * width:
            continue

        mirrored = _mirror_horizontal(norm, width)
        overlap = len(set(norm) & set(mirrored))
        if 0 < overlap < len(norm):
            return {
                "norm": norm,
                "height": height,
                "width": width,
                "mirrored": mirrored,
                "union": tuple(sorted(set(norm) | set(mirrored))),
                "overlap": overlap,
            }
    return None


def _can_place(coords, blocked: np.ndarray, pad: int = 1):
    h, w = blocked.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if np.any(blocked[r0:r1, c0:c1]):
            return False
    return True


def _mark(coords, blocked: np.ndarray, pad: int = 1):
    h, w = blocked.shape
    for r, c in coords:
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        blocked[r0:r1, c0:c1] = True


def _place_shape(rng: random.Random, blocked: np.ndarray, shape_coords, box_h: int, box_w: int):
    h, w = blocked.shape
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        top = rng.randint(0, h - box_h)
        left = rng.randint(0, w - box_w)
        cells = _translate(shape_coords, top, left)
        if _can_place(cells, blocked, pad=1):
            _mark(cells, blocked, pad=1)
            return top, left, cells
    return None


def _gen_attempt(rng: random.Random):
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    pair_count = rng.choice(PAIR_COUNT_CHOICES)

    overlap_color = rng.choice(COLOR_POOL)
    sprite_palette = [c for c in COLOR_POOL if c != overlap_color]
    if len(sprite_palette) < 2 * pair_count:
        return None
    sprite_colors = rng.sample(sprite_palette, 2 * pair_count)

    specs = []
    used_signatures = set()
    for idx in range(pair_count):
        shape = None
        for _ in range(MAX_SHAPE_ATTEMPTS):
            candidate = _random_connected_shape(rng)
            if candidate is None:
                continue
            signature = (candidate["height"], candidate["width"], candidate["norm"])
            if signature not in used_signatures:
                used_signatures.add(signature)
                shape = candidate
                break
        if shape is None:
            return None
        colors = (sprite_colors[2 * idx], sprite_colors[2 * idx + 1])
        specs.append(
            {
                "shape": shape,
                "colors": colors,
                "anchor_color": rng.choice(colors),
            }
        )

    order = sorted(range(pair_count), key=lambda i: len(specs[i]["shape"]["union"]), reverse=True)
    blocked = np.zeros((h, w), dtype=bool)
    _mark([(0, 0), (0, 1)], blocked, pad=1)

    for idx in order:
        shape = specs[idx]["shape"]
        box_h = shape["height"]
        box_w = shape["width"]
        placed = None
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            top = rng.randint(0, h - box_h)
            left = rng.randint(0, w - box_w)
            reserved = set(_translate(shape["union"], top, left))
            reserved.add((top, left))
            if _can_place(reserved, blocked, pad=1):
                _mark(reserved, blocked, pad=1)
                placed = (top, left)
                break
        if placed is None:
            return None
        specs[idx]["anchor"] = placed

    for idx in order:
        shape = specs[idx]["shape"]
        placements = []
        for _color in specs[idx]["colors"]:
            placed = _place_shape(rng, blocked, shape["norm"], shape["height"], shape["width"])
            if placed is None:
                return None
            placements.append(placed)
        specs[idx]["sources"] = placements

    grid = np.full((h, w), BG, dtype=int)
    grid[0, 0] = GREY
    grid[0, 1] = overlap_color

    for spec in specs:
        anchor_r, anchor_c = spec["anchor"]
        grid[anchor_r, anchor_c] = spec["anchor_color"]
        for color, placed in zip(spec["colors"], spec["sources"]):
            _top, _left, cells = placed
            for r, c in cells:
                grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.count_nonzero(out == overlap_color) <= 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
