import random
from collections import Counter, deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_ATTEMPTS = 400

LAYOUT_OPTIONS = (
    (1, 1),
    (1, 1, 1),
    (2,),
    (2, 1),
    (3,),
    (2, 2),
)

GROUP_MARGIN = 2
NEST_MARGIN_MIN = 2
NEST_MARGIN_MAX = 4


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _legend_order(grid: np.ndarray):
    order = []
    seen = set()
    for value in grid[-1]:
        color = int(value)
        if color != BG and color not in seen:
            seen.add(color)
            order.append(color)
    return order


def _interior_for_loop(coords, shape):
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    local_h = er1 - er0 + 1
    local_w = ec1 - ec0 + 1

    walls = np.zeros((local_h, local_w), dtype=bool)
    for r, c in coords:
        walls[r - er0, c - ec0] = True

    outside = np.zeros_like(walls, dtype=bool)
    q = deque()

    def push(rr: int, cc: int):
        if not walls[rr, cc] and not outside[rr, cc]:
            outside[rr, cc] = True
            q.append((rr, cc))

    for rr in range(local_h):
        push(rr, 0)
        push(rr, local_w - 1)
    for cc in range(local_w):
        push(0, cc)
        push(local_h - 1, cc)

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < local_h and 0 <= nc < local_w:
                if not walls[nr, nc] and not outside[nr, nc]:
                    outside[nr, nc] = True
                    q.append((nr, nc))

    interior = np.argwhere((~walls) & (~outside))
    return [(int(rr + er0), int(cc + ec0)) for rr, cc in interior]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed polygon with the boundary color that occupies the most pixels.
    Apply fills in the bottom-legend order so inner polygons can overwrite outer ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    if g.shape[0] < 2:
        return g.copy()

    legend = _legend_order(g)
    priority = {color: idx for idx, color in enumerate(legend)}

    work = g.copy()
    work[-1, :] = BG
    loops = []
    for comp in _components(work != BG):
        interior = _interior_for_loop(comp, work.shape)
        if not interior:
            continue

        counts = Counter(int(g[r, c]) for r, c in comp)
        best_count = max(counts.values())
        tied = [color for color, count in counts.items() if count == best_count]
        fill_color = min(tied, key=lambda color: (priority.get(color, len(priority)), color))
        loops.append(
            {
                "fill": fill_color,
                "priority": priority.get(fill_color, len(priority)),
                "bbox": _bbox(comp),
                "interior": interior,
            }
        )

    out = g.copy()
    original_black = g == BG
    loops.sort(
        key=lambda item: (
            item["priority"],
            item["bbox"][0],
            item["bbox"][1],
            item["bbox"][2] - item["bbox"][0],
            item["bbox"][3] - item["bbox"][1],
        )
    )
    for item in loops:
        color = int(item["fill"])
        for r, c in item["interior"]:
            if original_black[r, c]:
                out[r, c] = color

    return out


def _randint(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, max(lo, hi))


def _sample_chain(depth: int, rng: random.Random):
    for _ in range(100):
        inner_h = _randint(rng, 5, 8 if depth >= 2 else 10)
        inner_w = _randint(rng, 5, 9 if depth >= 2 else 12)
        boxes = [(0, 0, inner_h - 1, inner_w - 1)]
        for _level in range(depth - 1):
            top = _randint(rng, NEST_MARGIN_MIN, NEST_MARGIN_MAX)
            bottom = _randint(rng, NEST_MARGIN_MIN, NEST_MARGIN_MAX)
            left = _randint(rng, NEST_MARGIN_MIN, NEST_MARGIN_MAX)
            right = _randint(rng, NEST_MARGIN_MIN, NEST_MARGIN_MAX)
            r0, c0, r1, c1 = boxes[0]
            boxes.insert(0, (r0 - top, c0 - left, r1 + bottom, c1 + right))

        min_r = min(r0 for r0, _, _, _ in boxes)
        min_c = min(c0 for _, c0, _, _ in boxes)
        boxes = [(r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in boxes]
        outer_h = boxes[0][2] - boxes[0][0] + 1
        outer_w = boxes[0][3] - boxes[0][1] + 1
        if outer_h <= 17 and outer_w <= 19:
            return boxes
    return None


def _boxes_separated(a, b, pad: int) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _pick_canvas(group_count: int, max_group_h: int, max_group_w: int, rng: random.Random):
    if group_count == 1:
        h_lo, h_hi, w_lo, w_hi = 16, 24, 18, 28
    elif group_count == 2:
        h_lo, h_hi, w_lo, w_hi = 20, 28, 24, 34
    else:
        h_lo, h_hi, w_lo, w_hi = 24, 30, 28, 36

    h_lo = max(h_lo, max_group_h + 4)
    w_lo = max(w_lo, max_group_w + 4)
    return _randint(rng, h_lo, h_hi), _randint(rng, w_lo, w_hi)


def _place_groups(chains, h: int, w: int, rng: random.Random):
    draw_h = h - 2
    order = sorted(
        range(len(chains)),
        key=lambda idx: -(
            (chains[idx][0][2] - chains[idx][0][0] + 1)
            * (chains[idx][0][3] - chains[idx][0][1] + 1)
        ),
    )
    placed_outer = []
    offsets = {}
    for idx in order:
        r0, c0, r1, c1 = chains[idx][0]
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if draw_h - box_h - 1 < 1 or w - box_w - 1 < 1:
            return None

        found = False
        for _ in range(250):
            top = _randint(rng, 1, draw_h - box_h - 1)
            left = _randint(rng, 1, w - box_w - 1)
            candidate = (top, left, top + box_h - 1, left + box_w - 1)
            if all(_boxes_separated(candidate, other, GROUP_MARGIN) for other in placed_outer):
                placed_outer.append(candidate)
                offsets[idx] = (top, left)
                found = True
                break
        if not found:
            return None

    translated = []
    for idx, chain in enumerate(chains):
        top, left = offsets[idx]
        translated.append([(r0 + top, c0 + left, r1 + top, c1 + left) for r0, c0, r1, c1 in chain])
    return translated


def _paint_rectangle_loop(grid: np.ndarray, bbox, fill_color: int, rng: random.Random):
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    side_colors = rng.sample([color for color in COLORS if color != fill_color], 2)
    if h > w or (h == w and rng.random() < 0.5):
        grid[r0, c0 + 1 : c1] = side_colors[0]
        grid[r1, c0 + 1 : c1] = side_colors[1]
        grid[r0 : r1 + 1, c0] = fill_color
        grid[r0 : r1 + 1, c1] = fill_color
    else:
        grid[r0 + 1 : r1, c0] = side_colors[0]
        grid[r0 + 1 : r1, c1] = side_colors[1]
        grid[r0, c0 : c1 + 1] = fill_color
        grid[r1, c0 : c1 + 1] = fill_color


def _legend_sequence(loops, rng: random.Random):
    by_depth = {}
    for loop in loops:
        by_depth.setdefault(loop["depth"], []).append(loop)

    ordered = []
    for depth in sorted(by_depth):
        group = list(by_depth[depth])
        rng.shuffle(group)
        ordered.extend(group)
    return ordered


def _paint_legend(grid: np.ndarray, colors, rng: random.Random):
    gaps = [_randint(rng, 1, 2) for _ in range(max(0, len(colors) - 1))]
    span = len(colors) + sum(gaps)
    start = _randint(rng, 1, grid.shape[1] - span - 1)
    col = start
    for idx, color in enumerate(colors):
        grid[-1, col] = color
        if idx < len(gaps):
            col += 1 + gaps[idx]


def _render_expected(grid: np.ndarray, ordered_loops):
    out = grid.copy()
    original_black = grid == BG
    for loop in ordered_loops:
        r0, c0, r1, c1 = loop["bbox"]
        fill_color = int(loop["fill"])
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if original_black[r, c]:
                    out[r, c] = fill_color
    return out


def _gen_attempt(rng: random.Random):
    depths = rng.choice(LAYOUT_OPTIONS)
    chains = []
    for depth in depths:
        chain = _sample_chain(depth, rng)
        if chain is None:
            return None
        chains.append(chain)

    max_group_h = max(chain[0][2] - chain[0][0] + 1 for chain in chains)
    max_group_w = max(chain[0][3] - chain[0][1] + 1 for chain in chains)
    h, w = _pick_canvas(len(chains), max_group_h, max_group_w, rng)
    placed = _place_groups(chains, h, w, rng)
    if placed is None:
        return None

    total_loops = sum(len(chain) for chain in placed)
    fill_colors = rng.sample(list(COLORS), total_loops)
    color_idx = 0

    grid = np.zeros((h, w), dtype=int)
    loops = []
    for chain in placed:
        for depth, bbox in enumerate(chain):
            fill_color = fill_colors[color_idx]
            color_idx += 1
            _paint_rectangle_loop(grid, bbox, fill_color, rng)
            loops.append({"bbox": bbox, "fill": fill_color, "depth": depth})

    ordered_loops = _legend_sequence(loops, rng)
    _paint_legend(grid, [loop["fill"] for loop in ordered_loops], rng)

    expected = _render_expected(grid, ordered_loops)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle")
