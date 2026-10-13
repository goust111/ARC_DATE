import random
from collections import Counter, deque

import numpy as np


BG = 0
DIVIDER = 5
SELECTOR_SIZE = 5
MIN_TARGET_AREA = 5
MAX_TARGET_AREA = 10
PALETTE = tuple(c for c in range(1, 10) if c != DIVIDER)
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _selector_candidate(window: np.ndarray) -> bool:
    if window.shape != (SELECTOR_SIZE, SELECTOR_SIZE):
        return False
    if not np.all(window[2, :] == DIVIDER):
        return False
    if not np.all(window[:, 2] == DIVIDER):
        return False

    quads = (
        window[:2, :2],
        window[:2, 3:5],
        window[3:5, :2],
        window[3:5, 3:5],
    )
    colors = []
    for quad in quads:
        values = np.unique(quad)
        if len(values) != 1:
            return False
        color = int(values[0])
        if color in (BG, DIVIDER):
            return False
        colors.append(color)
    return len(set(colors)) == 4


def _find_selector(grid: np.ndarray):
    h, w = grid.shape
    hits = []
    for r in range(h - SELECTOR_SIZE + 1):
        for c in range(w - SELECTOR_SIZE + 1):
            window = grid[r : r + SELECTOR_SIZE, c : c + SELECTOR_SIZE]
            if _selector_candidate(window):
                hits.append((r, c))
    assert len(hits) == 1, "expected exactly one selector block"
    return hits[0]


def _target_component(grid: np.ndarray, selector_rc, source_colors):
    sr, sc = selector_rc
    mask = np.isin(grid, list(source_colors))
    mask[sr : sr + SELECTOR_SIZE, sc : sc + SELECTOR_SIZE] = False
    parts = components_mask(mask)
    valid = []
    for coords in parts:
        if not (MIN_TARGET_AREA <= len(coords) <= MAX_TARGET_AREA):
            continue
        colors = {int(grid[r, c]) for r, c in coords}
        if colors != set(source_colors):
            continue
        valid.append(coords)
    assert len(valid) == 1, "expected exactly one target component"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Translate the source-color target pattern using the selector block's quadrant-defined color mapping.

    Input structure:
    - The background is black.
    - Exactly one 5x5 selector block is split by a grey cross at its middle row and middle column.
    - The four 2x2 selector quadrants are solid colors; the left column gives source colors and the right column gives their mapped target colors.
    - Outside the selector, exactly one 4-connected target pattern uses only the two source colors from the selector's left quadrants.

    Rule:
    - Detect the 5x5 selector block and read its quadrant colors.
    - Build the color mapping top-left -> top-right and bottom-left -> bottom-right.
    - Find the unique external 4-connected component made of the two source colors.
    - Recolor that component by the learned mapping and return its tight bounding-box crop on a black background.

    Key insight:
    - The selector block is not an index legend; its left and right quadrants form two example source-to-target color pairs that define the recoloring rule for the distant pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sr, sc = _find_selector(g)
    selector = g[sr : sr + SELECTOR_SIZE, sc : sc + SELECTOR_SIZE]
    a = int(selector[0, 0])
    b = int(selector[0, 3])
    c = int(selector[3, 0])
    d = int(selector[3, 3])
    mapping = {a: b, c: d}

    coords = _target_component(g, (sr, sc), (a, c))
    target_mask = np.zeros_like(g, dtype=bool)
    for r, c_ in coords:
        target_mask[r, c_] = True
    tr0, tc0, tr1, tc1 = bbox(target_mask)
    out = np.full((tr1 - tr0 + 1, tc1 - tc0 + 1), get_bg(g), dtype=int)
    for r, c_ in coords:
        out[r - tr0, c_ - tc0] = mapping[int(g[r, c_])]

    used = set(map(int, out[out != get_bg(g)]))
    assert used == {b, d}, "both mapping pairs must affect the output"
    return out


def _shape_bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _random_connected_shape(rng: random.Random):
    area = rng.randint(MIN_TARGET_AREA, MAX_TARGET_AREA)
    cells = {(0, 0)}
    frontier = [(0, 0)]
    while len(cells) < area:
        br, bc = frontier[rng.randrange(len(frontier))]
        options = []
        for nr, nc in neighbors4(br, bc):
            if (nr, nc) not in cells:
                options.append((nr, nc))
        if not options:
            frontier = list(cells)
            continue
        cell = options[rng.randrange(len(options))]
        cells.add(cell)
        frontier.append(cell)

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    norm = {(r - min(rows), c - min(cols)) for r, c in cells}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    if h > 5 or w > 5:
        return None
    if h * w == len(norm):
        return None
    if h == 1 or w == 1:
        return None
    return sorted(norm)


def _color_shape(coords, a: int, c: int, rng: random.Random):
    area = len(coords)
    indexed = list(range(area))
    rng.shuffle(indexed)
    split = rng.randint(1, area - 1)
    colors = {coords[i]: a for i in indexed[:split]}
    for i in indexed[split:]:
        colors[coords[i]] = c

    used = Counter(colors.values())
    adjacency = False
    cells = set(coords)
    for cell, color in colors.items():
        for nb in neighbors4(*cell):
            if nb in cells and colors[nb] != color:
                adjacency = True
                break
        if adjacency:
            break
    if min(used.values()) < 2:
        return None
    if not adjacency:
        return None
    return colors


def _paint_shape(grid: np.ndarray, top: int, left: int, color_map):
    for (r, c), color in color_map.items():
        grid[top + r, left + c] = color


def _random_distractor(rng: random.Random, banned_colors):
    color = rng.choice([x for x in PALETTE if x not in banned_colors])
    if rng.random() < 0.5:
        h = rng.randint(1, 2)
        w = rng.randint(2, 3)
        return np.full((h, w), color, dtype=int)

    h = rng.randint(2, 3)
    w = rng.randint(2, 3)
    patch = np.full((h, w), BG, dtype=int)
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < rng.randint(2, h * w - 1):
        rr, cc = rng.choice(tuple(cells))
        candidates = [(nr, nc) for nr, nc in neighbors4(rr, cc) if 0 <= nr < h and 0 <= nc < w]
        rng.shuffle(candidates)
        for cell in candidates:
            cells.add(cell)
            break
    for rr, cc in cells:
        patch[rr, cc] = color
    return patch


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, margin: int = 0) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    rr0 = max(0, top - margin)
    cc0 = max(0, left - margin)
    rr1 = min(gh, top + h + margin)
    cc1 = min(gw, left + w + margin)
    return np.all(grid[rr0:rr1, cc0:cc1] == BG)


def _validate_instance(grid: np.ndarray):
    sr, sc = _find_selector(grid)
    selector = grid[sr : sr + SELECTOR_SIZE, sc : sc + SELECTOR_SIZE]
    a = int(selector[0, 0])
    b = int(selector[0, 3])
    c = int(selector[3, 0])
    d = int(selector[3, 3])
    assert len({a, b, c, d}) == 4

    target = _target_component(grid, (sr, sc), (a, c))
    mask = np.zeros_like(grid, dtype=bool)
    for r, c_ in target:
        mask[r, c_] = True
    box = bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    assert r0 >= sr + SELECTOR_SIZE + 2
    assert c0 >= sc + SELECTOR_SIZE + 2
    colors = {int(grid[r, c_]) for r, c_ in target}
    assert colors == {a, c}
    counts = Counter(int(grid[r, c_]) for r, c_ in target)
    assert counts[a] >= 2 and counts[c] >= 2

    out = main(grid)
    assert out.ndim == 2 and max(out.shape) <= 30
    fg = set(map(int, out[out != BG]))
    assert fg == {b, d}
    assert np.count_nonzero(out != BG) == len(target)


def _gen_attempt(rng: random.Random):
    h = rng.randint(13, 19)
    w = rng.randint(13, 19)
    grid = np.full((h, w), BG, dtype=int)

    colors = rng.sample(list(PALETTE), 4)
    a, b, c, d = colors
    selector = np.array(
        [
            [a, a, DIVIDER, b, b],
            [a, a, DIVIDER, b, b],
            [DIVIDER, DIVIDER, DIVIDER, DIVIDER, DIVIDER],
            [c, c, DIVIDER, d, d],
            [c, c, DIVIDER, d, d],
        ],
        dtype=int,
    )
    grid[:SELECTOR_SIZE, :SELECTOR_SIZE] = selector

    coords = None
    color_map = None
    for _ in range(80):
        coords = _random_connected_shape(rng)
        if coords is None:
            continue
        color_map = _color_shape(coords, a, c, rng)
        if color_map is not None:
            break
    if coords is None or color_map is None:
        return None

    sr0, sc0, sr1, sc1 = _shape_bbox(coords)
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1
    min_top = SELECTOR_SIZE + 2
    min_left = SELECTOR_SIZE + 2
    if min_top + sh > h or min_left + sw > w:
        return None

    top = rng.randint(min_top, h - sh)
    left = rng.randint(min_left, w - sw)
    _paint_shape(grid, top, left, color_map)

    banned = {a, b, c, d, DIVIDER}
    for _ in range(rng.randint(0, 2)):
        patch = _random_distractor(rng, banned)
        placed = False
        for _ in range(40):
            pr = rng.randint(0, h - patch.shape[0])
            pc = rng.randint(0, w - patch.shape[1])
            if _can_place_patch(grid, patch, pr, pc, margin=1):
                mask = patch != BG
                grid[pr : pr + patch.shape[0], pc : pc + patch.shape[1]][mask] = patch[mask]
                placed = True
                break
        if not placed:
            continue

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
