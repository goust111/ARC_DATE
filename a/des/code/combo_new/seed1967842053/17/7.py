from collections import Counter, defaultdict, deque
import random
import numpy as np

BLACK = 0
GREY = 5
MAROON = 9


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells):
    if not cells:
        return ()
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _hole_key_from_colored_component(cells):
    r0, c0, r1, c1 = _bbox(cells)
    bh, bw = r1 - r0 + 1, c1 - c0 + 1
    solid = np.zeros((bh, bw), dtype=bool)
    for r, c in cells:
        solid[r - r0, c - c0] = True

    exterior = np.zeros((bh, bw), dtype=bool)
    q = deque()
    for r in range(bh):
        for c in (0, bw - 1):
            if not solid[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(bw):
        for r in (0, bh - 1):
            if not solid[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < bh and 0 <= nc < bw and not solid[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    holes = [(r, c) for r in range(bh) for c in range(bw) if not solid[r, c] and not exterior[r, c]]
    return _normalize(holes)


def _target_infos(grid):
    target_mask = (grid == GREY) | (grid == MAROON)
    infos = []
    for cells in _components(target_mask):
        r0, c0, r1, c1 = _bbox(cells)
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        bh, bw = sub.shape
        if not (3 <= bh <= 5 and 3 <= bw <= 5):
            continue
        if not np.any(sub == GREY) or not np.any(sub == MAROON):
            continue
        if not np.all((sub == GREY) | (sub == MAROON)):
            continue
        maroon = [(r, c) for r in range(bh) for c in range(bw) if sub[r, c] == MAROON]
        infos.append((r0, c0, r1, c1, _normalize(maroon)))
    return infos


def _left_records(grid):
    records = defaultdict(list)
    for cells in _components(grid != BLACK):
        vals = {int(grid[r, c]) for r, c in cells}
        if GREY in vals or len(vals) != 1:
            continue
        r0, c0, r1, c1 = _bbox(cells)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if not (3 <= bh <= 5 and 3 <= bw <= 5):
            continue
        key = _hole_key_from_colored_component(cells)
        if key:
            records[key].append((next(iter(vals)), r0, c0))
    return records


def _chosen_color(entries):
    counts = Counter(color for color, _, _ in entries)
    top_count = max(counts.values())
    tied = [color for color, count in counts.items() if count == top_count]

    def first_position(color):
        return min((r, c) for entry_color, r, c in entries if entry_color == color)

    return min(tied, key=first_position)


def main(grid: np.ndarray) -> np.ndarray:
    """Color each right-side target block by the majority outline color of left patterns with the same internal hole shape.

    Input structure:
    The grid has a black background. The left area contains small single-color connected patterns in 3x3 to 5x5 bounding boxes; each pattern encloses one or more black hole cells. The right area contains solid grey 3x3 to 5x5 target blocks whose interior maroon cells form a small lookup shape.

    Rule:
    Leave all left-side patterns unchanged. For each grey target block, normalize the shape made by its maroon cells and match it to the normalized enclosed black hole shape of left-side patterns. Among matching left patterns, count their outline colors. Repaint the target block's grey cells with the most frequent matching color, breaking count ties by the matching pattern that appears highest in the grid, then leftmost. Preserve the maroon lookup cells.

    Key insight:
    The class label is the hidden black hole shape inside left patterns; the visible maroon shape in a target block is a key that retrieves the majority outline color for that hidden-hole class.
    """
    out = np.array(grid, copy=True)
    records = _left_records(grid)
    for r0, c0, r1, c1, key in _target_infos(grid):
        entries = records.get(key)
        if not entries:
            continue
        color = _chosen_color(entries)
        sub = out[r0:r1 + 1, c0:c1 + 1]
        sub[sub == GREY] = color
    return out


_SHAPES = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 1), (1, 0), (1, 2), (2, 1)),
]
_PALETTE = [1, 2, 3, 4, 6, 7, 8]


def _shape_size(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _random_block_size(shape):
    sh, sw = _shape_size(shape)
    return random.randint(sh + 2, 5), random.randint(sw + 2, 5)


def _random_offset(shape, bh, bw):
    sh, sw = _shape_size(shape)
    return random.randint(1, bh - sh - 1), random.randint(1, bw - sw - 1)


def _place_rect(occupied, bh, bw, cmin, cmax):
    rows = occupied.shape[0]
    if cmax - cmin + 1 < bw or rows < bh:
        return None
    for _ in range(300):
        r = random.randint(0, rows - bh)
        c = random.randint(cmin, cmax - bw + 1)
        rr0, rr1 = max(0, r - 1), min(rows, r + bh + 1)
        cc0, cc1 = max(0, c - 1), min(occupied.shape[1], c + bw + 1)
        if not occupied[rr0:rr1, cc0:cc1].any():
            occupied[r:r + bh, c:c + bw] = True
            return r, c
    return None


def _draw_left(grid, top, left, bh, bw, shape, color):
    grid[top:top + bh, left:left + bw] = color
    off_r, off_c = _random_offset(shape, bh, bw)
    for dr, dc in shape:
        grid[top + off_r + dr, left + off_c + dc] = BLACK


def _draw_target(grid, top, left, bh, bw, shape):
    grid[top:top + bh, left:left + bw] = GREY
    off_r, off_c = _random_offset(shape, bh, bw)
    for dr, dc in shape:
        grid[top + off_r + dr, left + off_c + dc] = MAROON


def _color_plan(n):
    colors = random.sample(_PALETTE, min(4, len(_PALETTE)))
    tie = random.choice([False, True])
    if tie:
        if n == 2:
            plan = colors[:2]
        elif n == 3:
            plan = colors[:3]
        else:
            plan = [colors[0], colors[0], colors[1], colors[1]]
    else:
        if n == 2:
            plan = [colors[0], colors[0]]
        elif n == 3:
            plan = [colors[0], colors[0], colors[1]]
        else:
            plan = [colors[0], colors[0], colors[0], colors[1]]
    random.shuffle(plan)
    return plan


def generate_input() -> np.ndarray:
    for _ in range(200):
        rows = random.randint(18, 30)
        cols = random.randint(24, 30)
        split = cols // 2
        grid = np.zeros((rows, cols), dtype=int)
        occupied = np.zeros((rows, cols), dtype=bool)
        left_max = split - 2
        right_min = split + 1
        if left_max < 6 or cols - right_min < 6:
            continue

        active_count = random.choice([3, 3, 4])
        active_shapes = random.sample(_SHAPES, active_count)
        specs = []
        ok = True
        for shape in active_shapes:
            bh, bw = _random_block_size(shape)
            n = random.choice([2, 2, 3])
            specs.append((shape, bh, bw, _color_plan(n)))

        for shape, bh, bw, colors in specs:
            for color in colors:
                pos = _place_rect(occupied, bh, bw, 0, left_max)
                if pos is None:
                    ok = False
                    break
                _draw_left(grid, pos[0], pos[1], bh, bw, shape, color)
            if not ok:
                break
        if not ok:
            continue

        target_specs = []
        for spec in specs:
            target_specs.append(spec)
        for spec in random.sample(specs, random.randint(0, min(2, len(specs)))):
            target_specs.append(spec)
        random.shuffle(target_specs)

        for shape, bh, bw, _ in target_specs:
            pos = _place_rect(occupied, bh, bw, right_min, cols - 1)
            if pos is None:
                ok = False
                break
            _draw_target(grid, pos[0], pos[1], bh, bw, shape)
        if ok:
            return grid

    grid = np.zeros((20, 24), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)
    specs = [
        (_SHAPES[0], 3, 3, [2, 2, 4]),
        (_SHAPES[1], 3, 4, [1, 3]),
        (_SHAPES[2], 4, 3, [6, 6]),
    ]
    for shape, bh, bw, colors in specs:
        for color in colors:
            pos = _place_rect(occupied, bh, bw, 0, 10)
            _draw_left(grid, pos[0], pos[1], bh, bw, shape, color)
    for shape, bh, bw, _ in specs:
        pos = _place_rect(occupied, bh, bw, 13, 23)
        _draw_target(grid, pos[0], pos[1], bh, bw, shape)
    return grid
