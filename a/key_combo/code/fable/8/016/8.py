import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

MAX_BOUNCES = 6
GEN_RETRIES = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    best: tuple[int, tuple[int, int, int, int]] | None = None
    for comp in _components(grid == BLACK):
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (sub != BLACK).any():
            continue
        cand = (len(comp), (r0, c0, r1, c1))
        if best is None or cand[0] > best[0]:
            best = cand
    assert best is not None, "expected a black legend box"
    return best[1]


def _extract_keys(box: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    key_col = None
    for c in range(box.shape[1]):
        if (box[:, c] != BLACK).any():
            key_col = c
            break
    assert key_col is not None, "expected a keyed legend column"
    keys = [(r, int(box[r, key_col])) for r in range(box.shape[0]) if int(box[r, key_col]) != BLACK]
    assert keys, "expected at least one legend key"
    return key_col, keys


def _outside_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return not (r0 <= r <= r1 and c0 <= c <= c1)


def _find_emitter(
    grid: np.ndarray,
    color: int,
    legend_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            block_cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            if not all(_outside_bbox(cell, legend_bbox) for cell in block_cells):
                continue
            if not np.all(grid[r : r + 2, c : c + 2] == color):
                continue
            nozzles = []
            for dr, dc in DIRS:
                nr = r - 1 if dr < 0 else r + 2
                nc = c - 1 if dc < 0 else c + 2
                if _in_bounds(grid.shape, nr, nc) and _outside_bbox((nr, nc), legend_bbox):
                    if int(grid[nr, nc]) == color:
                        nozzles.append(((nr, nc), (dr, dc)))
            if len(nozzles) == 1:
                candidates.append(nozzles[0])
    assert len(candidates) == 1, f"expected one emitter for color {color}"
    return candidates[0]


def _count_bounces_to_wall(
    shape: tuple[int, int],
    grey_mask: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> int:
    h, w = shape
    r, c = start
    dr, dc = direction
    bounces = 0
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "beam must hit a wall before cycling"
        seen.add(state)

        nr, nc = r + dr, c + dc
        reflected = False
        if nr < 0 or nr >= h:
            dr = -dr
            reflected = True
        if nc < 0 or nc >= w:
            dc = -dc
            reflected = True
        if reflected:
            bounces += 1
            nr, nc = r + dr, c + dc

        if grey_mask[nr, nc]:
            return bounces
        r, c = nr, nc


def _render_legend(box_shape: tuple[int, int], key_col: int, keys: list[tuple[int, int]], counts: list[int]) -> np.ndarray:
    out = np.zeros(box_shape, dtype=int)
    for (r, color), count in zip(keys, counts):
        out[r, key_col] = color
        for i in range(1, count + 1):
            cc = key_col + 2 * i
            assert cc < box_shape[1], "legend row too short for bounce count"
            out[r, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate each keyed diagonal beam, count its boundary bounces, and redraw only the legend box.

    Input structure:
    - A black legend box contains one keyed color column whose non-black cells define the legend rows.
    - Outside the legend, each key color has exactly one emitter made of a same-colored 2x2 block plus a diagonally adjacent same-colored nozzle pixel.
    - Grey cells are absorbing walls, while extra same-colored single pixels are distractors.

    Rule:
    - Find the legend box and read its keyed rows from top to bottom.
    - For each key color, locate the unique 2x2 block with a same-colored diagonal nozzle; the nozzle direction sets a 45-degree beam.
    - Move the beam cell by cell, reflecting on grid edges and incrementing the bounce count once per boundary hit, until the beam enters a grey wall cell.
    - Output only the legend box, keeping the key pixel and adding that many same-colored dots every other cell to the right.

    Key insight:
    - The legend records a geometric statistic, not object counts: distractor pixels share the emitter colors, so the only reliable route is to identify the true emitter and simulate its reflected beam until absorption.
    """
    g = np.asarray(grid, dtype=int)
    legend_bbox = _find_legend_bbox(g)
    r0, c0, r1, c1 = legend_bbox
    box = g[r0 : r1 + 1, c0 : c1 + 1]
    key_col, keys = _extract_keys(box)
    grey_mask = g == GREY

    counts = []
    for _row, color in keys:
        start, direction = _find_emitter(g, color, legend_bbox)
        counts.append(_count_bounces_to_wall(g.shape, grey_mask, start, direction))
    return _render_legend(box.shape, key_col, keys, counts)


def _mark(blocked: np.ndarray, cells: list[tuple[int, int]], pad: int) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                blocked[rr, cc] = True


def _path_states(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[tuple[int, int], int]]:
    h, w = shape
    r, c = start
    dr, dc = direction
    bounces = 0
    seen = {(r, c, dr, dc)}
    states: list[tuple[tuple[int, int], int]] = []

    while True:
        nr, nc = r + dr, c + dc
        reflected = False
        if nr < 0 or nr >= h:
            dr = -dr
            reflected = True
        if nc < 0 or nc >= w:
            dc = -dc
            reflected = True
        if reflected:
            bounces += 1
            nr, nc = r + dr, c + dc

        states.append(((nr, nc), bounces))
        r, c = nr, nc
        state = (r, c, dr, dc)
        if state in seen:
            break
        seen.add(state)
    return states


def _desired_counts(n_keys: int) -> list[int]:
    counts = [random.randint(0, MAX_BOUNCES) for _ in range(n_keys)]
    if n_keys >= 2 and len(set(counts)) == 1:
        counts[0] = 0
        counts[1] = min(MAX_BOUNCES, max(2, counts[1]))
    if 0 not in counts:
        counts[random.randrange(n_keys)] = 0
    if not any(count >= 2 for count in counts):
        idx = random.randrange(n_keys)
        counts[idx] = random.randint(2, MAX_BOUNCES)
    random.shuffle(counts)
    return counts


def _try_place_emitter(
    grid: np.ndarray,
    blocked: np.ndarray,
    wall_cells: set[tuple[int, int]],
    forbidden_wall_cells: set[tuple[int, int]],
    color: int,
    target_count: int,
) -> dict | None:
    h, w = grid.shape
    for _ in range(250):
        dr, dc = random.choice(DIRS)
        r_lo = 1 if dr < 0 else 0
        r_hi = h - 2 if dr < 0 else h - 3
        c_lo = 1 if dc < 0 else 0
        c_hi = w - 2 if dc < 0 else w - 3
        if r_lo > r_hi or c_lo > c_hi:
            continue

        r = random.randint(r_lo, r_hi)
        c = random.randint(c_lo, c_hi)
        block = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
        nozzle = (r - 1 if dr < 0 else r + 2, c - 1 if dc < 0 else c + 2)
        visible = block + [nozzle]
        if any(blocked[rr, cc] for rr, cc in visible):
            continue

        states = _path_states((h, w), nozzle, (dr, dc))
        first_index: dict[tuple[int, int], int] = {}
        for idx, (cell, _count) in enumerate(states):
            first_index.setdefault(cell, idx)

        candidates = []
        for cell, idx in first_index.items():
            count = states[idx][1]
            if count != target_count:
                continue
            if blocked[cell[0], cell[1]] or cell in forbidden_wall_cells:
                continue
            prefix = [states[i][0] for i in range(idx + 1)]
            if any(step in wall_cells for step in prefix[:-1]):
                continue
            candidates.append((cell, prefix))

        if not candidates:
            continue

        wall_cell, prefix = random.choice(candidates)
        return {
            "block": block,
            "nozzle": nozzle,
            "direction": (dr, dc),
            "wall": wall_cell,
            "prefix": prefix,
            "count": target_count,
        }
    return None


def _place_extra_walls(
    grid: np.ndarray,
    blocked: np.ndarray,
    forbidden: set[tuple[int, int]],
    bg: int,
) -> None:
    h, w = grid.shape
    for _ in range(random.randint(0, 2)):
        for _ in range(80):
            horizontal = random.choice([False, True])
            length = random.randint(2, 3)
            if horizontal:
                r = random.randint(0, h - 1)
                c = random.randint(0, w - length)
                cells = [(r, c + i) for i in range(length)]
            else:
                r = random.randint(0, h - length)
                c = random.randint(0, w - 1)
                cells = [(r + i, c) for i in range(length)]
            if any(blocked[rr, cc] for rr, cc in cells):
                continue
            if any(cell in forbidden for cell in cells):
                continue
            if any(int(grid[rr, cc]) != bg for rr, cc in cells):
                continue
            for rr, cc in cells:
                grid[rr, cc] = GREY
            _mark(blocked, cells, pad=0)
            break


def _place_distractors(
    grid: np.ndarray,
    blocked: np.ndarray,
    colors: list[int],
    bg: int,
) -> None:
    h, w = grid.shape
    for color in colors:
        for _ in range(random.randint(0, 2)):
            for _ in range(60):
                r = random.randint(0, h - 1)
                c = random.randint(0, w - 1)
                if blocked[r, c] or int(grid[r, c]) != bg:
                    continue
                grid[r, c] = color
                _mark(blocked, [(r, c)], pad=1)
                break


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 30)
    n_keys = random.randint(2, 4)
    legend_h = 2 * n_keys + 1
    legend_w = 2 * MAX_BOUNCES + 5
    if legend_h > h or legend_w > w:
        return None

    bg = random.choice(COLOR_POOL)
    key_colors = random.sample([c for c in COLOR_POOL if c != bg], n_keys)
    target_counts = _desired_counts(n_keys)

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    legend_top = random.choice([0, h - legend_h])
    legend_left = random.randint(0, w - legend_w)
    legend_rows = [legend_top + 1 + 2 * i for i in range(n_keys)]
    key_col = legend_left + 1

    grid[legend_top : legend_top + legend_h, legend_left : legend_left + legend_w] = BLACK
    for row, color in zip(legend_rows, key_colors):
        grid[row, key_col] = color
    legend_cells = [
        (r, c)
        for r in range(legend_top, legend_top + legend_h)
        for c in range(legend_left, legend_left + legend_w)
    ]
    _mark(blocked, legend_cells, pad=1)

    wall_cells: set[tuple[int, int]] = set()
    forbidden_wall_cells: set[tuple[int, int]] = set()
    all_prefix_cells: set[tuple[int, int]] = set()
    placed_counts: list[int] = []

    for color, target_count in zip(key_colors, target_counts):
        placed = _try_place_emitter(grid, blocked, wall_cells, forbidden_wall_cells, color, target_count)
        if placed is None:
            return None

        for rr, cc in placed["block"]:
            grid[rr, cc] = color
        nr, nc = placed["nozzle"]
        grid[nr, nc] = color
        _mark(blocked, placed["block"] + [placed["nozzle"]], pad=1)

        wr, wc = placed["wall"]
        if blocked[wr, wc]:
            return None
        grid[wr, wc] = GREY
        _mark(blocked, [placed["wall"]], pad=0)

        wall_cells.add(placed["wall"])
        forbidden_wall_cells.update(placed["prefix"])
        all_prefix_cells.update(placed["prefix"])
        placed_counts.append(placed["count"])

    if len(set(placed_counts)) == 1:
        return None
    if 0 not in placed_counts or not any(count >= 2 for count in placed_counts):
        return None

    _place_extra_walls(grid, blocked, all_prefix_cells, bg)
    _place_distractors(grid, blocked, key_colors, bg)

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected = _render_legend((legend_h, legend_w), 1, list(zip([1 + 2 * i for i in range(n_keys)], key_colors)), placed_counts)
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
