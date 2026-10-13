from collections import Counter, deque
import random

import numpy as np


BG = 0
DIGIT_COLOR = 1
WALL = 5
LEGEND_WIDTH = 7

DIGIT_SHAPES = {
    1: (
        (0, 1),
        (1, 0), (1, 1),
        (2, 1),
        (3, 1),
        (4, 0), (4, 1), (4, 2),
    ),
    2: (
        (0, 0), (0, 1), (0, 2),
        (1, 2),
        (2, 0), (2, 1), (2, 2),
        (3, 0),
        (4, 0), (4, 1), (4, 2),
    ),
    3: (
        (0, 0), (0, 1), (0, 2),
        (1, 2),
        (2, 0), (2, 1), (2, 2),
        (3, 2),
        (4, 0), (4, 1), (4, 2),
    ),
}

PATTERN_SHAPES = [
    ((1, 0), (2, 0), (2, 1)),
    ((1, 0), (2, 0), (2, 1), (2, 2)),
    ((1, 0), (1, 1), (2, 1), (3, 1)),
    ((1, 0), (2, 0), (2, 1), (3, 1)),
    ((1, 0), (1, 1), (2, 1), (3, 1), (3, 2)),
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


DIGIT_LOOKUP = {_normalize(shape): digit for digit, shape in DIGIT_SHAPES.items()}


def _parse_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    mapping: dict[int, int] = {}
    legend_colors = {DIGIT_COLOR}
    for comp in _components(grid == DIGIT_COLOR):
        key = _normalize(comp)
        if key not in DIGIT_LOOKUP:
            continue
        digit = DIGIT_LOOKUP[key]
        r0, _c0, r1, c1 = _bbox(comp)
        window = grid[r0 : r1 + 1, c1 + 1 : min(grid.shape[1], c1 + 5)]
        colors = [int(v) for v in window.ravel() if int(v) not in (BG, WALL, DIGIT_COLOR)]
        assert colors, f"legend digit {digit} must have an adjacent color swatch"
        swatch = Counter(colors).most_common(1)[0][0]
        mapping[digit] = int(swatch)
        legend_colors.add(int(swatch))
    assert mapping, "legend mapping must be present"
    legend_mask = np.isin(grid, list(legend_colors))
    return mapping, legend_mask


def _shortest_path(
    walkable: np.ndarray,
    starts: set[tuple[int, int]],
    goals: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    assert starts and goals, "pattern endpoints must touch the corridor"
    h, w = walkable.shape
    q = deque(sorted(starts))
    prev: dict[tuple[int, int], tuple[int, int] | None] = {p: None for p in starts}
    goal_set = set(goals)
    end = None
    while q:
        cur = q.popleft()
        if cur in goal_set:
            end = cur
            break
        for nr, nc in _neighbors4(cur[0], cur[1]):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev or not bool(walkable[nr, nc]):
                continue
            prev[nxt] = cur
            q.append(nxt)
    assert end is not None, "paired patterns must be connected by a corridor path"
    path: list[tuple[int, int]] = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _count_turns(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    dirs = [(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
    return sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each paired corridor path by looking up its turn count in the edge legend.

    Input structure:
    - Background is black and grey wall pixels carve the grid into one-cell-wide corridor channels.
    - Colored 3-5-pixel sprites sit inside corridor pockets, and each sprite appears exactly twice with the same color and exact same orientation-preserving shape.
    - A legend on the grid edge contains pixel-drawn digits with a color swatch immediately to the right of each digit.

    Rule:
    - Match the sprites into pairs using exact shape and color equality.
    - For each pair, find the shortest corridor path between the black corridor cells adjacent to the two sprites.
    - Count how many times that shortest path switches between horizontal and vertical motion.
    - Use the legend's digit-to-color mapping to recolor every corridor cell on that path, leaving the sprite pixels unchanged.
    - Corridor cells not used by any matched path stay black, and walls and legend pixels stay unchanged.

    Key insight:
    The transformation is a three-step chain: pair identical sprites, measure the matched path's turn count, then translate that count through the digit legend into the final path color.
    """
    g = np.asarray(grid, dtype=int)
    legend_map, legend_mask = _parse_legend(g)
    out = g.copy()
    walkable = g == BG

    groups: dict[tuple[int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]] = {}
    for color in sorted(int(v) for v in np.unique(g)):
        if color in (BG, WALL) or bool(np.any(legend_mask & (g == color))):
            continue
        for comp in _components(g == color):
            if any(bool(legend_mask[r, c]) for r, c in comp):
                continue
            assert 3 <= len(comp) <= 5, "sprite components must have 3-5 pixels"
            groups.setdefault((color, _normalize(comp)), []).append(comp)

    for (_color, _shape), comps in groups.items():
        assert len(comps) == 2, "each sprite signature must appear exactly twice"
        endpoints = []
        for comp in comps:
            frontier = set()
            for r, c in comp:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and bool(walkable[nr, nc]):
                        frontier.add((nr, nc))
            assert frontier, "every sprite must touch the corridor"
            endpoints.append(frontier)
        path = _shortest_path(walkable, endpoints[0], endpoints[1])
        turns = _count_turns(path)
        assert turns in legend_map, f"missing legend entry for {turns} turns"
        out[tuple(np.array(path).T)] = legend_map[turns]
    return out


def _paint_digit(grid: np.ndarray, digit: int, top: int, left: int) -> None:
    for dr, dc in DIGIT_SHAPES[digit]:
        grid[top + dr, left + dc] = DIGIT_COLOR


def _carve_segment(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        c0, c1 = sorted((ac, bc))
        mask[ar, c0 : c1 + 1] = True
        return
    assert ac == bc
    r0, r1 = sorted((ar, br))
    mask[r0 : r1 + 1, ac] = True


def _canonical_masks(turns: int, shape: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, np.ndarray]:
    if turns == 1:
        h = random.randint(8, 10)
        w = random.randint(8, 11)
        top_row = random.randint(1, 2)
        start = (h - 4, 1)
        end = (top_row, w - 4)
        vertices = [start, (top_row, 1), end]
    elif turns == 2:
        h = random.randint(8, 10)
        w = random.randint(8, 11)
        top_row = random.randint(1, 2)
        end_row = random.randint(top_row + 2, h - 4)
        start = (h - 4, 1)
        end = (end_row, w - 4)
        vertices = [start, (top_row, 1), (top_row, w - 4), end]
    else:
        h = random.randint(8, 10)
        w = random.randint(10, 13)
        top_row = random.randint(1, 2)
        mid_row = random.randint(top_row + 2, h - 4)
        mid_col = random.randint(3, w - 6)
        start = (h - 4, 1)
        end = (mid_row, w - 4)
        vertices = [start, (top_row, 1), (top_row, mid_col), (mid_row, mid_col), end]

    path_mask = np.zeros((h, w), dtype=bool)
    for a, b in zip(vertices, vertices[1:]):
        _carve_segment(path_mask, a, b)

    pattern_mask = np.zeros((h, w), dtype=bool)
    for door in (vertices[0], vertices[-1]):
        for dr, dc in shape:
            rr, cc = door[0] + dr, door[1] + dc
            assert 0 <= rr < h and 0 <= cc < w
            assert not bool(path_mask[rr, cc]), "pattern must not overlap corridor"
            pattern_mask[rr, cc] = True
    return path_mask, pattern_mask


def _transform_mask(mask: np.ndarray, flip: bool, rot: int) -> np.ndarray:
    out = np.fliplr(mask) if flip else mask.copy()
    return np.rot90(out, rot)


def _build_motif(turns: int, pattern_color: int, path_color: int, shape: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(50):
        path_mask, pattern_mask = _canonical_masks(turns, shape)
        flip = bool(random.getrandbits(1))
        rot = random.randrange(4)
        path_mask = _transform_mask(path_mask, flip, rot)
        pattern_mask = _transform_mask(pattern_mask, flip, rot)

        h, w = path_mask.shape
        inp = np.full((h, w), WALL, dtype=int)
        inp[path_mask] = BG
        inp[pattern_mask] = pattern_color

        comps = _components(inp == pattern_color)
        if len(comps) != 2 or not all(3 <= len(comp) <= 5 for comp in comps):
            continue
        if any(
            len(
                {
                    (nr, nc)
                    for r, c in comp
                    for nr, nc in _neighbors4(r, c)
                    if 0 <= nr < h and 0 <= nc < w and int(inp[nr, nc]) == BG
                }
            ) != 1
            for comp in comps
        ):
            continue

        out = inp.copy()
        out[path_mask] = path_color
        return inp, out
    raise AssertionError("failed to build a valid local motif")


def _draw_legend(grid: np.ndarray, legend_map: dict[int, int], order: list[int], top: int) -> None:
    for idx, digit in enumerate(order):
        r0 = top + idx * 6
        _paint_digit(grid, digit, r0, 0)
        grid[r0 + 1 : r0 + 3, 4:6] = legend_map[digit]


def _gen_attempt() -> np.ndarray | None:
    palette = [2, 3, 4, 6, 7, 8, 9]
    legend_colors = random.sample(palette, 3)
    legend_map = {1: legend_colors[0], 2: legend_colors[1], 3: legend_colors[2]}
    remaining = [c for c in palette if c not in legend_colors]

    pair_count = random.randint(2, 3)
    pair_turns = random.sample([1, 2, 3], pair_count)
    pattern_colors = random.sample(remaining, pair_count)

    motifs = []
    for turns, color in zip(pair_turns, pattern_colors):
        shape = random.choice(PATTERN_SHAPES)
        inp_box, out_box = _build_motif(turns, color, legend_map[turns], shape)
        motifs.append({"inp": inp_box, "out": out_box})
    random.shuffle(motifs)

    heights = [int(m["inp"].shape[0]) for m in motifs]
    widths = [int(m["inp"].shape[1]) for m in motifs]
    gaps = [random.randint(1, 2) for _ in range(max(0, pair_count - 1))]
    motif_stack_h = sum(heights) + sum(gaps)
    legend_h = 17

    h_min = max(legend_h, motif_stack_h)
    if h_min > 30:
        return None
    h = h_min + random.randint(0, min(30 - h_min, 3))

    w_min = LEGEND_WIDTH + max(widths)
    if w_min > 30:
        return None
    w = w_min + random.randint(0, min(30 - w_min, 4))

    grid = np.full((h, w), WALL, dtype=int)
    expected = np.full((h, w), WALL, dtype=int)
    grid[:, : LEGEND_WIDTH - 1] = BG
    expected[:, : LEGEND_WIDTH - 1] = BG

    legend_top = random.randint(0, h - legend_h)
    legend_order = list(legend_map)
    random.shuffle(legend_order)
    _draw_legend(grid, legend_map, legend_order, legend_top)
    _draw_legend(expected, legend_map, legend_order, legend_top)

    y = random.randint(0, h - motif_stack_h)
    for idx, motif in enumerate(motifs):
        box_h, box_w = motif["inp"].shape
        x = random.randint(LEGEND_WIDTH, w - box_w)
        grid[y : y + box_h, x : x + box_w] = motif["inp"]
        expected[y : y + box_h, x : x + box_w] = motif["out"]
        if idx < len(gaps):
            y += box_h + gaps[idx]

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for task 4")
