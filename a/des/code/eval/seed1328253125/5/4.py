from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


WALL_COLOR = 5
GRID_MIN = 19
GRID_MAX = 30
LINE_COUNT_MIN = 2
LINE_COUNT_MAX = 4
MIN_REGION_SPAN = 3
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _detect_wall_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    best_color = None
    best_score = None
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        line_like = 0
        straight_like = 0
        for r, c in coords:
            r = int(r)
            c = int(c)
            up = r > 0 and int(grid[r - 1, c]) == color
            down = r + 1 < h and int(grid[r + 1, c]) == color
            left = c > 0 and int(grid[r, c - 1]) == color
            right = c + 1 < w and int(grid[r, c + 1]) == color
            if up or down or left or right:
                line_like += 1
            if (left and right and not up and not down) or (up and down and not left and not right):
                straight_like += 1
        score = (line_like, straight_like, int(coords.shape[0]))
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None, "expected at least one non-background color"
    return int(best_color)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _region_map_and_seeds(grid: np.ndarray, wall_color: int, bg: int) -> tuple[np.ndarray, list[int | None]]:
    h, w = grid.shape
    region_ids = np.full((h, w), -1, dtype=int)
    seeds: list[int | None] = []
    for rid, comp in enumerate(_components(grid != wall_color)):
        color_counts = Counter(
            int(grid[r, c])
            for r, c in comp
            if int(grid[r, c]) not in (bg, wall_color)
        )
        seed_color = color_counts.most_common(1)[0][0] if color_counts else None
        seeds.append(seed_color)
        for r, c in comp:
            region_ids[r, c] = rid
    return region_ids, seeds


def _node_mask(wall_mask: np.ndarray) -> np.ndarray:
    h, w = wall_mask.shape
    nodes = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(wall_mask):
        r = int(r)
        c = int(c)
        up = r > 0 and wall_mask[r - 1, c]
        down = r + 1 < h and wall_mask[r + 1, c]
        left = c > 0 and wall_mask[r, c - 1]
        right = c + 1 < w and wall_mask[r, c + 1]
        straight_h = left and right and not up and not down
        straight_v = up and down and not left and not right
        if not (straight_h or straight_v):
            nodes[r, c] = True
    return nodes


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _trace_segments(wall_mask: np.ndarray, nodes: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = wall_mask.shape
    seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    segments: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(nodes):
        start = (int(r), int(c))
        for nr, nc in _neighbors4(start[0], start[1], h, w):
            if not wall_mask[nr, nc]:
                continue
            key = _edge_key(start, (nr, nc))
            if key in seen_edges:
                continue
            path = [start]
            prev = start
            cur = (nr, nc)
            while True:
                seen_edges.add(_edge_key(prev, cur))
                path.append(cur)
                if nodes[cur]:
                    break
                next_steps = [step for step in _neighbors4(cur[0], cur[1], h, w) if wall_mask[step] and step != prev]
                if not next_steps:
                    break
                prev, cur = cur, next_steps[0]
            if len(path) >= 2:
                segments.append(path)
    return segments


def _majority(items: list[int]) -> int | None:
    if not items:
        return None
    return int(Counter(items).most_common(1)[0][0])


def _segment_side_regions(
    path: list[tuple[int, int]],
    orientation: str,
    region_ids: np.ndarray,
) -> tuple[int | None, int | None]:
    h, w = region_ids.shape
    side_a: list[int] = []
    side_b: list[int] = []
    for r, c in path:
        if orientation == "h":
            a = (r - 1, c)
            b = (r + 1, c)
        else:
            a = (r, c - 1)
            b = (r, c + 1)
        if not (0 <= a[0] < h and 0 <= a[1] < w and 0 <= b[0] < h and 0 <= b[1] < w):
            continue
        ra = int(region_ids[a])
        rb = int(region_ids[b])
        if ra != -1 and rb != -1:
            side_a.append(ra)
            side_b.append(rb)
    return _majority(side_a), _majority(side_b)


def _line_color(seed_a: int | None, seed_b: int | None, wall_color: int) -> int:
    if seed_a is None and seed_b is None:
        return wall_color
    if seed_a is None:
        return int(seed_b)
    if seed_b is None:
        return int(seed_a)
    if int(seed_a) == int(seed_b):
        return int(seed_a)
    return wall_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each wall-separated region with its seed color and recolor each wall segment
    from the seed relationship of the regions on its two sides.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    wall_color = _detect_wall_color(g, bg)
    region_ids, region_seeds = _region_map_and_seeds(g, wall_color, bg)

    out = g.copy()
    for rid, seed_color in enumerate(region_seeds):
        if seed_color is None:
            continue
        fill_mask = (region_ids == rid) & (g == bg)
        out[fill_mask] = int(seed_color)

    wall_mask = g == wall_color
    nodes = _node_mask(wall_mask)
    segments = _trace_segments(wall_mask, nodes)

    incident_colors: dict[tuple[int, int], list[int]] = {}
    for path in segments:
        orientation = "h" if path[0][0] == path[1][0] else "v"
        region_a, region_b = _segment_side_regions(path, orientation, region_ids)
        seed_a = region_seeds[region_a] if region_a is not None else None
        seed_b = region_seeds[region_b] if region_b is not None else None
        color = _line_color(seed_a, seed_b, wall_color)

        for r, c in path[1:-1]:
            out[r, c] = color
        incident_colors.setdefault(path[0], []).append(color)
        incident_colors.setdefault(path[-1], []).append(color)

    for node, colors in incident_colors.items():
        unique = {int(color) for color in colors}
        out[node] = colors[0] if len(unique) == 1 else wall_color

    return out


def _sample_positions(length: int, count: int, rng: random.Random) -> list[int]:
    candidates = list(range(MIN_REGION_SPAN, length - MIN_REGION_SPAN))
    for _ in range(200):
        positions = sorted(rng.sample(candidates, count))
        left_span = positions[0]
        right_span = length - 1 - positions[-1]
        if left_span < MIN_REGION_SPAN or right_span < MIN_REGION_SPAN:
            continue
        if all(b - a >= MIN_REGION_SPAN + 1 for a, b in zip(positions, positions[1:], strict=False)):
            return positions
    raise RuntimeError("failed to place partition lines")


def _region_bands(lines: list[int], size: int) -> list[tuple[int, int]]:
    bounds = [-1] + lines + [size]
    return [(bounds[i] + 1, bounds[i + 1] - 1) for i in range(len(bounds) - 1)]


def _adjacency_edges(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols - 1):
            edges.append(((r, c), (r, c + 1)))
    for r in range(rows - 1):
        for c in range(cols):
            edges.append(((r, c), (r + 1, c)))
    return edges


def _pick_disjoint_edges(
    rows: int,
    cols: int,
    count: int,
    rng: random.Random,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    all_edges = _adjacency_edges(rows, cols)
    for _ in range(200):
        rng.shuffle(all_edges)
        chosen: list[tuple[tuple[int, int], tuple[int, int]]] = []
        used: set[tuple[int, int]] = set()
        for edge in all_edges:
            a, b = edge
            if a in used or b in used:
                continue
            chosen.append(edge)
            used.add(a)
            used.add(b)
            if len(chosen) == count:
                return chosen
    raise RuntimeError("failed to find disjoint edges")


def _branch_key(seed_a: int | None, seed_b: int | None) -> str:
    if seed_a is None and seed_b is None:
        return "none"
    if seed_a is None or seed_b is None:
        return "one"
    if int(seed_a) == int(seed_b):
        return "same"
    return "different"


def _build_region_seeds(
    rows: int,
    cols: int,
    seed_palette: list[int],
    rng: random.Random,
) -> list[list[int | None]]:
    unset = object()
    grid: list[list[object | int | None]] = [[unset for _ in range(cols)] for _ in range(rows)]
    edges = _pick_disjoint_edges(rows, cols, 4, rng)
    cases = ["same", "different", "one", "none"]
    rng.shuffle(cases)
    for edge, case in zip(edges, cases, strict=False):
        (r1, c1), (r2, c2) = edge
        if case == "same":
            color = rng.choice(seed_palette)
            grid[r1][c1] = color
            grid[r2][c2] = color
        elif case == "different":
            color_a, color_b = rng.sample(seed_palette, 2)
            grid[r1][c1] = color_a
            grid[r2][c2] = color_b
        elif case == "one":
            color = rng.choice(seed_palette)
            if rng.random() < 0.5:
                grid[r1][c1] = color
                grid[r2][c2] = None
            else:
                grid[r1][c1] = None
                grid[r2][c2] = color
        else:
            grid[r1][c1] = None
            grid[r2][c2] = None

    pool: list[int | None] = [None, None] + seed_palette + seed_palette
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is unset:
                grid[r][c] = rng.choice(pool)
    return [[None if cell is None else int(cell) for cell in row] for row in grid]


def _segment_branch_counts(region_seeds: list[list[int | None]]) -> Counter[str]:
    rows = len(region_seeds)
    cols = len(region_seeds[0])
    counts: Counter[str] = Counter()
    for r in range(rows):
        for c in range(cols - 1):
            counts[_branch_key(region_seeds[r][c], region_seeds[r][c + 1])] += 1
    for r in range(rows - 1):
        for c in range(cols):
            counts[_branch_key(region_seeds[r][c], region_seeds[r + 1][c])] += 1
    return counts


def _seed_position(
    band_r: tuple[int, int],
    band_c: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int]:
    r0, r1 = band_r
    c0, c1 = band_c
    inner_r0 = r0 + 1 if r1 - r0 + 1 >= 3 else r0
    inner_r1 = r1 - 1 if r1 - r0 + 1 >= 3 else r1
    inner_c0 = c0 + 1 if c1 - c0 + 1 >= 3 else c0
    inner_c1 = c1 - 1 if c1 - c0 + 1 >= 3 else c1
    return rng.randint(inner_r0, inner_r1), rng.randint(inner_c0, inner_c1)


def _render_input(
    height: int,
    width: int,
    bg: int,
    wall_color: int,
    row_lines: list[int],
    col_lines: list[int],
    region_seeds: list[list[int | None]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    if row_lines:
        grid[row_lines, :] = wall_color
    if col_lines:
        grid[:, col_lines] = wall_color

    row_bands = _region_bands(row_lines, height)
    col_bands = _region_bands(col_lines, width)
    for r in range(len(row_bands)):
        for c in range(len(col_bands)):
            seed_color = region_seeds[r][c]
            if seed_color is None:
                continue
            sr, sc = _seed_position(row_bands[r], col_bands[c], rng)
            grid[sr, sc] = int(seed_color)
    return grid


def _expected_output(
    grid: np.ndarray,
    bg: int,
    wall_color: int,
    row_lines: list[int],
    col_lines: list[int],
    region_seeds: list[list[int | None]],
) -> np.ndarray:
    out = grid.copy()
    row_bands = _region_bands(row_lines, grid.shape[0])
    col_bands = _region_bands(col_lines, grid.shape[1])

    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            seed_color = region_seeds[ri][ci]
            if seed_color is None:
                continue
            region = out[r0 : r1 + 1, c0 : c1 + 1]
            region[region == bg] = int(seed_color)

    vertical_colors: dict[tuple[int, int], int] = {}
    for ci, col in enumerate(col_lines):
        for ri, (r0, r1) in enumerate(row_bands):
            color = _line_color(region_seeds[ri][ci], region_seeds[ri][ci + 1], wall_color)
            vertical_colors[(ci, ri)] = color
            out[r0 : r1 + 1, col] = color

    horizontal_colors: dict[tuple[int, int], int] = {}
    for ri, row in enumerate(row_lines):
        for ci, (c0, c1) in enumerate(col_bands):
            color = _line_color(region_seeds[ri][ci], region_seeds[ri + 1][ci], wall_color)
            horizontal_colors[(ri, ci)] = color
            out[row, c0 : c1 + 1] = color

    for ri, row in enumerate(row_lines):
        for ci, col in enumerate(col_lines):
            incident = [
                vertical_colors[(ci, ri)],
                vertical_colors[(ci, ri + 1)],
                horizontal_colors[(ri, ci)],
                horizontal_colors[(ri, ci + 1)],
            ]
            out[row, col] = incident[0] if len(set(incident)) == 1 else wall_color

    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    row_line_count = rng.randint(LINE_COUNT_MIN, LINE_COUNT_MAX)
    col_line_count = rng.randint(LINE_COUNT_MIN, LINE_COUNT_MAX)

    try:
        row_lines = _sample_positions(height, row_line_count, rng)
        col_lines = _sample_positions(width, col_line_count, rng)
    except RuntimeError:
        return None

    bg_choices = [color for color in range(10) if color != WALL_COLOR]
    bg = rng.choice(bg_choices)
    seed_palette_size = rng.choice([2, 3])
    seed_palette = rng.sample([color for color in range(10) if color not in (bg, WALL_COLOR)], seed_palette_size)

    region_seeds = _build_region_seeds(row_line_count + 1, col_line_count + 1, seed_palette, rng)
    branch_counts = _segment_branch_counts(region_seeds)
    if any(branch_counts[key] == 0 for key in ("same", "different", "one", "none")):
        return None

    seeded_colors = {seed for row in region_seeds for seed in row if seed is not None}
    if len(seeded_colors) < 2:
        return None

    inp = _render_input(height, width, bg, WALL_COLOR, row_lines, col_lines, region_seeds, rng)
    expected = _expected_output(inp, bg, WALL_COLOR, row_lines, col_lines, region_seeds)
    if _most_common_color(inp) != bg:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
