import random
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend_and_board(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    grey_components = _components(grid == GREY)
    assert len(grey_components) >= 2, "expected a legend frame and a board frame"
    info = []
    for comp in grey_components:
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        info.append(
            {
                "bbox": box,
                "bbox_area": (r1 - r0 + 1) * (c1 - c0 + 1),
                "pixel_count": len(comp),
            }
        )
    board_idx = max(range(len(info)), key=lambda i: (info[i]["bbox_area"], info[i]["pixel_count"]))
    board_bbox = info[board_idx]["bbox"]
    legend_candidates = [item for i, item in enumerate(info) if i != board_idx]
    legend_info = min(legend_candidates, key=lambda item: (item["bbox"][1], item["bbox_area"]))
    return legend_info["bbox"], board_bbox


def _read_legend(legend: np.ndarray) -> tuple[list[int], dict[int, set[int]]]:
    inner = legend[1:-1, 1:-1]
    colors: set[int] = set()
    compat: dict[int, set[int]] = {}
    h, w = inner.shape
    for r in range(h):
        for c in range(w - 1):
            a = int(inner[r, c])
            b = int(inner[r, c + 1])
            if a in (BLACK, GREY) or b in (BLACK, GREY) or a == b:
                continue
            colors.add(a)
            colors.add(b)
            compat.setdefault(a, set()).add(b)
            compat.setdefault(b, set()).add(a)
    ordered = sorted(colors)
    assert ordered, "legend must contain at least one allowed color pair"
    for color in ordered:
        compat.setdefault(color, set())
    return ordered, compat


def _extract_regions(board: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[int]]:
    region_mask = board != GREY
    h, w = board.shape
    labels = np.full((h, w), -1, dtype=int)
    regions: list[list[tuple[int, int]]] = []
    region_colors: list[int] = []
    for comp in _components(region_mask):
        rid = len(regions)
        color_values = {int(board[r, c]) for r, c in comp}
        assert len(color_values) == 1, "each region must be monochromatic"
        region_colors.append(color_values.pop())
        regions.append(comp)
        for r, c in comp:
            labels[r, c] = rid
    return labels, regions, region_colors


def _region_adjacency(labels: np.ndarray, board: np.ndarray) -> list[set[int]]:
    h, w = board.shape
    n_regions = int(labels.max()) + 1
    adjacency = [set() for _ in range(n_regions)]
    for r in range(h):
        for c in range(w):
            if int(board[r, c]) != GREY:
                continue
            if 0 < c < w - 1:
                left = int(labels[r, c - 1])
                right = int(labels[r, c + 1])
                if left != -1 and right != -1 and left != right:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
            if 0 < r < h - 1:
                up = int(labels[r - 1, c])
                down = int(labels[r + 1, c])
                if up != -1 and down != -1 and up != down:
                    adjacency[up].add(down)
                    adjacency[down].add(up)
    return adjacency


def _solve_board(board: np.ndarray, colors: list[int], compat: dict[int, set[int]]) -> np.ndarray:
    labels, regions, region_colors = _extract_regions(board)
    adjacency = _region_adjacency(labels, board)
    out = board.copy()

    progress = True
    while progress:
        progress = False
        for rid, current in enumerate(region_colors):
            if current != BLACK:
                continue
            neighbor_colors = {region_colors[nid] for nid in adjacency[rid] if region_colors[nid] != BLACK}
            if not neighbor_colors:
                continue
            candidates = []
            for color in colors:
                if all(neighbor in compat.get(color, set()) for neighbor in neighbor_colors):
                    candidates.append(color)
            if len(candidates) != 1:
                continue
            fill = candidates[0]
            region_colors[rid] = fill
            for r, c in regions[rid]:
                out[r, c] = fill
            progress = True

    assert BLACK not in region_colors, "every uncolored region should be solvable"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_bbox, board_bbox = _find_legend_and_board(g)
    lr0, lc0, lr1, lc1 = legend_bbox
    br0, bc0, br1, bc1 = board_bbox
    legend = g[lr0 : lr1 + 1, lc0 : lc1 + 1]
    board = g[br0 : br1 + 1, bc0 : bc1 + 1]
    colors, compat = _read_legend(legend)
    return _solve_board(board, colors, compat)


def _random_partition(rng: random.Random, rows: int, cols: int, n_regions: int) -> np.ndarray:
    grid = np.full((rows, cols), -1, dtype=int)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    seeds = rng.sample(cells, n_regions)
    for rid, (r, c) in enumerate(seeds):
        grid[r, c] = rid

    unassigned = {cell for cell in cells if cell not in seeds}
    while unassigned:
        frontier = []
        for r, c in unassigned:
            neighbor_ids = {
                int(grid[nr, nc])
                for nr, nc in _neighbors4(r, c, rows, cols)
                if int(grid[nr, nc]) != -1
            }
            if neighbor_ids:
                frontier.append(((r, c), list(neighbor_ids)))
        assert frontier, "partition growth must have an assignable frontier"
        (r, c), options = rng.choice(frontier)
        grid[r, c] = rng.choice(options)
        unassigned.remove((r, c))
    return grid


def _adjacency_from_region_map(region_map: np.ndarray) -> list[set[int]]:
    n_regions = int(region_map.max()) + 1
    adjacency = [set() for _ in range(n_regions)]
    rows, cols = region_map.shape
    for r in range(rows):
        for c in range(cols):
            cur = int(region_map[r, c])
            if r + 1 < rows:
                other = int(region_map[r + 1, c])
                if cur != other:
                    adjacency[cur].add(other)
                    adjacency[other].add(cur)
            if c + 1 < cols:
                other = int(region_map[r, c + 1])
                if cur != other:
                    adjacency[cur].add(other)
                    adjacency[other].add(cur)
    return adjacency


def _proper_coloring(rng: random.Random, adjacency: list[set[int]], colors: list[int]) -> list[int] | None:
    n_regions = len(adjacency)
    order = sorted(range(n_regions), key=lambda rid: (-len(adjacency[rid]), rng.random()))
    assignment = [-1] * n_regions
    use_count = {color: 0 for color in colors}

    def backtrack(idx: int) -> bool:
        unused = [color for color in colors if use_count[color] == 0]
        if n_regions - idx < len(unused):
            return False
        if idx == n_regions:
            return all(use_count[color] > 0 for color in colors)

        rid = order[idx]
        blocked = {assignment[nid] for nid in adjacency[rid] if assignment[nid] != -1}
        options = [color for color in colors if color not in blocked]
        options.sort(key=lambda color: (use_count[color] > 0, use_count[color], rng.random()))

        for color in options:
            assignment[rid] = color
            use_count[color] += 1
            if backtrack(idx + 1):
                return True
            use_count[color] -= 1
            assignment[rid] = -1
        return False

    if backtrack(0):
        return assignment
    return None


def _compatibility_from_assignment(adjacency: list[set[int]], assignment: list[int], colors: list[int]) -> tuple[set[tuple[int, int]], dict[int, set[int]]]:
    pairs: set[tuple[int, int]] = set()
    compat = {color: set() for color in colors}
    for rid, neighbors in enumerate(adjacency):
        for nid in neighbors:
            if nid < rid:
                continue
            a = int(assignment[rid])
            b = int(assignment[nid])
            if a == b:
                continue
            pair = tuple(sorted((a, b)))
            pairs.add(pair)
            compat[a].add(b)
            compat[b].add(a)
    return pairs, compat


def _unique_blankable_regions(adjacency: list[set[int]], assignment: list[int], colors: list[int], compat: dict[int, set[int]]) -> list[int]:
    blankable = []
    for rid, true_color in enumerate(assignment):
        neighbor_colors = {assignment[nid] for nid in adjacency[rid]}
        if not neighbor_colors:
            continue
        candidates = [
            color
            for color in colors
            if all(neighbor in compat.get(color, set()) for neighbor in neighbor_colors)
        ]
        if candidates == [true_color]:
            blankable.append(rid)
    return blankable


def _render_board(region_map: np.ndarray, assignment: list[int], cell_h: int, cell_w: int) -> np.ndarray:
    rows, cols = region_map.shape
    height = rows * (cell_h + 1) + 1
    width = cols * (cell_w + 1) + 1
    board = np.full((height, width), GREY, dtype=int)

    for r in range(1, height - 1):
        raw_r = r - 1
        on_row_boundary = raw_r % (cell_h + 1) == cell_h
        coarse_r = raw_r // (cell_h + 1)
        for c in range(1, width - 1):
            raw_c = c - 1
            on_col_boundary = raw_c % (cell_w + 1) == cell_w
            coarse_c = raw_c // (cell_w + 1)

            if not on_row_boundary and not on_col_boundary:
                board[r, c] = assignment[int(region_map[coarse_r, coarse_c])]
                continue

            if on_row_boundary and not on_col_boundary:
                top = int(region_map[coarse_r, coarse_c])
                bottom = int(region_map[coarse_r + 1, coarse_c])
                if top == bottom:
                    board[r, c] = assignment[top]
                continue

            if not on_row_boundary and on_col_boundary:
                left = int(region_map[coarse_r, coarse_c])
                right = int(region_map[coarse_r, coarse_c + 1])
                if left == right:
                    board[r, c] = assignment[left]
                continue

            ids = {
                int(region_map[coarse_r, coarse_c]),
                int(region_map[coarse_r + 1, coarse_c]),
                int(region_map[coarse_r, coarse_c + 1]),
                int(region_map[coarse_r + 1, coarse_c + 1]),
            }
            if len(ids) == 1:
                only = next(iter(ids))
                board[r, c] = assignment[only]
    return board


def _render_legend(rng: random.Random, pairs: set[tuple[int, int]]) -> np.ndarray:
    ordered_pairs = list(pairs)
    rng.shuffle(ordered_pairs)
    pad_left = rng.randint(0, 1)
    pad_right = rng.randint(0, 1)
    pad_top = rng.randint(0, 1)
    pad_bottom = rng.randint(0, 1)
    use_spacers = rng.choice([False, True])

    step = 2 if use_spacers else 1
    inner_h = pad_top + pad_bottom + 1 + step * (len(ordered_pairs) - 1)
    inner_w = pad_left + pad_right + 2

    legend = np.full((inner_h + 2, inner_w + 2), GREY, dtype=int)
    legend[1:-1, 1:-1] = BLACK
    for idx, (a, b) in enumerate(ordered_pairs):
        rr = 1 + pad_top + idx * step
        cc = 1 + pad_left
        legend[rr, cc] = a
        legend[rr, cc + 1] = b
    return legend


def _compose_input(rng: random.Random, legend: np.ndarray, board: np.ndarray) -> np.ndarray:
    legend_h, legend_w = legend.shape
    board_h, board_w = board.shape
    legend_top = rng.randint(1, 3)
    board_top = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    gap = rng.randint(2, 4)
    right_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)

    height = max(legend_top + legend_h, board_top + board_h) + bottom_margin
    width = left_margin + legend_w + gap + board_w + right_margin
    grid = np.full((height, width), BLACK, dtype=int)

    grid[legend_top : legend_top + legend_h, left_margin : left_margin + legend_w] = legend
    board_left = left_margin + legend_w + gap
    grid[board_top : board_top + board_h, board_left : board_left + board_w] = board
    return grid


def _choose_blank_regions(
    rng: random.Random, blankable: list[int], adjacency: list[set[int]]
) -> list[int] | None:
    by_size: dict[int, list[tuple[int, ...]]] = {1: [], 2: [], 3: []}
    limit = min(3, len(blankable))
    for size in range(1, limit + 1):
        for combo in combinations(blankable, size):
            if all(b not in adjacency[a] for a, b in combinations(combo, 2)):
                by_size[size].append(combo)
    feasible_sizes = [size for size, combos in by_size.items() if combos]
    if not feasible_sizes:
        return None
    chosen_size = rng.choice(feasible_sizes)
    return list(rng.choice(by_size[chosen_size]))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    coarse_rows = rng.randint(3, 5)
    coarse_cols = rng.randint(4, 7)
    n_cells = coarse_rows * coarse_cols
    max_regions = min(12, n_cells - 1)
    min_regions = min(max_regions, max(5, n_cells // 4))
    n_regions = rng.randint(min_regions, max_regions)
    region_map = _random_partition(rng, coarse_rows, coarse_cols, n_regions)
    adjacency = _adjacency_from_region_map(region_map)

    n_colors = rng.randint(3, min(5, n_regions))
    colors = rng.sample(PALETTE, n_colors)
    assignment = _proper_coloring(rng, adjacency, colors)
    if assignment is None:
        return None

    allowed_pairs, compat = _compatibility_from_assignment(adjacency, assignment, colors)
    if not allowed_pairs:
        return None

    blankable = _unique_blankable_regions(adjacency, assignment, colors, compat)
    blanks = _choose_blank_regions(rng, blankable, adjacency)
    if not blanks:
        return None

    cell_h = rng.randint(2, 3)
    cell_w = rng.randint(2, 3)
    solved_board = _render_board(region_map, assignment, cell_h, cell_w)
    input_assignment = assignment[:]
    for rid in blanks:
        input_assignment[rid] = BLACK
    input_board = _render_board(region_map, input_assignment, cell_h, cell_w)

    legend = _render_legend(rng, allowed_pairs)
    grid = _compose_input(rng, legend, input_board)
    if not np.array_equal(main(grid), solved_board):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry limit")
