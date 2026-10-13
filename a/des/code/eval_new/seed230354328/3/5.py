from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
MAX_SIZE = 30
DIVIDER_COLOR = 5
ROOM_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
LEGEND_LAYOUTS = [(2, 5), (5, 2)]
GEN_RETRIES = 500

# A small commutative but non-associative mixing algebra over five visible colors.
_MIX_INDEX = np.array(
    [
        [0, 3, 4, 2, 1],
        [3, 1, 0, 4, 2],
        [4, 0, 2, 1, 3],
        [2, 4, 1, 3, 0],
        [1, 2, 3, 0, 4],
    ],
    dtype=int,
)


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _build_mix_map(colors: list[int]) -> dict[tuple[int, int], int]:
    mix: dict[tuple[int, int], int] = {}
    for i, a in enumerate(colors):
        for j, b in enumerate(colors):
            mix[_pair_key(int(a), int(b))] = int(colors[int(_MIX_INDEX[i, j])])
    return mix


def _mix_colors(a: int, b: int, mix_map: dict[tuple[int, int], int]) -> int:
    if int(a) == int(b):
        return int(a)
    return int(mix_map[_pair_key(int(a), int(b))])


def _find_divider_color(grid: np.ndarray, bg: int) -> int:
    best: tuple[tuple[int, int, int], int] | None = None
    for color in sorted(set(map(int, np.unique(grid)))):
        if color == bg:
            continue
        mask = grid == color
        if not np.any(mask):
            continue
        r0, c0, r1, c1 = _bbox_from_mask(mask)
        sub = mask[r0 : r1 + 1, c0 : c1 + 1]
        row_hits = int(np.sum(sub.sum(axis=1) >= max(2, (sub.shape[1] + 1) // 2)))
        col_hits = int(np.sum(sub.sum(axis=0) >= max(2, (sub.shape[0] + 1) // 2)))
        score = (min(row_hits, col_hits), row_hits + col_hits, int(mask.sum()))
        if best is None or score > best[0]:
            best = (score, color)
    assert best is not None and best[0][0] >= 2, "expected a room-divider lattice color"
    return int(best[1])


def _divider_indices(
    grid: np.ndarray, bbox: tuple[int, int, int, int], divider: int
) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == divider
    width = c1 - c0 + 1
    height = r1 - r0 + 1
    row_threshold = max(2, (width + 1) // 2)
    col_threshold = max(2, (height + 1) // 2)
    row_lines = [r0 + i for i, cnt in enumerate(sub.sum(axis=1)) if int(cnt) >= row_threshold]
    col_lines = [c0 + i for i, cnt in enumerate(sub.sum(axis=0)) if int(cnt) >= col_threshold]
    assert len(row_lines) >= 3 and len(col_lines) >= 3, "expected multiple divider rows and columns"
    return row_lines, col_lines


def _extract_rooms(
    grid: np.ndarray, bg: int, divider: int
) -> tuple[list[dict], tuple[int, int, int, int], list[int], list[int]]:
    divider_mask = grid == divider
    bbox = _bbox_from_mask(divider_mask)
    row_lines, col_lines = _divider_indices(grid, bbox, divider)

    rooms: list[dict] = []
    room_by_cell: dict[tuple[int, int], dict] = {}
    room_id = 0
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            rr0 = row_lines[i] + 1
            rr1 = row_lines[i + 1] - 1
            cc0 = col_lines[j] + 1
            cc1 = col_lines[j + 1] - 1
            assert rr0 <= rr1 and cc0 <= cc1, "room interior must be non-empty"
            patch = grid[rr0 : rr1 + 1, cc0 : cc1 + 1]
            vals, cnts = np.unique(patch, return_counts=True)
            color = int(vals[cnts.argmax()])
            room = {
                "id": room_id,
                "cell": (i, j),
                "bbox": (rr0, cc0, rr1, cc1),
                "color": color,
                "windows": [],
            }
            rooms.append(room)
            room_by_cell[(i, j)] = room
            room_id += 1

    # Vertical shared borders.
    for j in range(1, len(col_lines) - 1):
        c = col_lines[j]
        for i in range(len(row_lines) - 1):
            upper = row_lines[i] + 1
            lower = row_lines[i + 1] - 1
            left_room = room_by_cell[(i, j - 1)]
            right_room = room_by_cell[(i, j)]
            for r in range(upper, lower + 1):
                if int(grid[r, c]) == divider:
                    continue
                left_room["windows"].append((r, c, int(right_room["color"])))
                right_room["windows"].append((r, c, int(left_room["color"])))

    # Horizontal shared borders.
    for i in range(1, len(row_lines) - 1):
        r = row_lines[i]
        for j in range(len(col_lines) - 1):
            left = col_lines[j] + 1
            right = col_lines[j + 1] - 1
            top_room = room_by_cell[(i - 1, j)]
            bottom_room = room_by_cell[(i, j)]
            for c in range(left, right + 1):
                if int(grid[r, c]) == divider:
                    continue
                top_room["windows"].append((r, c, int(bottom_room["color"])))
                bottom_room["windows"].append((r, c, int(top_room["color"])))

    return rooms, bbox, row_lines, col_lines


def _ordered_neighbor_colors(room: dict) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for r, c, color in sorted(room["windows"], key=lambda x: (int(x[0]), int(x[1]))):
        color = int(color)
        if color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _parse_legend(
    grid: np.ndarray, bg: int, divider: int, partition_bbox: tuple[int, int, int, int]
) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    pr0, pc0, pr1, pc1 = partition_bbox
    legend: dict[tuple[int, int], int] = {}
    for r in range(h - 1):
        for c in range(w - 1):
            block = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            if any(pr0 <= rr <= pr1 and pc0 <= cc <= pc1 for rr, cc in block):
                continue
            a = int(grid[r, c])
            b = int(grid[r, c + 1])
            x = int(grid[r + 1, c])
            y = int(grid[r + 1, c + 1])
            if a in (bg, divider) or b in (bg, divider):
                continue
            result: int | None = None
            if x not in (bg, divider) and y == bg:
                result = x
            elif y not in (bg, divider) and x == bg:
                result = y
            if result is None:
                continue
            key = _pair_key(a, b)
            if key in legend:
                assert legend[key] == result, "legend pair must map consistently"
            legend[key] = int(result)
    assert legend, "expected a mixing legend outside the room lattice"
    return legend


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each rectangular room by sequentially mixing in the neighbor colors exposed through border windows.

    Input structure:
    - A grey 1-pixel lattice partitions the grid into axis-aligned rectangular rooms whose interiors are uniformly colored.
    - Some divider cells are missing, creating 1-pixel windows on shared borders between adjacent rooms.
    - A corner legend contains many small L-shaped triples: the two top pixels are the input colors and the single pixel directly below one of them is the mixed result color.

    Rule:
    - Detect every room and collect the colors visible through its windows.
    - Sort that room's windows from top to bottom, then left to right, and keep only the first occurrence of each seen neighbor color.
    - If a room sees no neighbors, leave it unchanged; otherwise repeatedly mix its current color with those ordered neighbor colors using the legend, then repaint the whole room interior with the final color.
    - Grey divider cells, legend cells, and the window gaps themselves stay unchanged.

    Key insight: The task combines three structures at once: a room lattice, missing-divider windows that reveal adjacent rooms, and a corner legend that defines a two-color mixing function used sequentially in window order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    divider = _find_divider_color(g, bg)
    rooms, partition_bbox, _, _ = _extract_rooms(g, bg, divider)
    mix_map = _parse_legend(g, bg, divider, partition_bbox)

    out = g.copy()
    for room in rooms:
        new_color = int(room["color"])
        for neighbor_color in _ordered_neighbor_colors(room):
            new_color = _mix_colors(new_color, neighbor_color, mix_map)
        rr0, cc0, rr1, cc1 = room["bbox"]
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = new_color
    return out


# --------------- generator ---------------

def _room_id(n_cols: int, i: int, j: int) -> int:
    return i * n_cols + j


def _room_neighbors(n_rows: int, n_cols: int) -> dict[int, list[int]]:
    neighbors: dict[int, list[int]] = {}
    for i in range(n_rows):
        for j in range(n_cols):
            rid = _room_id(n_cols, i, j)
            cur: list[int] = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n_rows and 0 <= nj < n_cols:
                    cur.append(_room_id(n_cols, ni, nj))
            neighbors[rid] = cur
    return neighbors


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _legend_patch(entries: list[tuple[int, int, int, bool]], rows: int, cols: int) -> np.ndarray:
    patch = np.full((rows * 3 - 1, cols * 3 - 1), BG, dtype=int)
    for idx, (a, b, result, put_right) in enumerate(entries):
        rr = (idx // cols) * 3
        cc = (idx % cols) * 3
        patch[rr, cc] = int(a)
        patch[rr, cc + 1] = int(b)
        if put_right:
            patch[rr + 1, cc + 1] = int(result)
        else:
            patch[rr + 1, cc] = int(result)
    return patch


def _choose_room_sizes(count: int, max_partition: int) -> list[int] | None:
    for _ in range(80):
        sizes = [random.randint(2, 4) for _ in range(count)]
        if sum(sizes) + count + 1 <= max_partition:
            return sizes
    return None


def _pick_layout() -> tuple[int, int, int, int, int, int]:
    legend_rows, legend_cols = random.choice(LEGEND_LAYOUTS)
    legend_h = legend_rows * 3 - 1
    legend_w = legend_cols * 3 - 1
    max_partition_h = MAX_SIZE - legend_h - 5
    max_partition_w = MAX_SIZE - legend_w - 5
    max_room_rows = min(4, (max_partition_h - 1) // 3)
    max_room_cols = min(4, (max_partition_w - 1) // 3)
    if max_room_rows < 2 or max_room_cols < 2:
        raise AssertionError("legend layout leaves no room for a valid partition")

    n_rows = random.randint(2, max_room_rows)
    n_cols = random.randint(2, max_room_cols)
    row_sizes = _choose_room_sizes(n_rows, max_partition_h)
    col_sizes = _choose_room_sizes(n_cols, max_partition_w)
    assert row_sizes is not None and col_sizes is not None

    part_h = sum(row_sizes) + n_rows + 1
    part_w = sum(col_sizes) + n_cols + 1
    grid_h = random.randint(max(18, part_h + legend_h + 5), MAX_SIZE)
    grid_w = random.randint(max(18, part_w + legend_w + 5), MAX_SIZE)
    return n_rows, n_cols, grid_h, grid_w, legend_h, legend_w, row_sizes, col_sizes


def _place_partition(
    grid_h: int, grid_w: int, part_h: int, part_w: int, legend_h: int, legend_w: int, corner: str
) -> tuple[int, int]:
    if corner == "tl":
        r0_min, r0_max = legend_h + 2, grid_h - part_h - 2
        c0_min, c0_max = legend_w + 2, grid_w - part_w - 2
    elif corner == "tr":
        r0_min, r0_max = legend_h + 2, grid_h - part_h - 2
        c0_min, c0_max = 1, grid_w - part_w - legend_w - 3
    elif corner == "bl":
        r0_min, r0_max = 1, grid_h - part_h - legend_h - 3
        c0_min, c0_max = legend_w + 2, grid_w - part_w - 2
    else:
        r0_min, r0_max = 1, grid_h - part_h - legend_h - 3
        c0_min, c0_max = 1, grid_w - part_w - legend_w - 3
    if r0_min > r0_max or c0_min > c0_max:
        raise AssertionError("partition does not fit opposite the chosen legend corner")
    return random.randint(r0_min, r0_max), random.randint(c0_min, c0_max)


def _legend_origin(grid_h: int, grid_w: int, legend_h: int, legend_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 1, 1
    if corner == "tr":
        return 1, grid_w - legend_w - 1
    if corner == "bl":
        return grid_h - legend_h - 1, 1
    return grid_h - legend_h - 1, grid_w - legend_w - 1


def _draw_partition(
    grid: np.ndarray, part_r0: int, part_c0: int, row_sizes: list[int], col_sizes: list[int]
) -> tuple[list[int], list[int]]:
    row_lines = [part_r0]
    cur = part_r0
    for size in row_sizes:
        cur += size + 1
        row_lines.append(cur)

    col_lines = [part_c0]
    cur = part_c0
    for size in col_sizes:
        cur += size + 1
        col_lines.append(cur)

    grid[row_lines, part_c0 : col_lines[-1] + 1] = DIVIDER_COLOR
    grid[part_r0 : row_lines[-1] + 1, col_lines] = DIVIDER_COLOR
    return row_lines, col_lines


def _choose_special_rooms(n_rows: int, n_cols: int) -> tuple[int, int, int, int]:
    neighbors = _room_neighbors(n_rows, n_cols)
    focus_candidates = [rid for rid, neigh in neighbors.items() if len(neigh) >= 2]
    focus = random.choice(focus_candidates)
    n1, n2 = random.sample(neighbors[focus], 2)
    all_rooms = list(range(n_rows * n_cols))
    others = [rid for rid in all_rooms if rid not in {focus, n1, n2}]
    assert others, "need at least one isolated room"
    isolated = random.choice(others)
    return focus, n1, n2, isolated


def _assign_room_colors(total_rooms: int, focus: int, n1: int, n2: int) -> tuple[list[int], list[int], dict[tuple[int, int], int]]:
    palette = random.sample(ROOM_COLOR_POOL, 5)
    mix_map = _build_mix_map(palette)
    color_indices = [random.randrange(5) for _ in range(total_rooms)]
    focus_idx, n1_idx, n2_idx = random.sample(range(5), 3)
    color_indices[focus] = focus_idx
    color_indices[n1] = n1_idx
    color_indices[n2] = n2_idx
    remaining = [rid for rid in range(total_rooms) if rid not in {focus, n1, n2}]
    unused = [idx for idx in range(5) if idx not in {focus_idx, n1_idx, n2_idx}]
    if remaining and unused:
        color_indices[remaining[0]] = unused[0]
    return palette, color_indices, mix_map


def _paint_rooms(
    grid: np.ndarray,
    n_rows: int,
    n_cols: int,
    row_lines: list[int],
    col_lines: list[int],
    palette: list[int],
    color_indices: list[int],
) -> None:
    for i in range(n_rows):
        for j in range(n_cols):
            rid = _room_id(n_cols, i, j)
            color = int(palette[color_indices[rid]])
            rr0 = row_lines[i] + 1
            rr1 = row_lines[i + 1] - 1
            cc0 = col_lines[j] + 1
            cc1 = col_lines[j + 1] - 1
            grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = color


def _carve_window_for_edge(
    grid: np.ndarray,
    edge: tuple[int, int],
    n_rows: int,
    n_cols: int,
    row_lines: list[int],
    col_lines: list[int],
) -> None:
    a, b = edge
    ai, aj = divmod(a, n_cols)
    bi, bj = divmod(b, n_cols)
    if ai == bi:
        left_j = min(aj, bj)
        c = col_lines[left_j + 1]
        r0 = row_lines[ai] + 1
        r1 = row_lines[ai + 1] - 1
        grid[random.randint(r0, r1), c] = BG
    else:
        top_i = min(ai, bi)
        r = row_lines[top_i + 1]
        c0 = col_lines[aj] + 1
        c1 = col_lines[aj + 1] - 1
        grid[r, random.randint(c0, c1)] = BG


def _draw_legend(
    grid: np.ndarray,
    palette: list[int],
    mix_map: dict[tuple[int, int], int],
    corner: str,
    rows: int,
    cols: int,
) -> None:
    entries: list[tuple[int, int, int, bool]] = []
    pairs = list(itertools.combinations(palette, 2))
    random.shuffle(pairs)
    for a, b in pairs:
        left, right = (a, b) if random.random() < 0.5 else (b, a)
        entries.append((int(left), int(right), int(mix_map[_pair_key(a, b)]), random.random() < 0.5))
    patch = _legend_patch(entries, rows, cols)
    lr0, lc0 = _legend_origin(grid.shape[0], grid.shape[1], patch.shape[0], patch.shape[1], corner)
    grid[lr0 : lr0 + patch.shape[0], lc0 : lc0 + patch.shape[1]] = patch


def _branch_signature(grid: np.ndarray) -> list[int]:
    bg = _most_common_color(grid)
    divider = _find_divider_color(grid, bg)
    rooms, bbox, _, _ = _extract_rooms(grid, bg, divider)
    legend = _parse_legend(grid, bg, divider, bbox)
    assert len(legend) == 10, "generator should expose the full five-color mixing table"
    return [len(_ordered_neighbor_colors(room)) for room in rooms]


def _make_candidate() -> np.ndarray:
    while True:
        legend_rows, legend_cols = random.choice(LEGEND_LAYOUTS)
        legend_h = legend_rows * 3 - 1
        legend_w = legend_cols * 3 - 1
        max_partition_h = MAX_SIZE - legend_h - 5
        max_partition_w = MAX_SIZE - legend_w - 5
        max_room_rows = min(4, (max_partition_h - 1) // 3)
        max_room_cols = min(4, (max_partition_w - 1) // 3)
        if max_room_rows < 2 or max_room_cols < 2:
            continue

        n_rows = random.randint(2, max_room_rows)
        n_cols = random.randint(2, max_room_cols)
        row_sizes = _choose_room_sizes(n_rows, max_partition_h)
        col_sizes = _choose_room_sizes(n_cols, max_partition_w)
        if row_sizes is None or col_sizes is None:
            continue

        part_h = sum(row_sizes) + n_rows + 1
        part_w = sum(col_sizes) + n_cols + 1
        min_h = max(18, part_h + legend_h + 5)
        min_w = max(18, part_w + legend_w + 5)
        if min_h > MAX_SIZE or min_w > MAX_SIZE:
            continue

        grid_h = random.randint(min_h, MAX_SIZE)
        grid_w = random.randint(min_w, MAX_SIZE)
        grid = np.full((grid_h, grid_w), BG, dtype=int)

        corner = random.choice(["tl", "tr", "bl", "br"])
        try:
            part_r0, part_c0 = _place_partition(grid_h, grid_w, part_h, part_w, legend_h, legend_w, corner)
        except AssertionError:
            continue

        row_lines, col_lines = _draw_partition(grid, part_r0, part_c0, row_sizes, col_sizes)
        focus, n1, n2, isolated = _choose_special_rooms(n_rows, n_cols)
        palette, color_indices, mix_map = _assign_room_colors(n_rows * n_cols, focus, n1, n2)
        _paint_rooms(grid, n_rows, n_cols, row_lines, col_lines, palette, color_indices)

        open_edges = {_edge_key(focus, n1), _edge_key(focus, n2)}

        # Add a small amount of extra structure without disturbing the guaranteed branch coverage.
        protected = {isolated, n1, n2}
        all_edges = []
        for rid, neigh in _room_neighbors(n_rows, n_cols).items():
            for other in neigh:
                if rid < other:
                    all_edges.append((rid, other))
        random.shuffle(all_edges)
        extra_budget = random.randint(0, 2)
        for edge in all_edges:
            if extra_budget == 0:
                break
            a, b = edge
            if a in protected or b in protected:
                continue
            if edge in open_edges:
                continue
            if random.random() < 0.45:
                open_edges.add(edge)
                extra_budget -= 1

        for edge in sorted(open_edges):
            _carve_window_for_edge(grid, edge, n_rows, n_cols, row_lines, col_lines)

        # Force the isolated room to have no windows.
        iso_i, iso_j = divmod(isolated, n_cols)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = iso_i + di, iso_j + dj
            if not (0 <= ni < n_rows and 0 <= nj < n_cols):
                continue
            rid2 = _room_id(n_cols, ni, nj)
            edge = _edge_key(isolated, rid2)
            if edge not in open_edges:
                continue
            raise AssertionError("isolated room accidentally received a window")

        _draw_legend(grid, palette, mix_map, corner, legend_rows, legend_cols)

        if _most_common_color(grid) != BG:
            continue

        counts = _branch_signature(grid)
        if 0 not in counts or 1 not in counts or not any(cnt >= 2 for cnt in counts):
            continue

        out = main(grid)
        if out.shape != grid.shape:
            continue
        if not np.any(out != grid):
            continue
        return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _make_candidate()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid room-mixing puzzle")
