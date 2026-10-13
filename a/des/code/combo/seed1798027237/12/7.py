import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PIPE_COLOR = 5

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (20, 30)
NETWORK_H_RANGE = (10, 16)
NETWORK_W_RANGE = (10, 16)
TARGET_INTERSECTIONS = [1, 2, 3, 4]
TARGET_INTERSECTION_WEIGHTS = [1, 3, 3, 2]
END_SEGMENT_RANGE = (1, 5)
CHILD_SEGMENT_RANGE = (2, 5)
LEGEND_EXTRA_MAX = 2
MAX_NETWORK_ATTEMPTS = 200
MAX_GENERATE_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


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
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(points, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _pipe_neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    return [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if mask[nr, nc]]


def _intersection_points(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = []
    for r, c in np.argwhere(mask):
        if len(_pipe_neighbors(mask, int(r), int(c))) >= 3:
            pts.append((int(r), int(c)))
    return pts


def _find_pipe_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    scored: list[tuple[int, int, int, int]] = []
    for color in colors:
        mask = grid == color
        max_deg = 0
        branch_pixels = 0
        for r, c in np.argwhere(mask):
            deg = len(_pipe_neighbors(mask, int(r), int(c)))
            max_deg = max(max_deg, deg)
            if deg >= 3:
                branch_pixels += 1
        scored.append((max_deg, branch_pixels, int(mask.sum()), color))
    branching = [item for item in scored if item[0] >= 3]
    if branching:
        return max(branching)[-1]
    return max(scored, key=lambda item: (item[2], item[0]))[-1]


def _trace_segment_length(
    mask: np.ndarray,
    intersections: set[tuple[int, int]],
    start: tuple[int, int],
    nxt: tuple[int, int],
) -> int:
    prev = start
    cur = nxt
    length = 1
    while True:
        if cur in intersections:
            return length
        next_steps = [pt for pt in _pipe_neighbors(mask, cur[0], cur[1]) if pt != prev]
        if len(next_steps) != 1:
            return length
        prev, cur = cur, next_steps[0]
        length += 1


def _intersection_sums(mask: np.ndarray) -> dict[tuple[int, int], int]:
    intersections = _intersection_points(mask)
    i_set = set(intersections)
    sums: dict[tuple[int, int], int] = {}
    for point in intersections:
        total = 0
        for nxt in _pipe_neighbors(mask, point[0], point[1]):
            total += _trace_segment_length(mask, i_set, point, nxt)
        sums[point] = total
    return sums


def _legend_map(grid: np.ndarray, bg: int, pipe_color: int) -> dict[int, int]:
    legend: dict[int, int] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in {bg, pipe_color}:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r0 == r1 and len(comp) == c1 - c0 + 1:
                legend[len(comp)] = color
    return legend


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor pipe intersections using the sum of adjacent segment lengths as a legend lookup.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    pipe_color = _find_pipe_color(g, bg)
    pipe_mask = g == pipe_color
    legend = _legend_map(g, bg, pipe_color)

    out = g.copy()
    for (r, c), total in _intersection_sums(pipe_mask).items():
        if total in legend:
            out[r, c] = legend[total]
    return out


def _is_straight_degree2(mask: np.ndarray, r: int, c: int) -> bool:
    nbs = _pipe_neighbors(mask, r, c)
    if len(nbs) != 2:
        return True
    (r1, c1), (r2, c2) = nbs
    return r1 == r2 or c1 == c2


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _path_cells(
    start: tuple[int, int], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    r0, c0 = start
    dr, dc = direction
    return [(r0 + dr * step, c0 + dc * step) for step in range(1, length + 1)]


def _can_carve(
    mask: np.ndarray, start: tuple[int, int], direction: tuple[int, int], length: int
) -> bool:
    h, w = mask.shape
    prev = start
    for cell in _path_cells(start, direction, length):
        r, c = cell
        if not (0 <= r < h and 0 <= c < w):
            return False
        if mask[r, c]:
            return False
        occupied = {pt for pt in _neighbors4(r, c, h, w) if mask[pt]}
        if occupied - {prev}:
            return False
        prev = cell
    return True


def _carve(
    mask: np.ndarray, start: tuple[int, int], direction: tuple[int, int], length: int
) -> tuple[int, int]:
    end = start
    for r, c in _path_cells(start, direction, length):
        mask[r, c] = True
        end = (r, c)
    return end


def _future_branch_capacity(mask: np.ndarray, node: tuple[int, int], back: tuple[int, int]) -> int:
    count = 0
    for direction in DIRS4:
        if direction == back:
            continue
        for length in range(END_SEGMENT_RANGE[0], END_SEGMENT_RANGE[1] + 1):
            if _can_carve(mask, node, direction, length):
                count += 1
                break
    return count


def _branch_lengths(
    mask: np.ndarray, node: tuple[int, int], direction: tuple[int, int], kind: str
) -> list[int]:
    if kind == "end":
        low, high = END_SEGMENT_RANGE
    else:
        low, high = CHILD_SEGMENT_RANGE
    good = []
    for length in range(low, high + 1):
        if not _can_carve(mask, node, direction, length):
            continue
        if kind == "child":
            temp = mask.copy()
            end = _carve(temp, node, direction, length)
            if _future_branch_capacity(temp, end, (-direction[0], -direction[1])) < 2:
                continue
        good.append(length)
    return good


def _valid_network(mask: np.ndarray, target_intersections: int) -> bool:
    if len(_components(mask)) != 1:
        return False
    intersections = _intersection_points(mask)
    if len(intersections) != target_intersections:
        return False
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        deg = len(_pipe_neighbors(mask, rr, cc))
        if deg == 0:
            return False
        if deg == 2 and not _is_straight_degree2(mask, rr, cc):
            return False
    return True


def _generate_network_mask() -> np.ndarray | None:
    for _ in range(MAX_NETWORK_ATTEMPTS):
        h = random.randint(*NETWORK_H_RANGE)
        w = random.randint(*NETWORK_W_RANGE)
        target = random.choices(
            TARGET_INTERSECTIONS, weights=TARGET_INTERSECTION_WEIGHTS, k=1
        )[0]
        mask = np.zeros((h, w), dtype=bool)
        root = (random.randint(2, h - 3), random.randint(2, w - 3))
        mask[root] = True
        nodes = [root]
        index = 0
        ok = True

        while index < len(nodes):
            node = nodes[index]
            index += 1
            current_deg = len(_pipe_neighbors(mask, node[0], node[1]))
            free_dirs = []
            for direction in DIRS4:
                nr, nc = node[0] + direction[0], node[1] + direction[1]
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    free_dirs.append(direction)
            possible_degrees = [
                degree
                for degree in (3, 4)
                if current_deg <= degree <= current_deg + len(free_dirs)
            ]
            if not possible_degrees:
                ok = False
                break
            target_degree = random.choice(possible_degrees)
            extra = target_degree - current_deg
            remaining_new = target - len(nodes)
            future_nodes = len(nodes) - index
            min_child = max(0, remaining_new - future_nodes * 3)
            if remaining_new > 0 and future_nodes == 0:
                min_child = max(min_child, 1)
            max_child = min(extra, remaining_new)
            if min_child > max_child:
                ok = False
                break
            child_count = random.randint(min_child, max_child)
            branch_kinds = ["child"] * child_count + ["end"] * (extra - child_count)
            random.shuffle(branch_kinds)
            random.shuffle(free_dirs)

            for kind in branch_kinds:
                placed = False
                random.shuffle(free_dirs)
                for direction in list(free_dirs):
                    lengths = _branch_lengths(mask, node, direction, kind)
                    if not lengths:
                        continue
                    end = _carve(mask, node, direction, random.choice(lengths))
                    free_dirs.remove(direction)
                    if kind == "child":
                        nodes.append(end)
                    placed = True
                    break
                if not placed:
                    ok = False
                    break

            if not ok:
                break

        if not ok or len(nodes) != target or not _valid_network(mask, target):
            continue
        return _trim_mask(mask)
    return None


def _choose_legend_lengths(sum_values: list[int]) -> tuple[list[int], set[int]]:
    unique_sums = sorted(set(sum_values))
    if len(unique_sums) == 1 or random.random() < 0.45:
        matched = set(unique_sums)
    else:
        matched = set(random.sample(unique_sums, random.randint(1, len(unique_sums) - 1)))

    pool_limit = max(max(unique_sums) + 4, 8)
    pool = [value for value in range(2, pool_limit + 1) if value not in set(unique_sums)]
    min_extra = max(0, 2 - len(matched))
    max_extra = min(LEGEND_EXTRA_MAX, len(pool))
    if min_extra > max_extra:
        return sorted(matched), matched
    extra_count = random.randint(min_extra, max_extra)
    extras = set(random.sample(pool, extra_count))
    return sorted(matched | extras), matched


def _legend_bbox(h: int, w: int, legend_h: int, legend_w: int, corner: str) -> tuple[int, int, int, int]:
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - legend_w
    elif corner == "bl":
        r0, c0 = h - legend_h, 0
    else:
        r0, c0 = h - legend_h, w - legend_w
    return r0, c0, r0 + legend_h - 1, c0 + legend_w - 1


def _expanded_overlap(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    br0 = max(0, br0 - pad)
    bc0 = max(0, bc0 - pad)
    br1 += pad
    bc1 += pad
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _paint_legend(
    grid: np.ndarray,
    lengths: list[int],
    color_by_length: dict[int, int],
    corner: str,
) -> tuple[int, int, int, int]:
    legend_h = 2 * len(lengths) - 1
    legend_w = max(lengths)
    h, w = grid.shape
    r0, c0, r1, c1 = _legend_bbox(h, w, legend_h, legend_w, corner)
    for idx, length in enumerate(lengths):
        row = r0 + 2 * idx
        grid[row, c0 : c0 + length] = color_by_length[length]
    return r0, c0, r1, c1


def _place_network(
    grid: np.ndarray, network: np.ndarray, legend_box: tuple[int, int, int, int]
) -> bool:
    h, w = grid.shape
    nh, nw = network.shape
    candidates = []
    for r0 in range(h - nh + 1):
        for c0 in range(w - nw + 1):
            box = (r0, c0, r0 + nh - 1, c0 + nw - 1)
            if not _expanded_overlap(box, legend_box, pad=1):
                candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    sub = grid[r0 : r0 + nh, c0 : c0 + nw]
    sub[network] = PIPE_COLOR
    return True


def _gen_attempt() -> np.ndarray | None:
    network = _generate_network_mask()
    if network is None:
        return None
    sums = _intersection_sums(network)
    if not sums:
        return None

    legend_lengths, matched_lengths = _choose_legend_lengths(list(sums.values()))
    legend_h = 2 * len(legend_lengths) - 1
    legend_w = max(legend_lengths)

    h = max(
        random.randint(*GRID_H_RANGE),
        network.shape[0] + legend_h + 4,
        legend_h + 4,
    )
    w = max(
        random.randint(*GRID_W_RANGE),
        network.shape[1] + legend_w + 4,
        legend_w + 4,
    )

    bg = random.choice([color for color in range(10) if color != PIPE_COLOR])
    legend_colors = [color for color in range(10) if color not in {bg, PIPE_COLOR}]
    random.shuffle(legend_colors)
    if len(legend_colors) < len(legend_lengths):
        return None
    color_by_length = {
        length: legend_colors[idx] for idx, length in enumerate(legend_lengths)
    }

    grid = np.full((h, w), bg, dtype=int)
    corner = random.choice(["tl", "tr", "bl", "br"])
    legend_box = _paint_legend(grid, legend_lengths, color_by_length, corner)
    if not _place_network(grid, network, legend_box):
        return None

    if _get_bg(grid) != bg:
        return None

    expected = grid.copy()
    changed = 0
    stayed_grey = 0
    pipe_mask = grid == PIPE_COLOR
    for (r, c), total in _intersection_sums(pipe_mask).items():
        if total in color_by_length:
            expected[r, c] = color_by_length[total]
            changed += 1
        else:
            stayed_grey += 1

    if changed == 0:
        return None
    if matched_lengths != set(sums.values()) and stayed_grey == 0:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid pipe-legend input")
