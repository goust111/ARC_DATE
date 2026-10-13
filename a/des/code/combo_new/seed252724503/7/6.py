import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATION_ATTEMPTS = 400

Coord = Tuple[int, int]


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[Coord]:
    if r > 0:
        yield (r - 1, c)
    if r + 1 < h:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < w:
        yield (r, c + 1)


def _ordered_pair(a: Coord, b: Coord) -> Tuple[Coord, Coord]:
    return (a, b) if a <= b else (b, a)


def _segment(a: Coord, b: Coord) -> List[Coord]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment endpoints must share a row or column")


def _build_l_paths(start: Coord, end: Coord) -> Tuple[List[Coord], List[Coord]]:
    bend_hv = (start[0], end[1])
    bend_vh = (end[0], start[1])

    hv = _segment(start, bend_hv)
    hv += _segment(bend_hv, end)[1:]

    vh = _segment(start, bend_vh)
    vh += _segment(bend_vh, end)[1:]
    return hv, vh


def _find_relays(grid: np.ndarray) -> Tuple[Dict[Coord, int], set[Coord]]:
    h, w = grid.shape
    relays: Dict[Coord, int] = {}
    marker_positions: set[Coord] = set()
    for r, c in np.argwhere(grid == GREY):
        r = int(r)
        c = int(c)
        colored_neighbors: List[Tuple[Coord, int]] = []
        for rr, cc in _neighbors4(r, c, h, w):
            color = int(grid[rr, cc])
            if color not in (BLACK, GREY):
                colored_neighbors.append(((rr, cc), color))
        if len(colored_neighbors) == 1:
            marker_pos, marker_color = colored_neighbors[0]
            relays[(r, c)] = marker_color
            marker_positions.add(marker_pos)
    return relays, marker_positions


def _find_endpoint_pairs(grid: np.ndarray, marker_positions: set[Coord]) -> Dict[int, Tuple[Coord, Coord]]:
    pairs: Dict[int, List[Coord]] = {}
    for r, c in np.argwhere((grid != BLACK) & (grid != GREY)):
        pos = (int(r), int(c))
        if pos in marker_positions:
            continue
        color = int(grid[r, c])
        pairs.setdefault(color, []).append(pos)

    endpoint_pairs: Dict[int, Tuple[Coord, Coord]] = {}
    for color, coords in pairs.items():
        if len(coords) != 2:
            raise ValueError(f"endpoint color {color} should appear exactly twice after removing markers")
        endpoint_pairs[color] = _ordered_pair(coords[0], coords[1])
    return endpoint_pairs


def _relay_count(path: Sequence[Coord], relays: Dict[Coord, int]) -> int:
    return sum(1 for cell in path[1:-1] if cell in relays)


def _paint_path(out: np.ndarray, path: Sequence[Coord], start_color: int, relays: Dict[Coord, int]) -> None:
    current_color = int(start_color)
    for r, c in path[1:-1]:
        if (r, c) in relays:
            current_color = int(relays[(r, c)])
            continue
        if int(out[r, c]) == BLACK:
            out[r, c] = current_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color endpoint pair with an L-shaped route whose color updates at grey relay stations.

    Input structure:
    - The background is black.
    - Each endpoint color appears on exactly two singleton pixels; after removing relay markers, those two pixels form one pair to connect.
    - Grey singleton pixels are relay stations, and each relay has exactly one 4-neighbor colored marker pixel that gives the relay's recolor value.
    - Relay pixels, relay markers, and endpoints all stay visible in the output.

    Rule:
    - For each endpoint pair, use the lexicographically earlier endpoint (topmost, then leftmost) as the traversal start so recoloring order is deterministic.
    - Compare the two candidate L-routes from that start to the other endpoint: horizontal-then-vertical and vertical-then-horizontal.
    - Choose the route that passes through more relay pixels; if the counts tie, choose horizontal-then-vertical.
    - Paint only black cells on the chosen route. The route starts in the endpoint color, and whenever the traversal reaches a relay pixel, all later painted route cells switch to that relay's marker color.

    Key insight:
    - Solving requires three linked inferences: exclude relay markers to recover the true endpoint pairs, pick the correct L-route by counting relays, and propagate recolor changes in traversal order along the chosen route.
    """
    g = np.asarray(grid, dtype=int)
    relays, marker_positions = _find_relays(g)
    endpoint_pairs = _find_endpoint_pairs(g, marker_positions)

    out = g.copy()
    for color, (start, end) in sorted(endpoint_pairs.items(), key=lambda item: item[1][0]):
        hv_path, vh_path = _build_l_paths(start, end)
        if _relay_count(hv_path, relays) >= _relay_count(vh_path, relays):
            chosen = hv_path
        else:
            chosen = vh_path
        _paint_path(out, chosen, color, relays)
    return out


def _relay_counts_for_mode(mode: str, max_hv: int, max_vh: int) -> Tuple[int, int]:
    common_cap = min(max_hv, max_vh, 2)
    if mode == "tie":
        count = random.randint(0, common_cap)
        return count, count

    if mode == "hv_more":
        loser = random.randint(0, min(1, max_vh))
        winner_min = loser + 1
        winner_max = min(max_hv, 2)
        winner = random.randint(winner_min, winner_max)
        return winner, loser

    if mode == "vh_more":
        loser = random.randint(0, min(1, max_hv))
        winner_min = loser + 1
        winner_max = min(max_vh, 2)
        winner = random.randint(winner_min, winner_max)
        return loser, winner

    raise ValueError(f"unknown mode: {mode}")


def _build_box(height: int, width: int, endpoint_color: int, marker_palette: Sequence[int], mode: str) -> np.ndarray:
    box = np.zeros((height, width), dtype=int)
    box[1, 1] = endpoint_color
    box[height - 2, width - 2] = endpoint_color

    top_slots = [((1, c), (2, c)) for c in range(3, width - 3)]
    right_slots = [((r, width - 2), (r, width - 3)) for r in range(3, height - 3)]
    left_slots = [((r, 1), (r, 2)) for r in range(3, height - 3)]
    bottom_slots = [((height - 2, c), (height - 3, c)) for c in range(3, width - 3)]

    hv_slots = top_slots + right_slots
    vh_slots = left_slots + bottom_slots
    n_hv, n_vh = _relay_counts_for_mode(mode, len(hv_slots), len(vh_slots))

    for relay_pos, marker_pos in random.sample(hv_slots, n_hv) + random.sample(vh_slots, n_vh):
        box[relay_pos] = GREY
        box[marker_pos] = random.choice(marker_palette)

    if random.random() < 0.5:
        box = np.fliplr(box).copy()
    if random.random() < 0.5:
        box = np.flipud(box).copy()
    return box


def _distribute(extra: int, buckets: int) -> List[int]:
    values = [1] * buckets
    for _ in range(extra):
        values[random.randrange(buckets)] += 1
    return values


def _generate_row_layout() -> np.ndarray | None:
    n_pairs = random.choice([2, 3])
    size_hi = 12 if n_pairs == 2 else 10
    heights = [random.randint(8, size_hi) for _ in range(n_pairs)]
    widths = [random.randint(8, 12) for _ in range(n_pairs)]

    base_height = sum(heights) + (n_pairs + 1)
    base_width = max(widths) + 2
    if base_height > 30 or base_width > 30:
        return None

    extra_height = random.randint(0, min(30 - base_height, 6))
    extra_width = random.randint(0, min(30 - base_width, 6))
    gaps = _distribute(extra_height, n_pairs + 1)
    final_height = sum(heights) + sum(gaps)
    final_width = base_width + extra_width

    endpoint_colors = random.sample(PALETTE, n_pairs)
    marker_palette = [color for color in PALETTE if color not in endpoint_colors]
    modes = random.sample(["hv_more", "vh_more", "tie"], k=n_pairs)
    random.shuffle(modes)

    grid = np.zeros((final_height, final_width), dtype=int)
    top = gaps[0]
    for i in range(n_pairs):
        box = _build_box(heights[i], widths[i], endpoint_colors[i], marker_palette, modes[i])
        max_left_shift = final_width - widths[i] - 2
        left = 1 + random.randint(0, max_left_shift)
        grid[top : top + heights[i], left : left + widths[i]] = box
        top += heights[i] + gaps[i + 1]

    if random.random() < 0.5:
        grid = grid.T.copy()
    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    if grid is None:
        return False
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    relay_colors, marker_positions = _find_relays(grid)
    if len(relay_colors) != int((grid == GREY).sum()):
        return False

    for marker_pos in marker_positions:
        r, c = marker_pos
        adjacent_relays = sum(1 for rr, cc in _neighbors4(r, c, *grid.shape) if int(grid[rr, cc]) == GREY)
        if adjacent_relays != 1:
            return False

    try:
        endpoint_pairs = _find_endpoint_pairs(grid, marker_positions)
    except ValueError:
        return False

    if len(endpoint_pairs) < 2:
        return False

    for start, end in endpoint_pairs.values():
        if start[0] == end[0] or start[1] == end[1]:
            return False
        if any(int(grid[pos]) == GREY for pos in _build_l_paths(start, end)[0][1:-1] + _build_l_paths(start, end)[1][1:-1]):
            break
    else:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid = _generate_row_layout()
    if not _valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-path puzzle input")
