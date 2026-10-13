from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FRAME = 5
SEPARATOR = 4
MOVE_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIRS = ("up", "down", "left", "right")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords += [(r, c1) for r in range(r0 + 1, r1)]
    coords += [(r1, c) for c in range(c1, c0 - 1, -1)]
    coords += [(r, c0) for r in range(r1 - 1, r0, -1)]
    return coords


def _find_ring_bbox(side: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components(side != bg):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        if len(comp) != 2 * (h + w) - 4:
            continue
        comp_set = set(comp)
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if ((r, c) in comp_set) != on_border:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append((r0, c0, r1, c1))
    if not candidates:
        h, w = side.shape
        for r0 in range(h - 3):
            for r1 in range(r0 + 3, h):
                for c0 in range(w - 3):
                    for c1 in range(c0 + 3, w):
                        perimeter = _perimeter_coords(r0, c0, r1, c1)
                        values = [int(side[r, c]) for r, c in perimeter]
                        if any(v == bg for v in values):
                            continue
                        counts = Counter(values)
                        frame_color, frame_count = counts.most_common(1)[0]
                        if frame_count < len(perimeter) - 6:
                            continue
                        if (
                            int(side[r0, c0]) != frame_color
                            or int(side[r0, c1]) != frame_color
                            or int(side[r1, c0]) != frame_color
                            or int(side[r1, c1]) != frame_color
                        ):
                            continue
                        candidates.append((r0, c0, r1, c1))
        if not candidates:
            return None
    assert len(candidates) == 1, f"expected one frame ring, got {candidates}"
    return candidates[0]


def _find_separator_and_frame(
    grid: np.ndarray, bg: int
) -> tuple[int, int, tuple[int, int, int, int], list[int]]:
    h, w = grid.shape
    matches: list[tuple[int, int, tuple[int, int, int, int], list[int]]] = []
    for c in range(w):
        col = grid[:, c]
        if int(col[0]) == bg or not np.all(col == col[0]):
            continue

        left_bbox = _find_ring_bbox(grid[:, :c], bg) if c > 0 else None
        right_bbox = _find_ring_bbox(grid[:, c + 1 :], bg) if c < w - 1 else None
        if (left_bbox is None) == (right_bbox is None):
            continue

        if left_bbox is not None:
            arena_side = -1
            frame_bbox = left_bbox
            legend_c = c + 1
        else:
            arena_side = 1
            rb = right_bbox
            assert rb is not None
            frame_bbox = (rb[0], rb[1] + c + 1, rb[2], rb[3] + c + 1)
            legend_c = c - 1

        if not (0 <= legend_c < w):
            continue
        legend = [int(grid[r, legend_c]) for r in range(h) if int(grid[r, legend_c]) != bg]
        if len(legend) >= 2 and len(set(legend)) == len(legend):
            matches.append((c, arena_side, frame_bbox, legend))

    assert len(matches) == 1, f"expected one separator/frame parse, got {matches}"
    return matches[0]


def _parse(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep_c, arena_side, frame_bbox, order = _find_separator_and_frame(g, bg)
    r0, c0, r1, c1 = frame_bbox
    interior_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)

    direction_by_color: dict[int, str] = {}
    for color in order:
        marker_coords = [
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if (r in (r0, r1) or c in (c0, c1)) and int(g[r, c]) == int(color)
        ]
        assert len(marker_coords) == 1, f"expected one marker for color {color}, got {marker_coords}"
        mr, mc = marker_coords[0]
        if mr == r0:
            direction = "up"
        elif mr == r1:
            direction = "down"
        elif mc == c0:
            direction = "left"
        elif mc == c1:
            direction = "right"
        else:
            raise AssertionError("marker must lie on the frame border")
        direction_by_color[int(color)] = direction

    return {
        "bg": bg,
        "separator_c": int(sep_c),
        "arena_side": int(arena_side),
        "frame_bbox": frame_bbox,
        "interior_bbox": interior_bbox,
        "order": [int(c) for c in order],
        "direction_by_color": direction_by_color,
    }


def _slide_batch(
    coords: list[tuple[int, int]],
    direction: str,
    occupied: np.ndarray,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    if not coords:
        return []

    h, w = shape
    landed: list[tuple[int, int]] = []

    if direction in {"up", "down"}:
        by_col: dict[int, list[int]] = {}
        for r, c in coords:
            by_col.setdefault(int(c), []).append(int(r))
        for c, rows in by_col.items():
            obstacles = [-1] + [int(r) for r in np.where(occupied[:, c])[0]] + [h]
            rows = sorted(rows)
            idx = 0
            for a, b in zip(obstacles, obstacles[1:]):
                start = idx
                while idx < len(rows) and a < rows[idx] < b:
                    idx += 1
                count = idx - start
                if count == 0:
                    continue
                if direction == "up":
                    for k in range(count):
                        landed.append((a + 1 + k, c))
                else:
                    for k in range(count):
                        landed.append((b - count + k, c))
        return landed

    by_row: dict[int, list[int]] = {}
    for r, c in coords:
        by_row.setdefault(int(r), []).append(int(c))
    for r, cols in by_row.items():
        obstacles = [-1] + [int(c) for c in np.where(occupied[r])[0]] + [w]
        cols = sorted(cols)
        idx = 0
        for a, b in zip(obstacles, obstacles[1:]):
            start = idx
            while idx < len(cols) and a < cols[idx] < b:
                idx += 1
            count = idx - start
            if count == 0:
                continue
            if direction == "left":
                for k in range(count):
                    landed.append((r, a + 1 + k))
            else:
                for k in range(count):
                    landed.append((r, b - count + k))
    return landed


def _simulate_interior(
    shape: tuple[int, int],
    sources_by_color: dict[int, list[tuple[int, int]]],
    order: list[int],
    direction_by_color: dict[int, str],
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    for color in order:
        coords = sources_by_color.get(int(color), [])
        landed = _slide_batch(coords, direction_by_color[int(color)], occupied, shape)
        for r, c in landed:
            occupied[r, c] = True
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each color batch to the edge named by its boundary marker, in legend order.

    Input structure:
    - The grid has a one-cell rectangular frame around the playfield, plus a full-height separator line.
    - Outside the separator is a one-column legend listing the moving colors from top to bottom.
    - Each moving color also has one same-colored marker embedded in the frame border, while matching source pixels are scattered inside the frame.

    Rule:
    - Read the legend top to bottom to get the processing order.
    - For each color in that order, slide all of its interior pixels simultaneously straight toward the frame side containing that color's border marker.
    - Within each row or column lane, the current batch compresses against the frame or against pixels already landed by earlier batches; then erase original interior positions and keep the frame, markers, separator, and legend unchanged.

    Key insight:
    - The border markers encode direction, but only the legend encodes time; earlier landed batches become the obstacles that reshape later batches.
    """
    g = np.asarray(grid, dtype=int)
    info = _parse(g)
    bg = int(info["bg"])
    ir0, ic0, ir1, ic1 = info["interior_bbox"]
    interior = g[ir0 : ir1 + 1, ic0 : ic1 + 1]

    sources_by_color = {
        int(color): [(int(r), int(c)) for r, c in np.argwhere(interior == int(color))]
        for color in info["order"]
    }
    settled = _simulate_interior(
        interior.shape,
        sources_by_color,
        info["order"],
        info["direction_by_color"],
    )

    out = g.copy()
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = bg
    mask = settled != 0
    out[ir0 : ir1 + 1, ic0 : ic1 + 1][mask] = settled[mask]
    return out


def _add_source(
    sources: dict[int, list[tuple[int, int]]],
    occupied: set[tuple[int, int]],
    color: int,
    coord: tuple[int, int],
) -> bool:
    if coord in occupied:
        return False
    occupied.add(coord)
    sources.setdefault(int(color), []).append((int(coord[0]), int(coord[1])))
    return True


def _random_source_cell(
    shape: tuple[int, int],
    direction: str,
    occupied: set[tuple[int, int]],
    preferred_rows: list[int],
    preferred_cols: list[int],
) -> tuple[int, int] | None:
    h, w = shape
    coords = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in occupied
        and not (
            (direction == "up" and r == 0)
            or (direction == "down" and r == h - 1)
            or (direction == "left" and c == 0)
            or (direction == "right" and c == w - 1)
        )
    ]
    if not coords:
        return None
    preferred = [
        (r, c)
        for r, c in coords
        if (not preferred_rows or r in preferred_rows) or (not preferred_cols or c in preferred_cols)
    ]
    return random.choice(preferred or coords)


def _simulate_full_output(
    inp: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    sources_by_color: dict[int, list[tuple[int, int]]],
    order: list[int],
    direction_by_color: dict[int, str],
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    out = inp.copy()
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG
    interior = _simulate_interior(shape, sources_by_color, order, direction_by_color)
    mask = interior != 0
    out[ir0 : ir1 + 1, ic0 : ic1 + 1][mask] = interior[mask]
    return out


def _clockwise_marker_order(
    frame_bbox: tuple[int, int, int, int], markers: dict[int, tuple[int, int]]
) -> list[int]:
    r0, c0, r1, c1 = frame_bbox
    seq = []
    for c in range(c0 + 1, c1):
        seq.append((r0, c))
    for r in range(r0 + 1, r1):
        seq.append((r, c1))
    for c in range(c1 - 1, c0, -1):
        seq.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        seq.append((r, c0))
    inv = {coord: color for color, coord in markers.items()}
    return [int(inv[coord]) for coord in seq if coord in inv]


def _validate_instance(
    inp: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    sources_by_color: dict[int, list[tuple[int, int]]],
    order: list[int],
    direction_by_color: dict[int, str],
    special_colors: tuple[int, int],
    clockwise_order: list[int],
) -> bool:
    expected = _simulate_full_output(inp, frame_bbox, shape, sources_by_color, order, direction_by_color)

    try:
        actual = main(inp)
        parsed = _parse(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if parsed["order"] != order or parsed["direction_by_color"] != direction_by_color:
        return False
    if len(set(direction_by_color.values())) < 2:
        return False
    if order == sorted(order) or order == clockwise_order:
        return False

    changed_order = False
    for i in range(len(order) - 1):
        alt_order = order.copy()
        alt_order[i], alt_order[i + 1] = alt_order[i + 1], alt_order[i]
        alt = _simulate_interior(shape, sources_by_color, alt_order, direction_by_color)
        if not np.array_equal(alt, expected[frame_bbox[0] + 1 : frame_bbox[2], frame_bbox[1] + 1 : frame_bbox[3]]):
            changed_order = True
            break
    if not changed_order:
        return False

    for color in special_colors:
        alt_dirs = dict(direction_by_color)
        alt_dirs[int(color)] = OPPOSITE[alt_dirs[int(color)]]
        alt = _simulate_interior(shape, sources_by_color, order, alt_dirs)
        if np.array_equal(alt, expected[frame_bbox[0] + 1 : frame_bbox[2], frame_bbox[1] + 1 : frame_bbox[3]]):
            return False

    for color in order:
        src = sorted(sources_by_color[int(color)])
        dst = sorted(map(tuple, np.argwhere(expected[frame_bbox[0] + 1 : frame_bbox[2], frame_bbox[1] + 1 : frame_bbox[3]] == int(color))))
        if not src or src == dst:
            return False

    return not np.array_equal(inp, expected)


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.randint(3, 5)
    colors = random.sample(MOVE_COLORS, n_colors)
    special_a, special_b = random.sample(colors, 2)

    direction_by_color = {int(special_a): "left", int(special_b): "up"}
    for color in colors:
        if color in direction_by_color:
            continue
        direction_by_color[int(color)] = random.choice(DIRS)

    order = colors[:]
    random.shuffle(order)
    if order.index(special_a) > order.index(special_b):
        ia, ib = order.index(special_a), order.index(special_b)
        order[ia], order[ib] = order[ib], order[ia]

    interior_h = random.randint(max(6, n_colors + 2), 11)
    interior_w = random.randint(max(7, n_colors + 3), 13)
    shape = (interior_h, interior_w)

    sources_by_color: dict[int, list[tuple[int, int]]] = {int(color): [] for color in colors}
    occupied: set[tuple[int, int]] = set()

    block_row = random.randint(1, interior_h - 3)
    a_col = random.randint(1, interior_w - 2)
    b_row = random.randint(block_row + 1, interior_h - 1)
    if not _add_source(sources_by_color, occupied, special_a, (block_row, a_col)):
        return None
    if not _add_source(sources_by_color, occupied, special_b, (b_row, 0)):
        return None

    preferred_rows = [block_row, b_row]
    preferred_cols = [0, a_col]
    target_counts = {int(color): random.randint(2, 4) for color in colors}
    target_counts[int(special_a)] = max(target_counts[int(special_a)], 2)
    target_counts[int(special_b)] = max(target_counts[int(special_b)], 2)

    while len(sources_by_color[int(special_a)]) < target_counts[int(special_a)]:
        cell = _random_source_cell(shape, direction_by_color[int(special_a)], occupied, [block_row], preferred_cols)
        if cell is None or not _add_source(sources_by_color, occupied, special_a, cell):
            return None
    while len(sources_by_color[int(special_b)]) < target_counts[int(special_b)]:
        cell = _random_source_cell(shape, direction_by_color[int(special_b)], occupied, preferred_rows, [0, a_col])
        if cell is None or not _add_source(sources_by_color, occupied, special_b, cell):
            return None

    for color in colors:
        while len(sources_by_color[int(color)]) < target_counts[int(color)]:
            cell = _random_source_cell(shape, direction_by_color[int(color)], occupied, preferred_rows, preferred_cols)
            if cell is None or not _add_source(sources_by_color, occupied, color, cell):
                return None

    legend_on_left = random.choice([True, False])
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    far_margin = random.randint(0, 2)
    frame_h = interior_h + 2
    frame_w = interior_w + 2
    grid_h = top_margin + frame_h + bottom_margin

    if legend_on_left:
        frame_c0 = 2
        sep_c = 1
        legend_c = 0
        grid_w = frame_c0 + frame_w + far_margin
    else:
        frame_c0 = far_margin
        sep_c = frame_c0 + frame_w
        legend_c = sep_c + 1
        grid_w = legend_c + 1

    frame_r0 = top_margin
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)
    r0, c0, r1, c1 = frame_bbox

    dir_to_colors: dict[str, list[int]] = {d: [] for d in DIRS}
    for color, direction in direction_by_color.items():
        dir_to_colors[direction].append(int(color))

    for direction, avail in {
        "up": interior_w,
        "down": interior_w,
        "left": interior_h,
        "right": interior_h,
    }.items():
        if len(dir_to_colors[direction]) > avail:
            return None

    markers: dict[int, tuple[int, int]] = {}
    top_slots = random.sample(list(range(c0 + 1, c1)), len(dir_to_colors["up"]))
    bottom_slots = random.sample(list(range(c0 + 1, c1)), len(dir_to_colors["down"]))
    left_slots = random.sample(list(range(r0 + 1, r1)), len(dir_to_colors["left"]))
    right_slots = random.sample(list(range(r0 + 1, r1)), len(dir_to_colors["right"]))

    for color, c in zip(dir_to_colors["up"], top_slots):
        markers[int(color)] = (r0, int(c))
    for color, c in zip(dir_to_colors["down"], bottom_slots):
        markers[int(color)] = (r1, int(c))
    for color, r in zip(dir_to_colors["left"], left_slots):
        markers[int(color)] = (int(r), c0)
    for color, r in zip(dir_to_colors["right"], right_slots):
        markers[int(color)] = (int(r), c1)

    clockwise_order = _clockwise_marker_order(frame_bbox, markers)
    if clockwise_order == order or sorted(order) == order:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[:, sep_c] = SEPARATOR
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME

    legend_start = random.randint(r0, r1 - len(order) + 1)
    for i, color in enumerate(order):
        grid[legend_start + i, legend_c] = int(color)

    for color, (mr, mc) in markers.items():
        grid[mr, mc] = int(color)

    ir0, ic0 = r0 + 1, c0 + 1
    for color, coords in sources_by_color.items():
        for rr, cc in coords:
            grid[ir0 + rr, ic0 + cc] = int(color)

    if not _validate_instance(
        grid,
        frame_bbox,
        shape,
        sources_by_color,
        [int(c) for c in order],
        {int(k): v for k, v in direction_by_color.items()},
        (int(special_a), int(special_b)),
        clockwise_order,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
