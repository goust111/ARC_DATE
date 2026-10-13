from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREEN = 3
YELLOW = 4
ORANGE = 7
MAROON = 9
MARKER_COLORS = (1, 2, 4)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _foreground_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _gate_candidates(grid: np.ndarray, comp_set: set[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    gates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in comp_set:
        if int(grid[r, c]) != GREEN:
            continue
        if (
            (r, c - 1) in comp_set
            and (r, c + 1) in comp_set
            and int(grid[r, c - 1]) == MAROON
            and int(grid[r, c + 1]) == MAROON
        ):
            gates.append(((r, c), (0, 1)))
        if (
            (r - 1, c) in comp_set
            and (r + 1, c) in comp_set
            and int(grid[r - 1, c]) == MAROON
            and int(grid[r + 1, c]) == MAROON
        ):
            gates.append(((r, c), (1, 0)))
    return gates


def _path_reaches_parking(
    grid: np.ndarray,
    comp_set: set[tuple[int, int]],
    marker: tuple[int, int],
    parking: tuple[int, int],
    marker_color: int,
) -> bool:
    q = deque([marker])
    seen = {marker}
    allowed = {GREEN, marker_color}
    while q:
        r, c = q.popleft()
        if (r, c) == parking:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in comp_set or (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) not in allowed:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return False


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _fill_enclosure(
    grid: np.ndarray,
    start: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bounds
    if not in_bounds(grid.shape, *start) or int(grid[start]) != BG:
        return set()
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) != BG:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored marker along its green path, then use the parked marker as the color source for filling the gate-adjacent enclosure.

    Input structure:
    - The grid uses black as the dominant background and contains 2-3 disconnected foreground components.
    - Each active component contains one colored marker pixel, one green path, one maroon-green-maroon gate, and one orange U-shaped wall touching the gate ends.
    - The colored marker is the only foreground pixel in its component whose color is neither green, maroon, nor orange.

    Rule:
    - In each valid component, find the unique gate and the unique green path cell adjacent to the gate center but not between the maroon ends; this is the marker's parking cell.
    - Erase the marker and all green path pixels except the gate center, then place the marker color on the parking cell.
    - Let the cell on the opposite side of the gate center define the enclosure entry; flood-fill the black region inside the adjacent orange wall with the marker's color, while keeping the orange wall, the maroon gate ends, and the green gate center unchanged.

    Key insight:
    - The marker's color is transported by the slide step and becomes the fill instruction for the enclosure reached through the gate, so the slide geometry and the fill result are inseparable.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    for comp in _foreground_components(g, bg):
        comp_set = set(comp)
        marker_cells = [
            (r, c)
            for r, c in comp
            if int(g[r, c]) not in (bg, GREEN, ORANGE, MAROON)
        ]
        if len(marker_cells) != 1:
            continue
        marker = marker_cells[0]
        marker_color = int(g[marker])

        gates = _gate_candidates(g, comp_set)
        if len(gates) != 1:
            continue
        gate, axis = gates[0]
        gr, gc = gate
        ar, ac = axis
        gate_ends = {(gr - ar, gc - ac), (gr + ar, gc + ac)}

        parking_cells = [
            (nr, nc)
            for nr, nc in neighbors4(gr, gc)
            if (nr, nc) in comp_set and (nr, nc) not in gate_ends and int(g[nr, nc]) == GREEN
        ]
        wall_cells = {(r, c) for r, c in comp if int(g[r, c]) == ORANGE}
        if len(parking_cells) != 1 or len(wall_cells) < 7:
            continue
        parking = parking_cells[0]

        if not _path_reaches_parking(g, comp_set, marker, parking, marker_color):
            continue

        entry = (2 * gr - parking[0], 2 * gc - parking[1])

        for r, c in comp:
            if (r, c) == gate:
                continue
            if int(g[r, c]) == GREEN or (r, c) == marker:
                out[r, c] = bg
        out[gate] = GREEN
        out[parking] = marker_color

        wr0, wc0, wr1, wc1 = _bbox(wall_cells)
        fill_cells = _fill_enclosure(out, entry, (wr0, wc0, wr1, wc1))
        for r, c in fill_cells:
            out[r, c] = marker_color

    return out


def _rot_left(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _rot_right(dr: int, dc: int) -> tuple[int, int]:
    return dc, -dr


def _wall_geometry(open_dir: tuple[int, int], depth: int) -> dict[str, object]:
    if open_dir == (-1, 0):
        wall = {(r, -2) for r in range(-depth, 1)} | {(r, 2) for r in range(-depth, 1)} | {
            (-depth, c) for c in range(-2, 3)
        }
        interior = {(r, c) for r in range(-depth + 1, 0) for c in range(-1, 2)}
        gate_ends = {(0, -1), (0, 1)}
        parking = (1, 0)
        entry = (-1, 0)
    elif open_dir == (1, 0):
        wall = {(r, -2) for r in range(0, depth + 1)} | {(r, 2) for r in range(0, depth + 1)} | {
            (depth, c) for c in range(-2, 3)
        }
        interior = {(r, c) for r in range(1, depth) for c in range(-1, 2)}
        gate_ends = {(0, -1), (0, 1)}
        parking = (-1, 0)
        entry = (1, 0)
    elif open_dir == (0, -1):
        wall = {(-2, c) for c in range(-depth, 1)} | {(2, c) for c in range(-depth, 1)} | {
            (r, -depth) for r in range(-2, 3)
        }
        interior = {(r, c) for r in range(-1, 2) for c in range(-depth + 1, 0)}
        gate_ends = {(-1, 0), (1, 0)}
        parking = (0, 1)
        entry = (0, -1)
    else:
        wall = {(-2, c) for c in range(0, depth + 1)} | {(2, c) for c in range(0, depth + 1)} | {
            (r, depth) for r in range(-2, 3)
        }
        interior = {(r, c) for r in range(-1, 2) for c in range(1, depth)}
        gate_ends = {(-1, 0), (1, 0)}
        parking = (0, -1)
        entry = (0, 1)
    return {
        "gate": (0, 0),
        "gate_ends": gate_ends,
        "parking": parking,
        "entry": entry,
        "wall": wall,
        "interior": interior,
    }


def _segment_split(turn_count: int, total_path: int) -> list[int]:
    tail = total_path - 1
    if turn_count == 0:
        return [tail]
    if turn_count == 1:
        a = random.randint(1, min(3, tail - 1))
        return [a, tail - a]
    a = random.randint(1, min(3, tail - 2))
    b = random.randint(1, min(3, tail - a - 1))
    c = tail - a - b
    if c < 1:
        return _segment_split(turn_count, total_path)
    return [a, b, c]


def _build_path(
    parking: tuple[int, int],
    park_dir: tuple[int, int],
    turn_count: int,
    total_path: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    segs = _segment_split(turn_count, total_path)
    turn_dir = random.choice([_rot_left(*park_dir), _rot_right(*park_dir)])
    dirs = [park_dir]
    if turn_count >= 1:
        dirs.append(turn_dir)
    if turn_count >= 2:
        dirs.append(park_dir)

    path = [parking]
    cur = parking
    for d, seg_len in zip(dirs, segs):
        dr, dc = d
        for _ in range(seg_len):
            cur = (cur[0] + dr, cur[1] + dc)
            if cur in path:
                raise ValueError("self-intersecting path")
            path.append(cur)
    last_dir = dirs[-1]
    marker = (cur[0] + last_dir[0], cur[1] + last_dir[1])
    return path, marker


def _neighbor_count(cells: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    return sum((nr, nc) in cells for nr, nc in neighbors4(*cell))


def _normalize_coords(coords: set[tuple[int, int]]) -> tuple[dict[tuple[int, int], tuple[int, int]], tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    shift = (-min(rows), -min(cols))
    mapping = {(r, c): (r + shift[0], c + shift[1]) for r, c in coords}
    return mapping, shift


def _build_group(color: int, open_dir: tuple[int, int], turn_count: int) -> dict[str, object]:
    depth = random.randint(3, 5)
    total_path = random.randint(4, 10)
    geom = _wall_geometry(open_dir, depth)
    gate = geom["gate"]
    parking = geom["parking"]
    path, marker = _build_path(parking, (-open_dir[0], -open_dir[1]), turn_count, total_path)

    all_coords = set(geom["wall"]) | set(geom["gate_ends"]) | set(path) | {gate, marker} | set(geom["interior"]) | {
        geom["entry"]
    }
    mapping, _ = _normalize_coords(all_coords)

    def sh(cell: tuple[int, int]) -> tuple[int, int]:
        return mapping[cell]

    norm_path = [sh(cell) for cell in path]
    norm_marker = sh(marker)
    norm_gate = sh(gate)
    norm_parking = sh(parking)
    norm_entry = sh(geom["entry"])
    norm_gate_ends = {sh(cell) for cell in geom["gate_ends"]}
    norm_wall = {sh(cell) for cell in geom["wall"]}
    norm_interior = {sh(cell) for cell in geom["interior"]}

    occupied = set(norm_path) | norm_gate_ends | norm_wall | {norm_gate, norm_marker}
    if len(occupied) != len(norm_path) + len(norm_gate_ends) + len(norm_wall) + 2:
        raise ValueError("overlapping local geometry")
    if _neighbor_count(occupied, norm_marker) != 1:
        raise ValueError("marker is not a leaf")

    rows = [r for r, _ in mapping.values()]
    cols = [c for _, c in mapping.values()]
    h = max(rows) + 1
    w = max(cols) + 1

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    for r, c in norm_wall:
        inp[r, c] = ORANGE
        out[r, c] = ORANGE
    for r, c in norm_gate_ends:
        inp[r, c] = MAROON
        out[r, c] = MAROON
    for r, c in norm_path:
        inp[r, c] = GREEN
    inp[norm_gate] = GREEN
    out[norm_gate] = GREEN
    inp[norm_marker] = color
    out[norm_parking] = color
    for r, c in norm_interior:
        out[r, c] = color

    return {
        "input_patch": inp,
        "output_patch": out,
        "color": color,
        "open_dir": open_dir,
        "turn_count": turn_count,
        "gate": norm_gate,
        "parking": norm_parking,
        "entry": norm_entry,
        "marker": norm_marker,
        "gate_ends": norm_gate_ends,
        "wall": norm_wall,
        "interior": norm_interior,
        "path": set(norm_path),
        "erased": set(norm_path) - {norm_parking},
        "shape": inp.shape,
    }


def _fits_patch(
    patch: np.ndarray,
    occupied: np.ndarray,
    top: int,
    left: int,
) -> bool:
    h, w = patch.shape
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    mask = patch != BG
    if not mask.any():
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    if occupied[r0:r1, c0:c1].any():
        return False
    return True


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    view = dst[top : top + h, left : left + w]
    view[mask] = patch[mask]


def _translate_points(points: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(r + top, c + left) for r, c in points}


def _translate_point(point: tuple[int, int], top: int, left: int) -> tuple[int, int]:
    return point[0] + top, point[1] + left


def _expected_from_groups(shape: tuple[int, int], groups: list[dict[str, object]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for group in groups:
        patch = group["output_patch"]
        top, left = group["origin"]
        _paste_nonzero(out, patch, top, left)
    return out


def _semantic_check(inp: np.ndarray, expected: np.ndarray, groups: list[dict[str, object]]) -> bool:
    pred = main(inp)
    if not np.array_equal(pred, expected):
        return False
    if get_bg(inp) != BG or inp.shape != expected.shape:
        return False
    new_fill = (inp == BG) & (expected != BG)
    if len(set(map(int, expected[new_fill].tolist()))) < 2:
        return False
    if not any(int(group["turn_count"]) == 0 for group in groups):
        return False
    if not any(int(group["turn_count"]) > 0 for group in groups):
        return False
    axes = {(0 if group["open_dir"][0] else 1) for group in groups}
    if len(axes) < 2:
        return False

    for group in groups:
        color = int(group["color"])
        parking = group["parking_g"]
        marker = group["marker_g"]
        gate = group["gate_g"]
        if int(inp[marker]) != color or int(expected[parking]) != color:
            return False
        if int(expected[gate]) != GREEN:
            return False
        for cell in group["wall_g"]:
            if int(inp[cell]) != ORANGE or int(expected[cell]) != ORANGE:
                return False
        for cell in group["gate_ends_g"]:
            if int(inp[cell]) != MAROON or int(expected[cell]) != MAROON:
                return False
        for cell in group["interior_g"]:
            if int(inp[cell]) != BG or int(expected[cell]) != color:
                return False
        for cell in group["erased_g"]:
            if int(expected[cell]) != BG:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_groups = random.choice([2, 3])
    colors = random.sample(list(MARKER_COLORS), n_groups)
    if n_groups == 2:
        dirs = [random.choice([(-1, 0), (1, 0)]), random.choice([(0, -1), (0, 1)])]
        random.shuffle(dirs)
    else:
        dirs = random.sample([(-1, 0), (1, 0), (0, -1), (0, 1)], n_groups)

    turns = [0, 1]
    while len(turns) < n_groups:
        turns.append(random.choice([0, 1, 2]))
    random.shuffle(turns)

    groups = []
    for color, open_dir, turn_count in zip(colors, dirs, turns):
        try:
            groups.append(_build_group(color, open_dir, turn_count))
        except ValueError:
            return None

    H = random.randint(18, 24) if n_groups == 2 else random.randint(22, 28)
    W = random.randint(18, 24) if n_groups == 2 else random.randint(22, 28)
    inp = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    random.shuffle(groups)
    placed_groups: list[dict[str, object]] = []
    for group in groups:
        patch = group["input_patch"]
        h, w = patch.shape
        candidates = [
            (top, left)
            for top in range(H - h + 1)
            for left in range(W - w + 1)
            if _fits_patch(patch, occupied, top, left)
        ]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _paste_nonzero(inp, patch, top, left)
        occupied[top : top + h, left : left + w] |= patch != BG

        placed = dict(group)
        placed["origin"] = (top, left)
        placed["wall_g"] = _translate_points(group["wall"], top, left)
        placed["gate_ends_g"] = _translate_points(group["gate_ends"], top, left)
        placed["interior_g"] = _translate_points(group["interior"], top, left)
        placed["erased_g"] = _translate_points(group["erased"], top, left)
        placed["gate_g"] = _translate_point(group["gate"], top, left)
        placed["parking_g"] = _translate_point(group["parking"], top, left)
        placed["marker_g"] = _translate_point(group["marker"], top, left)
        placed_groups.append(placed)

    counts = Counter(map(int, inp.ravel()))
    if counts[BG] <= max(count for color, count in counts.items() if color != BG):
        return None

    expected = _expected_from_groups(inp.shape, placed_groups)
    if not _semantic_check(inp, expected, placed_groups):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
