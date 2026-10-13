import numpy as np


BG = 0
WALL = 5
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MIN_H = 12
MAX_H = 20
MIN_W = 12
MAX_W = 20
MIN_PAIRS = 3
MAX_PAIRS = 5
MAX_WALLS_PER_PAIR = 3
EXTRA_WALL_MAX = 3
PAIR_TRIES = 300
WALL_TRIES = 120
EXTRA_WALL_TRIES = 120
GEN_TRIES = 400

_RNG = np.random.default_rng()


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            stack = [(r, c)]
            labels[r, c] = comp_id
            while stack:
                cr, cc = stack.pop()
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        stack.append((nr, nc))
    return labels, comp_id


def _wall_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid, return_counts=True)
    non_bg = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) != bg]
    if WALL in dict(non_bg) and dict(non_bg)[WALL] > 2:
        return WALL
    candidates = [color for color, count in non_bg if count != 2]
    assert len(candidates) == 1, f"expected one wall color, got {candidates}"
    return int(candidates[0])


def _count_crossed_walls(labels: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> int:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        c0, c3 = sorted((c1, c2))
        crossed = {int(v) for v in labels[r1, c0 + 1 : c3] if int(v) != 0}
    else:
        assert c1 == c2, "marker pair must be collinear"
        r0, r3 = sorted((r1, r2))
        crossed = {int(v) for v in labels[r0 + 1 : r3, c1] if int(v) != 0}
    return len(crossed)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color marker pairs iff the number of crossed grey wall segments is even.

    Input structure:
    - Background is black.
    - Grey pixels form disjoint horizontal/vertical wall segments.
    - Every other foreground color appears exactly twice as a marker pair.
    - Each pair lies on one row or one column.

    Rule:
    - Count how many grey wall segments lie on the open interval between each pair.
    - If that count is even, draw the full straight connection in the pair color.
    - If that count is odd, leave the two marker pixels unconnected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    wall = _wall_color(g, bg)
    wall_labels, _ = _label_components(g == wall)

    values, counts = np.unique(g, return_counts=True)
    pair_colors = [int(v) for v, c in zip(values, counts) if int(v) not in {bg, wall} and int(c) == 2]

    out = g.copy()
    for color in pair_colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 2, f"expected exactly two markers for color {color}"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        assert a[0] == b[0] or a[1] == b[1], f"pair {color} is not axis-aligned"
        if _count_crossed_walls(wall_labels, a, b) % 2 == 0:
            if a[0] == b[0]:
                r = a[0]
                c0, c1 = sorted((a[1], b[1]))
                out[r, c0 : c1 + 1] = color
            else:
                c = a[1]
                r0, r1 = sorted((a[0], b[0]))
                out[r0 : r1 + 1, c] = color
    return out


def _path_cells(spec: dict) -> set[tuple[int, int]]:
    if spec["orient"] == "h":
        return {(spec["row"], c) for c in range(spec["c0"], spec["c1"] + 1)}
    return {(r, spec["col"]) for r in range(spec["r0"], spec["r1"] + 1)}


def _endpoints(spec: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    if spec["orient"] == "h":
        return (spec["row"], spec["c0"]), (spec["row"], spec["c1"])
    return (spec["r0"], spec["col"]), (spec["r1"], spec["col"])


def _too_close_to_existing(points: list[tuple[int, int]], endpoints: list[tuple[int, int]]) -> bool:
    for r, c in points:
        for er, ec in endpoints:
            if max(abs(r - er), abs(c - ec)) <= 1:
                return True
    return False


def _sample_spaced_positions(candidates: list[int], count: int) -> list[int] | None:
    if count == 0:
        return []
    for _ in range(80):
        shuffled = candidates.copy()
        _RNG.shuffle(shuffled)
        chosen: list[int] = []
        for value in shuffled:
            if all(abs(value - prev) >= 2 for prev in chosen):
                chosen.append(int(value))
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _segment_cells(kind: str, fixed: int, start: int, end: int) -> list[tuple[int, int]]:
    if kind == "v":
        return [(r, fixed) for r in range(start, end + 1)]
    return [(fixed, c) for c in range(start, end + 1)]


def _can_place_wall_segment(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BG:
            return False
        if (r, c) in blocked:
            return False
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cell_set:
                continue
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == WALL:
                return False
    return True


def _candidate_crossing_segments(
    shape: tuple[int, int],
    kind: str,
    anchor_r: int,
    anchor_c: int,
) -> list[list[tuple[int, int]]]:
    h, w = shape
    candidates: list[list[tuple[int, int]]] = []
    lengths = [2, 3, 4, 5]
    _RNG.shuffle(lengths)
    if kind == "v":
        for length in lengths:
            offsets = list(range(length))
            _RNG.shuffle(offsets)
            for offset in offsets:
                start = anchor_r - offset
                end = start + length - 1
                if 0 <= start <= end < h:
                    candidates.append(_segment_cells("v", anchor_c, start, end))
    else:
        for length in lengths:
            offsets = list(range(length))
            _RNG.shuffle(offsets)
            for offset in offsets:
                start = anchor_c - offset
                end = start + length - 1
                if 0 <= start <= end < w:
                    candidates.append(_segment_cells("h", anchor_r, start, end))
    return candidates


def _place_crossing_segment(
    grid: np.ndarray,
    kind: str,
    anchor_r: int,
    anchor_c: int,
    blocked: set[tuple[int, int]],
) -> bool:
    for cells in _candidate_crossing_segments(grid.shape, kind, anchor_r, anchor_c):
        if _can_place_wall_segment(grid, cells, blocked):
            for r, c in cells:
                grid[r, c] = WALL
            return True
    return False


def _place_extra_wall(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for _ in range(EXTRA_WALL_TRIES):
        kind = "h" if _RNG.random() < 0.5 else "v"
        if kind == "h":
            row = int(_RNG.integers(0, h))
            length = int(_RNG.integers(2, min(7, w) + 1))
            start = int(_RNG.integers(0, w - length + 1))
            cells = _segment_cells("h", row, start, start + length - 1)
        else:
            col = int(_RNG.integers(0, w))
            length = int(_RNG.integers(2, min(7, h) + 1))
            start = int(_RNG.integers(0, h - length + 1))
            cells = _segment_cells("v", col, start, start + length - 1)
        if _can_place_wall_segment(grid, cells, blocked):
            for r, c in cells:
                grid[r, c] = WALL
            return True
    return False


def _build_pair_specs(h: int, w: int, pair_count: int) -> list[dict] | None:
    colors = list(_RNG.choice(np.array(PAIR_COLORS), size=pair_count, replace=False))
    parities = [0, 1]
    while len(parities) < pair_count:
        parities.append(int(_RNG.integers(0, 2)))
    _RNG.shuffle(parities)

    specs: list[dict] = []
    used_paths: set[tuple[int, int]] = set()
    endpoints: list[tuple[int, int]] = []
    used_h_rows: list[int] = []
    used_v_cols: list[int] = []

    for color, parity in zip(colors, parities):
        placed = False
        for _ in range(PAIR_TRIES):
            orient = "h" if _RNG.random() < 0.5 else "v"
            counts = [k for k in range(MAX_WALLS_PER_PAIR + 1) if k % 2 == parity]
            _RNG.shuffle(counts)
            count = int(counts[0])

            if orient == "h":
                gap_min = max(3, 2 * count)
                if w - 1 < gap_min:
                    continue
                row = int(_RNG.integers(1, h - 1))
                if any(abs(row - prev) < 3 for prev in used_h_rows):
                    continue
                gap = int(_RNG.integers(gap_min, min(w - 1, gap_min + 6) + 1))
                c0 = int(_RNG.integers(0, w - gap))
                c1 = c0 + gap
                spec = {
                    "color": int(color),
                    "orient": "h",
                    "row": row,
                    "c0": c0,
                    "c1": c1,
                    "target_count": count,
                }
            else:
                gap_min = max(3, 2 * count)
                if h - 1 < gap_min:
                    continue
                col = int(_RNG.integers(1, w - 1))
                if any(abs(col - prev) < 3 for prev in used_v_cols):
                    continue
                gap = int(_RNG.integers(gap_min, min(h - 1, gap_min + 6) + 1))
                r0 = int(_RNG.integers(0, h - gap))
                r1 = r0 + gap
                spec = {
                    "color": int(color),
                    "orient": "v",
                    "col": col,
                    "r0": r0,
                    "r1": r1,
                    "target_count": count,
                }

            path = _path_cells(spec)
            if path & used_paths:
                continue
            pts = list(_endpoints(spec))
            if _too_close_to_existing(pts, endpoints):
                continue

            spec["path"] = path
            specs.append(spec)
            used_paths |= path
            endpoints.extend(pts)
            if orient == "h":
                used_h_rows.append(spec["row"])
            else:
                used_v_cols.append(spec["col"])
            placed = True
            break
        if not placed:
            return None
    return specs


def _paint_pair_markers(grid: np.ndarray, specs: list[dict]) -> None:
    for spec in specs:
        a, b = _endpoints(spec)
        grid[a] = spec["color"]
        grid[b] = spec["color"]


def _place_pair_walls(grid: np.ndarray, specs: list[dict]) -> bool:
    all_paths = set().union(*(spec["path"] for spec in specs))
    v_pair_cols = {spec["col"] for spec in specs if spec["orient"] == "v"}
    h_pair_rows = {spec["row"] for spec in specs if spec["orient"] == "h"}

    for spec in specs:
        if spec["orient"] == "h":
            row = spec["row"]
            candidates = [c for c in range(spec["c0"] + 1, spec["c1"]) if c not in v_pair_cols]
            chosen = _sample_spaced_positions(candidates, spec["target_count"])
            if chosen is None:
                return False
            for c in chosen:
                blocked = all_paths - {(row, c)}
                if not _place_crossing_segment(grid, "v", row, c, blocked):
                    return False
        else:
            col = spec["col"]
            candidates = [r for r in range(spec["r0"] + 1, spec["r1"]) if r not in h_pair_rows]
            chosen = _sample_spaced_positions(candidates, spec["target_count"])
            if chosen is None:
                return False
            for r in chosen:
                blocked = all_paths - {(r, col)}
                if not _place_crossing_segment(grid, "h", r, col, blocked):
                    return False
    return True


def _validate_structure(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    if bg != BG:
        return False
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    if WALL not in count_map or count_map[WALL] < 3:
        return False

    pair_colors = [color for color, count in count_map.items() if color not in {BG, WALL} and count == 2]
    if len(pair_colors) < 2:
        return False
    if sum(count for color, count in count_map.items() if color not in {BG, WALL}) != 2 * len(pair_colors):
        return False

    labels, n_comp = _label_components(grid == WALL)
    if n_comp == 0:
        return False
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labels == comp_id)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if r0 != r1 and c0 != c1:
            return False
        if r0 == r1 and len(coords) != c1 - c0 + 1:
            return False
        if c0 == c1 and len(coords) != r1 - r0 + 1:
            return False

    odd = 0
    even = 0
    for color in pair_colors:
        coords = np.argwhere(grid == color)
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        if not (a[0] == b[0] or a[1] == b[1]):
            return False
        crossed = _count_crossed_walls(labels, a, b)
        if crossed % 2 == 0:
            even += 1
        else:
            odd += 1
    if even == 0 or odd == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(MIN_H, MAX_H + 1))
    w = int(_RNG.integers(MIN_W, MAX_W + 1))
    pair_count = int(_RNG.integers(MIN_PAIRS, MAX_PAIRS + 1))

    specs = _build_pair_specs(h, w, pair_count)
    if specs is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    if not _place_pair_walls(grid, specs):
        return None

    blocked = set().union(*(spec["path"] for spec in specs))
    extra_walls = int(_RNG.integers(0, EXTRA_WALL_MAX + 1))
    placed_extra = 0
    for _ in range(extra_walls):
        if _place_extra_wall(grid, blocked):
            placed_extra += 1
    if extra_walls > 0 and placed_extra == 0:
        return None

    _paint_pair_markers(grid, specs)
    if not _validate_structure(grid):
        return None

    out = main(grid)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
