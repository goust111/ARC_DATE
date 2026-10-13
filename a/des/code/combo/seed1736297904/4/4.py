import numpy as np


BG = 0
ORANGE = 7
MAROON = 9
BORDER_PALETTE = (1, 2, 3, 4, 5, 6, 8)
SIDES = ("top", "bottom", "left", "right")
MODE_SHAPES = {
    "none": ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2)),
    "partial": ((1, 3), (3, 1), (2, 2), (2, 3), (3, 2)),
    "all_connected": ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2)),
}
MAX_GENERATE_ATTEMPTS = 300


def _scan_to_border(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> tuple[int, int]:
    h, w = grid.shape
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BG:
        rr += dr
        cc += dc
    if not (0 <= rr < h and 0 <= cc < w):
        raise ValueError("probe must be enclosed by a frame border")
    return rr, cc


def _active_sides(local_r: int, local_c: int, interior_h: int, interior_w: int) -> set[str]:
    distances = {
        "top": local_r,
        "bottom": interior_h + 1 - local_r,
        "left": local_c,
        "right": interior_w + 1 - local_c,
    }
    best = min(distances.values())
    return {side for side, dist in distances.items() if dist == best}


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    seen = set()
    for r, c in np.argwhere(grid == ORANGE):
        r = int(r)
        c = int(c)
        r0, _ = _scan_to_border(grid, r, c, -1, 0)
        r1, _ = _scan_to_border(grid, r, c, 1, 0)
        _, c0 = _scan_to_border(grid, r, c, 0, -1)
        _, c1 = _scan_to_border(grid, r, c, 0, 1)
        bbox = (r0, c0, r1, c1)
        if bbox in seen:
            raise ValueError("each frame must contain exactly one probe")
        seen.add(bbox)
        local_r = r - r0
        local_c = c - c0
        frames.append(
            {
                "bbox": bbox,
                "probe": (r, c),
                "active": _active_sides(local_r, local_c, r1 - r0 - 1, c1 - c0 - 1),
            }
        )
    return frames


def _adjacencies(frames: list[dict]) -> list[tuple[int, int, str, str, tuple[int, ...]]]:
    edges = []
    for i in range(len(frames)):
        ar0, ac0, ar1, ac1 = frames[i]["bbox"]
        for j in range(i + 1, len(frames)):
            br0, bc0, br1, bc1 = frames[j]["bbox"]
            if ac1 == bc0:
                row0 = max(ar0, br0)
                row1 = min(ar1, br1)
                if row0 < row1:
                    edges.append((i, j, "right", "left", (row0, row1, ac1, 1)))
            elif bc1 == ac0:
                row0 = max(ar0, br0)
                row1 = min(ar1, br1)
                if row0 < row1:
                    edges.append((i, j, "left", "right", (row0, row1, ac0, 1)))
            elif ar1 == br0:
                col0 = max(ac0, bc0)
                col1 = min(ac1, bc1)
                if col0 < col1:
                    edges.append((i, j, "bottom", "top", (col0, col1, ar1, 0)))
            elif br1 == ar0:
                col0 = max(ac0, bc0)
                col1 = min(ac1, bc1)
                if col0 < col1:
                    edges.append((i, j, "top", "bottom", (col0, col1, ar0, 0)))
    return edges


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[tuple[int, int]], list[list[int]]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _detect_frames(g)
    parent = list(range(len(frames)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    opened = []
    for i, j, side_i, side_j, payload in _adjacencies(frames):
        if side_i not in frames[i]["active"] or side_j not in frames[j]["active"]:
            continue
        start, end, fixed, axis = payload
        if axis == 1:
            out[start : end + 1, fixed] = BG
        else:
            out[fixed, start : end + 1] = BG
        union(i, j)
        opened.append((i, j))

    groups: dict[int, list[int]] = {}
    for idx in range(len(frames)):
        groups.setdefault(find(idx), []).append(idx)
    components = list(groups.values())
    for group in components:
        if len(group) < 2:
            continue
        for idx in group:
            pr, pc = frames[idx]["probe"]
            out[pr, pc] = MAROON

    return out, frames, opened, components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open a shared wall only when both neighboring frames activate that same wall.
    Activated walls come from the orange probe's nearest side(s); connected probes turn maroon.
    """
    return _solve(np.asarray(grid, dtype=int))[0]


def _room_edges(rows: int, cols: int) -> list[tuple[tuple[int, int], str, tuple[int, int], str]]:
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append(((r, c), "right", (r, c + 1), "left"))
            if r + 1 < rows:
                edges.append(((r, c), "bottom", (r + 1, c), "top"))
    return edges


def _target_edges(mode: str, rows: int, cols: int, rng: np.random.Generator) -> set[int]:
    edges = _room_edges(rows, cols)
    if mode == "none":
        return set()
    if mode == "partial":
        return {int(rng.integers(len(edges)))}
    assert mode == "all_connected"
    if (rows, cols) in ((1, 2), (2, 1), (1, 3), (3, 1)):
        return set(range(len(edges)))
    if (rows, cols) == (2, 2):
        drop = int(rng.integers(len(edges)))
        return {idx for idx in range(len(edges)) if idx != drop}
    raise ValueError("unsupported all_connected shape")


def _pick_probe_positions(
    rows: int,
    cols: int,
    interior_h: int,
    interior_w: int,
    target_edge_ids: set[int],
    rng: np.random.Generator,
) -> dict[tuple[int, int], tuple[int, int]]:
    rooms = [(r, c) for r in range(rows) for c in range(cols)]
    required = {room: set() for room in rooms}
    forbidden = {room: set() for room in rooms}

    edges = _room_edges(rows, cols)
    for idx, (room_a, side_a, room_b, side_b) in enumerate(edges):
        if idx in target_edge_ids:
            required[room_a].add(side_a)
            required[room_b].add(side_b)
        else:
            forbidden[room_a].add(side_a)
            forbidden[room_b].add(side_b)

    positions = {}
    for room in rooms:
        candidates = []
        for local_r in range(1, interior_h + 1):
            for local_c in range(1, interior_w + 1):
                active = _active_sides(local_r, local_c, interior_h, interior_w)
                if required[room].issubset(active) and active.isdisjoint(forbidden[room]):
                    candidates.append((local_r, local_c))
        if not candidates:
            raise ValueError("no probe position satisfies the requested edge pattern")
        positions[room] = candidates[int(rng.integers(len(candidates)))]
    return positions


def _paint_frame_layout(
    rows: int,
    cols: int,
    interior_h: int,
    interior_w: int,
    positions: dict[tuple[int, int], tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    top = int(rng.integers(1, 4))
    bottom = int(rng.integers(1, 4))
    left = int(rng.integers(1, 4))
    right = int(rng.integers(1, 4))

    cluster_h = rows * (interior_h + 1) + 1
    cluster_w = cols * (interior_w + 1) + 1
    grid = np.zeros((top + cluster_h + bottom, left + cluster_w + right), dtype=int)

    row_lines = [top + k * (interior_h + 1) for k in range(rows + 1)]
    col_lines = [left + k * (interior_w + 1) for k in range(cols + 1)]
    colors = np.array(rng.permutation(BORDER_PALETTE)[: rows * cols], dtype=int).reshape(rows, cols)

    for r in range(rows):
        for c in range(cols):
            r0, r1 = row_lines[r], row_lines[r + 1]
            c0, c1 = col_lines[c], col_lines[c + 1]
            color = int(colors[r, c])
            if r == 0:
                grid[r0, c0 : c1 + 1] = color
            if r == rows - 1:
                grid[r1, c0 : c1 + 1] = color
            if c == 0:
                grid[r0 : r1 + 1, c0] = color
            if c == cols - 1:
                grid[r0 : r1 + 1, c1] = color

    for r in range(rows):
        for c in range(cols - 1):
            color = int(colors[r, c + int(rng.integers(2))])
            grid[row_lines[r] : row_lines[r + 1] + 1, col_lines[c + 1]] = color

    for r in range(rows - 1):
        for c in range(cols):
            color = int(colors[r + int(rng.integers(2)), c])
            grid[row_lines[r + 1], col_lines[c] : col_lines[c + 1] + 1] = color

    for (r, c), (local_r, local_c) in positions.items():
        grid[row_lines[r] + local_r, col_lines[c] + local_c] = ORANGE

    return grid


def _choose_dimensions(mode: str, rows: int, cols: int, rng: np.random.Generator) -> tuple[int, int]:
    interior_h = int(rng.integers(3, 7))
    interior_w = int(rng.integers(3, 7))
    if mode == "all_connected" and cols == 3:
        interior_w = int(rng.choice(np.array([3, 5], dtype=int)))
    if mode == "all_connected" and rows == 3:
        interior_h = int(rng.choice(np.array([3, 5], dtype=int)))
    return interior_h, interior_w


def _valid_mode_output(mode: str, out: np.ndarray, frames: list[dict], opened: list[tuple[int, int]]) -> bool:
    probe_colors = [int(out[r, c]) for r, c in (frame["probe"] for frame in frames)]
    if mode == "none":
        return len(opened) == 0 and all(color == ORANGE for color in probe_colors)
    if mode == "partial":
        return len(opened) == 1 and ORANGE in probe_colors and MAROON in probe_colors
    return all(color == MAROON for color in probe_colors)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    mode = str(rng.choice(np.array(["none", "partial", "all_connected"], dtype=object)))
    rows, cols = MODE_SHAPES[mode][int(rng.integers(len(MODE_SHAPES[mode])))]
    interior_h, interior_w = _choose_dimensions(mode, rows, cols, rng)

    try:
        target_edge_ids = _target_edges(mode, rows, cols, rng)
        positions = _pick_probe_positions(rows, cols, interior_h, interior_w, target_edge_ids, rng)
        grid = _paint_frame_layout(rows, cols, interior_h, interior_w, positions, rng)
        out, frames, opened, _ = _solve(grid)
    except ValueError:
        return None

    if len(frames) != rows * cols:
        return None
    if not _valid_mode_output(mode, out, frames, opened):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATE_ATTEMPTS} attempts")
