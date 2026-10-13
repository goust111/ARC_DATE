import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ENTRY_SIDES = ("left", "right", "top", "bottom")
LEGEND_CORNERS = ("tl", "tr", "bl", "br")
GEN_ATTEMPTS = 500


def _rot_left(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return -dc, dr


def _rot_right(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return dc, -dr


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _find_legend(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for r, c in ((0, 0), (0, w - 1), (h - 2, 0), (h - 2, w - 1)):
        if 0 <= r < h - 1 and grid[r, c] == BLUE and grid[r + 1, c] == GREEN:
            candidates.append((r, c))
    assert len(candidates) == 1, "expected one corner legend"
    return candidates[0]


def _trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    starts = np.argwhere(g == RED)
    assert len(starts) == 1, "expected one red start"
    sr, sc = map(int, starts[0])

    path_mask = (g == RED) | (g == GREY)
    path_cells = {tuple(map(int, pt)) for pt in np.argwhere(path_mask)}
    assert (sr, sc) in path_cells

    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in path_cells:
        nxt = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if (nr, nc) in path_cells]
        neighbor_map[(r, c)] = nxt
        assert 1 <= len(nxt) <= 2, "path must be a simple chain"

    assert len(neighbor_map[(sr, sc)]) == 1, "start must be an endpoint"
    order = [(sr, sc)]
    prev = None
    cur = (sr, sc)
    while True:
        nxts = [pt for pt in neighbor_map[cur] if pt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        prev, cur = cur, nxts[0]
        order.append(cur)

    assert len(order) == len(path_cells), "path traversal must cover all cells"
    er, ec = order[-1]
    assert len(neighbor_map[(er, ec)]) == 1, "path must end at an endpoint"
    return order


def _turn_data(path: list[tuple[int, int]]) -> list[tuple[int, tuple[int, int], tuple[int, int], tuple[int, int], int]]:
    turns = []
    turn_indices = []
    for i in range(1, len(path) - 1):
        d_in = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d_out = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d_in != d_out:
            turn_indices.append(i)

    for k, idx in enumerate(turn_indices):
        d_in = (path[idx][0] - path[idx - 1][0], path[idx][1] - path[idx - 1][1])
        d_out = (path[idx + 1][0] - path[idx][0], path[idx + 1][1] - path[idx][1])
        if d_out == _rot_left(d_in):
            turn = "left"
        else:
            assert d_out == _rot_right(d_in), "turn must be 90 degrees"
            turn = "right"
        next_key_idx = turn_indices[k + 1] if k + 1 < len(turn_indices) else len(path) - 1
        seg_len = abs(path[next_key_idx][0] - path[idx][0]) + abs(path[next_key_idx][1] - path[idx][1])
        turns.append((idx, path[idx], d_out, turn, seg_len))
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the red-to-grey polyline and grow a colored branch at each turn.

    Right turns use the legend's top color; left turns use the bottom color.
    Each branch grows on the outside of the turn, perpendicular to the outgoing
    path segment, with length equal to the next straight segment length.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _find_legend(g)
    path = _trace_path(g)
    path_cells = set(path)

    for _idx, (r, c), d_out, turn, seg_len in _turn_data(path):
        if turn == "right":
            color = BLUE
            bdir = _rot_left(d_out)
        else:
            color = GREEN
            bdir = _rot_right(d_out)
        dr, dc = bdir
        for step in range(1, seg_len + 1):
            nr, nc = r + dr * step, c + dc * step
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                break
            if (nr, nc) in path_cells:
                continue
            if out[nr, nc] == BG:
                out[nr, nc] = color
    return out


def _legend_cells(h: int, w: int, corner: str) -> set[tuple[int, int]]:
    if corner == "tl":
        return {(0, 0), (1, 0)}
    if corner == "tr":
        return {(0, w - 1), (1, w - 1)}
    if corner == "bl":
        return {(h - 2, 0), (h - 1, 0)}
    return {(h - 2, w - 1), (h - 1, w - 1)}


def _place_legend(grid: np.ndarray, corner: str) -> None:
    h, w = grid.shape
    if corner == "tl":
        r, c = 0, 0
    elif corner == "tr":
        r, c = 0, w - 1
    elif corner == "bl":
        r, c = h - 2, 0
    else:
        r, c = h - 2, w - 1
    grid[r, c] = BLUE
    grid[r + 1, c] = GREEN


def _compose_total(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    vals = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(cut - prev)
        prev = cut
    return vals


def _local_to_global(h: int, w: int, side: str, s: int, p: int) -> tuple[int, int]:
    if side == "left":
        return s, p
    if side == "right":
        return s, w - 1 - p
    if side == "top":
        return p, s
    return h - 1 - p, s


def _build_path_points(
    h: int,
    w: int,
    side: str,
    start_s: int,
    primary_lengths: list[int],
    vertical_moves: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    s = start_s
    p = 0
    path.append(_local_to_global(h, w, side, s, p))
    seg_count = len(primary_lengths) + len(vertical_moves)
    for seg_idx in range(seg_count):
        if seg_idx % 2 == 0:
            step_s, step_p = 0, 1
            seg_len = primary_lengths[seg_idx // 2]
        else:
            sign, seg_len = vertical_moves[seg_idx // 2]
            step_s, step_p = sign, 0
        for _ in range(seg_len):
            s += step_s
            p += step_p
            path.append(_local_to_global(h, w, side, s, p))
    return path


def _analyze_branches(path: list[tuple[int, int]], h: int, w: int) -> tuple[list[tuple[int, int]], bool]:
    branch_cells: list[tuple[int, int]] = []
    truncated = False
    path_set = set(path)
    for _idx, (r, c), d_out, turn, seg_len in _turn_data(path):
        bdir = _rot_left(d_out) if turn == "right" else _rot_right(d_out)
        dr, dc = bdir
        for step in range(1, seg_len + 1):
            nr, nc = r + dr * step, c + dc * step
            if not (0 <= nr < h and 0 <= nc < w):
                truncated = True
                break
            if (nr, nc) in path_set:
                return [], truncated
            branch_cells.append((nr, nc))
    return branch_cells, truncated


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 21))
    w = int(rng.integers(12, 21))
    side = str(rng.choice(ENTRY_SIDES))
    corner = str(rng.choice(LEGEND_CORNERS))
    want_trunc = bool(rng.integers(0, 2))

    primary_limit = w - 1 if side in ("left", "right") else h - 1
    secondary_limit = h - 1 if side in ("left", "right") else w - 1
    turn_count = int(rng.integers(2, 6))
    seg_count = turn_count + 1
    primary_count = (seg_count + 1) // 2
    vertical_count = seg_count // 2

    max_primary_total = primary_limit - 1
    if max_primary_total < primary_count + 1 or secondary_limit < 4:
        return None
    primary_total = int(rng.integers(primary_count + 1, max_primary_total + 1))
    primary_lengths = _compose_total(rng, primary_total, primary_count)

    for _ in range(80):
        start_s = int(rng.integers(1, secondary_limit))
        cur_s = start_s
        vertical_moves: list[tuple[int, int]] = []
        ok = True
        for i in range(vertical_count):
            directions = []
            if cur_s > 1:
                directions.append(-1)
            if cur_s < secondary_limit - 1:
                directions.append(1)
            if not directions:
                ok = False
                break
            if i == 0:
                sign = int(rng.choice(directions))
            else:
                preferred = -vertical_moves[-1][0]
                sign = preferred if preferred in directions and rng.random() < 0.7 else int(rng.choice(directions))
            max_len = cur_s - 1 if sign < 0 else (secondary_limit - 1 - cur_s)
            if max_len < 1:
                ok = False
                break
            seg_len = int(rng.integers(1, min(max_len, 5) + 1))
            vertical_moves.append((sign, seg_len))
            cur_s += sign * seg_len
        if not ok:
            continue

        path = _build_path_points(h, w, side, start_s, primary_lengths, vertical_moves)
        if len(path) != len(set(path)):
            continue

        end_r, end_c = path[-1]
        if end_r in (0, h - 1) or end_c in (0, w - 1):
            continue

        legend = _legend_cells(h, w, corner)
        path_set = set(path)
        if legend & path_set:
            continue

        branch_cells, truncated = _analyze_branches(path, h, w)
        if not branch_cells:
            continue
        if want_trunc != truncated:
            continue
        if len(branch_cells) != len(set(branch_cells)):
            continue
        if legend & set(branch_cells):
            continue

        grid = np.full((h, w), BG, dtype=int)
        _place_legend(grid, corner)
        for idx, (r, c) in enumerate(path):
            grid[r, c] = RED if idx == 0 else GREY

        try:
            _trace_path(grid)
        except AssertionError:
            continue

        out = main(grid)
        if np.count_nonzero(out == BLUE) <= np.count_nonzero(grid == BLUE):
            continue
        if np.count_nonzero(out == GREEN) <= np.count_nonzero(grid == GREEN):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
