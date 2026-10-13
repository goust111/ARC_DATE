import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MAROON = 9
START_COLORS = (BLUE, GREEN, RED)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GEN_RETRIES = 400
PATH_RETRIES = 300
FRAME_RETRIES = 150
LOCAL_SPAN_LIMIT = 24
SEGMENT_MIN = 2
SEGMENT_MAX = 6
EXTRA_PAD_RANGE = (1, 4)


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _same_color_component(
    grid: np.ndarray, start: tuple[int, int], seen: np.ndarray
) -> list[tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    color = int(grid[sr, sc])
    q = deque([start])
    seen[sr, sc] = True
    comp = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if _inside(h, w, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                q.append((nr, nc))
    return comp


def _find_start(grid: np.ndarray) -> tuple[int, tuple[int, int], tuple[int, int], int]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GREY, MAROON) or seen[r, c]:
                continue
            comp = _same_color_component(grid, (r, c), seen)
            if len(comp) != 2:
                continue
            (r0, c0), (r1, c1) = comp
            if abs(r0 - r1) + abs(c0 - c1) != 1:
                continue
            boundary = [
                p
                for p in comp
                if p[0] in (0, h - 1) or p[1] in (0, w - 1)
            ]
            if len(boundary) != 1:
                continue
            start = boundary[0]
            nxt = comp[0] if comp[1] == start else comp[1]
            dr = nxt[0] - start[0]
            dc = nxt[1] - start[1]
            dir_idx = DIRS.index((dr, dc))
            return color, start, nxt, dir_idx
    raise AssertionError("start bar not found")


def _grey_component(grid: np.ndarray) -> list[tuple[int, int]]:
    pts = np.argwhere(grid == GREY)
    if pts.size == 0:
        raise AssertionError("grey frame not found")
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    sr, sc = map(int, pts[0])
    return _same_color_component(grid, (sr, sc), seen)


def _find_frame(grid: np.ndarray) -> dict[str, int | tuple[int, int]]:
    comp = _grey_component(grid)
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 != c1 - c0:
        raise AssertionError("frame bbox must be square")
    outer = r1 - r0 + 1
    inner = outer - 2
    if inner < 1:
        raise AssertionError("frame cavity too small")
    opening = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r not in (r0, r1) and c not in (c0, c1):
                continue
            if int(grid[r, c]) != GREY:
                opening.append((r, c))
    if len(opening) != 1:
        raise AssertionError("frame opening must be unique")
    op_r, op_c = opening[0]
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != BLACK:
                raise AssertionError("frame cavity must be black")
    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "size": inner,
        "opening": (op_r, op_c),
    }


def _trace_path(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], int, dict[str, int | tuple[int, int]]]:
    h, w = grid.shape
    color, start, _nxt, dir_idx = _find_start(grid)
    frame = _find_frame(grid)
    opening = frame["opening"]
    assert isinstance(opening, tuple)

    path = [start]
    pos = start
    turns = 0
    seen_states = set()

    while True:
        if pos == opening:
            break
        state = (pos, dir_idx)
        if state in seen_states:
            raise AssertionError("path loop detected")
        seen_states.add(state)

        dr, dc = DIRS[dir_idx]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not _inside(h, w, nr, nc):
            raise AssertionError("path ran out of bounds")

        if (nr, nc) == opening:
            pos = (nr, nc)
            path.append(pos)
            continue

        ahead = int(grid[nr, nc])
        if ahead == MAROON:
            options = []
            for ndir in ((dir_idx - 1) % 4, (dir_idx + 1) % 4):
                rr, cc = pos[0] + DIRS[ndir][0], pos[1] + DIRS[ndir][1]
                if not _inside(h, w, rr, cc):
                    continue
                cell = int(grid[rr, cc])
                if cell not in (MAROON, GREY):
                    options.append(ndir)
            if len(options) != 1:
                raise AssertionError("turn must have a unique perpendicular continuation")
            dir_idx = options[0]
            turns += 1
            dr, dc = DIRS[dir_idx]
            nr, nc = pos[0] + dr, pos[1] + dc
            if not _inside(h, w, nr, nc):
                raise AssertionError("turned path ran out of bounds")
            if int(grid[nr, nc]) in (MAROON, GREY):
                raise AssertionError("turned path must enter a free cell")

        elif ahead == GREY:
            raise AssertionError("solver should only enter the frame through its opening")

        pos = (nr, nc)
        path.append(pos)
        if len(path) > h * w:
            raise AssertionError("path too long")

    return color, path, turns, frame


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    color, path, turns, frame = _trace_path(g)
    for r, c in path:
        out[r, c] = color

    size = int(frame["size"])
    if turns > 0:
        top = int(frame["r0"]) + 1 + (size - turns) // 2
        left = int(frame["c0"]) + 1 + (size - turns) // 2
        out[top : top + turns, left : left + turns] = color

    return out


def _segment_cells(
    start: tuple[int, int], dir_idx: int, length: int
) -> list[tuple[int, int]]:
    dr, dc = DIRS[dir_idx]
    return [(start[0] + dr * step, start[1] + dc * step) for step in range(1, length + 1)]


def _segment_valid(
    path_set: set[tuple[int, int]],
    current: tuple[int, int],
    cells: list[tuple[int, int]],
) -> bool:
    prev = current
    for r, c in cells:
        if r < 1 or (r, c) in path_set:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in path_set and (nr, nc) != prev:
                return False
        prev = (r, c)
    rows = [r for r, _ in path_set] + [r for r, _ in cells]
    cols = [c for _, c in path_set] + [c for _, c in cells]
    if max(rows) - min(rows) > LOCAL_SPAN_LIMIT or max(cols) - min(cols) > LOCAL_SPAN_LIMIT:
        return False
    return True


def _valid_lengths(
    current: tuple[int, int],
    dir_idx: int,
    path_set: set[tuple[int, int]],
) -> list[int]:
    choices = list(range(SEGMENT_MIN, SEGMENT_MAX + 1))
    if DIRS[dir_idx] == (-1, 0):
        max_len = current[0] - 1
        choices = [length for length in range(1, max_len + 1)]
    random.shuffle(choices)
    valid = []
    for length in choices:
        cells = _segment_cells(current, dir_idx, length)
        if _segment_valid(path_set, current, cells):
            valid.append(length)
    return valid


def _build_canonical_path(turns: int) -> list[tuple[int, int]] | None:
    for _ in range(PATH_RETRIES):
        path = [(0, 0)]
        path_set = {(0, 0)}
        current = (0, 0)
        dir_idx = 2
        ok = True
        for seg_idx in range(turns + 1):
            lengths = _valid_lengths(current, dir_idx, path_set)
            if not lengths:
                ok = False
                break
            length = random.choice(lengths)
            cells = _segment_cells(current, dir_idx, length)
            path.extend(cells)
            path_set.update(cells)
            current = cells[-1]
            if seg_idx == turns:
                continue
            next_dirs = []
            for ndir in ((dir_idx - 1) % 4, (dir_idx + 1) % 4):
                if _valid_lengths(current, ndir, path_set):
                    next_dirs.append(ndir)
            if not next_dirs:
                ok = False
                break
            dir_idx = random.choice(next_dirs)
        if ok:
            return path
    return None


def _frame_from_opening(
    opening: tuple[int, int], dir_idx: int, size: int, offset: int
) -> tuple[int, int, int, int]:
    r, c = opening
    outer = size + 2
    dr, dc = DIRS[dir_idx]
    if (dr, dc) == (1, 0):
        r0, c0 = r, c - offset
    elif (dr, dc) == (-1, 0):
        r0, c0 = r - (outer - 1), c - offset
    elif (dr, dc) == (0, 1):
        r0, c0 = r - offset, c
    else:
        r0, c0 = r - offset, c - (outer - 1)
    return r0, c0, r0 + outer - 1, c0 + outer - 1


def _frame_candidates(turns: int) -> list[int]:
    return [size for size in range(turns + 2, 11) if size % 2 == turns % 2]


def _choose_frame(path: list[tuple[int, int]], turns: int) -> dict[str, int] | None:
    if len(path) < 2:
        return None
    opening = path[-1]
    prev = path[-2]
    dr = opening[0] - prev[0]
    dc = opening[1] - prev[1]
    dir_idx = DIRS.index((dr, dc))
    path_body = set(path[:-1])

    sizes = _frame_candidates(turns)
    random.shuffle(sizes)
    for _ in range(FRAME_RETRIES):
        if not sizes:
            break
        size = random.choice(sizes)
        offsets = list(range(1, size + 1))
        random.shuffle(offsets)
        for offset in offsets:
            r0, c0, r1, c1 = _frame_from_opening(opening, dir_idx, size, offset)
            if r0 < 0:
                continue
            if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in path_body):
                continue
            border = {
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)
            }
            border.discard(opening)
            if any(
                abs(r - br) + abs(c - bc) == 1
                for r, c in path_body
                for br, bc in border
            ):
                continue
            span_rows = [r for r, _ in path] + [r0, r1]
            span_cols = [c for _, c in path] + [c0, c1]
            if max(span_rows) - min(span_rows) > LOCAL_SPAN_LIMIT:
                continue
            if max(span_cols) - min(span_cols) > LOCAL_SPAN_LIMIT:
                continue
            return {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "size": size}
    return None


def _draw_frame(grid: np.ndarray, frame: dict[str, int], opening: tuple[int, int]) -> None:
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[opening] = BLACK


def _rotate_grid(grid: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(grid, turns).copy()


def _make_input(turns: int) -> np.ndarray | None:
    path = _build_canonical_path(turns)
    if path is None:
        return None
    frame = _choose_frame(path, turns)
    if frame is None:
        return None

    all_rows = [r for r, _ in path] + [frame["r0"], frame["r1"]]
    all_cols = [c for _, c in path] + [frame["c0"], frame["c1"]]

    left_extra = random.randint(*EXTRA_PAD_RANGE)
    right_extra = random.randint(*EXTRA_PAD_RANGE)
    bottom_extra = random.randint(*EXTRA_PAD_RANGE)
    col_shift = -min(all_cols) + left_extra

    height = max(all_rows) + 1 + bottom_extra
    width = max(all_cols) + col_shift + 1 + right_extra
    grid = np.full((height, width), BLACK, dtype=int)

    shifted_path = [(r, c + col_shift) for r, c in path]
    shifted_set = set(shifted_path)
    opening = shifted_path[-1]
    shifted_frame = {
        "r0": frame["r0"],
        "c0": frame["c0"] + col_shift,
        "r1": frame["r1"],
        "c1": frame["c1"] + col_shift,
        "size": frame["size"],
    }
    cavity = {
        (r, c)
        for r in range(shifted_frame["r0"] + 1, shifted_frame["r1"])
        for c in range(shifted_frame["c0"] + 1, shifted_frame["c1"])
    }

    _draw_frame(grid, shifted_frame, opening)

    for r, c in shifted_path:
        for nr, nc in _neighbors4(r, c):
            if not _inside(height, width, nr, nc):
                continue
            if (nr, nc) in shifted_set or (nr, nc) in cavity:
                continue
            if int(grid[nr, nc]) == BLACK:
                grid[nr, nc] = MAROON

    start_color = random.choice(START_COLORS)
    grid[shifted_path[0]] = start_color
    grid[shifted_path[1]] = start_color

    rot = random.randrange(4)
    grid = _rotate_grid(grid, rot)

    color, traced, found_turns, found_frame = _trace_path(grid)
    if color != start_color or found_turns != turns:
        return None
    opening_after = found_frame["opening"]
    assert isinstance(opening_after, tuple)
    if traced[-1] != opening_after:
        return None

    out = main(grid)
    inner = int(found_frame["size"])
    top = int(found_frame["r0"]) + 1 + (inner - turns) // 2
    left = int(found_frame["c0"]) + 1 + (inner - turns) // 2
    if not np.all(out[top : top + turns, left : left + turns] == start_color):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        turns = random.randint(1, 4)
        grid = _make_input(turns)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to build a valid puzzle")
