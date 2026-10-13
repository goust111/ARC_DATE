import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
LINE = 8
PATTERN_COLORS = [1, 2, 3, 4, 6, 7, 9]
CELL = 3
BRANCHES = ("straight", "left90", "right90", "left180", "right180")
DIR_INDEX = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
NEIGH4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
NEIGH8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = NEIGH4 if connectivity == 4 else NEIGH8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neigh:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    dst[r0 : r0 + h, c0 : c0 + w] = src


def _frame_candidate(mask: np.ndarray) -> tuple[list[int], list[int], int] | None:
    h, w = mask.shape
    full_rows = [r for r in range(h) if bool(mask[r, :].all())]
    full_cols = [c for c in range(w) if bool(mask[:, c].all())]
    if len(full_rows) < 3 or len(full_cols) < 3:
        return None
    if full_rows[0] != 0 or full_rows[-1] != h - 1:
        return None
    if full_cols[0] != 0 or full_cols[-1] != w - 1:
        return None
    row_steps = [full_rows[i + 1] - full_rows[i] for i in range(len(full_rows) - 1)]
    col_steps = [full_cols[i + 1] - full_cols[i] for i in range(len(full_cols) - 1)]
    if len(set(row_steps)) != 1 or len(set(col_steps)) != 1:
        return None
    if row_steps[0] <= 1 or col_steps[0] <= 1:
        return None
    cell_h = row_steps[0] - 1
    cell_w = col_steps[0] - 1
    if cell_h != cell_w:
        return None
    template = np.zeros_like(mask, dtype=bool)
    template[full_rows, :] = True
    template[:, full_cols] = True
    if not np.array_equal(mask, template):
        return None
    return full_rows, full_cols, cell_h


def _detect_frame(grid: np.ndarray, bg: int) -> dict:
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color, connectivity=4):
            r0, c0, r1, c1 = _bbox(coords)
            submask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            submask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            frame = _frame_candidate(submask)
            if frame is None:
                continue
            full_rows, full_cols, cell = frame
            cand = {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "rows": [r0 + rr for rr in full_rows],
                "cols": [c0 + cc for cc in full_cols],
                "cell": cell,
                "count": int(coords.shape[0]),
            }
            if best is None or cand["count"] > best["count"]:
                best = cand
    assert best is not None, "failed to detect frame"
    return best


def _is_path_component(coords: np.ndarray) -> tuple[bool, list[tuple[int, int]]]:
    cells = {tuple(map(int, rc)) for rc in coords}
    degrees: dict[tuple[int, int], int] = {}
    for rc in cells:
        deg = 0
        for dr, dc in NEIGH4:
            if (rc[0] + dr, rc[1] + dc) in cells:
                deg += 1
        if deg > 2:
            return False, []
        degrees[rc] = deg
    endpoints = [rc for rc, deg in degrees.items() if deg == 1]
    return len(endpoints) == 2, endpoints


def _adjacent_pattern_ids(
    rc: tuple[int, int], pattern_ids: np.ndarray
) -> set[int]:
    h, w = pattern_ids.shape
    ids: set[int] = set()
    for dr, dc in NEIGH4:
        nr, nc = rc[0] + dr, rc[1] + dc
        if 0 <= nr < h and 0 <= nc < w and pattern_ids[nr, nc] >= 0:
            ids.add(int(pattern_ids[nr, nc]))
    return ids


def _adjacent_to_frame(rc: tuple[int, int], frame_mask: np.ndarray) -> bool:
    h, w = frame_mask.shape
    for dr, dc in NEIGH4:
        nr, nc = rc[0] + dr, rc[1] + dc
        if 0 <= nr < h and 0 <= nc < w and frame_mask[nr, nc]:
            return True
    return False


def _detect_line_color(grid: np.ndarray, bg: int, frame: dict) -> int:
    frame_mask = grid == frame["color"]
    best_color = None
    best_score = -1
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, frame["color"]):
            continue
        score = 0
        for comp in _components(grid == color, connectivity=4):
            ok, endpoints = _is_path_component(comp)
            if not ok:
                continue
            touches = sum(_adjacent_to_frame(ep, frame_mask) for ep in endpoints)
            if touches == 1:
                score += 1
        if score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None and best_score > 0, "failed to detect line color"
    return int(best_color)


def _extract_patterns(
    grid: np.ndarray, bg: int, frame_color: int, line_color: int, cell: int
) -> tuple[list[dict], np.ndarray]:
    mask = (grid != bg) & (grid != frame_color) & (grid != line_color)
    pattern_ids = np.full(grid.shape, -1, dtype=int)
    patterns: list[dict] = []
    for comp in _components(mask, connectivity=8):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != cell or w != cell or comp.shape[0] != cell * cell:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        if np.any(patch == bg) or np.any(patch == frame_color) or np.any(patch == line_color):
            continue
        pid = len(patterns)
        pattern_ids[r0 : r1 + 1, c0 : c1 + 1] = pid
        patterns.append({"bbox": (r0, c0, r1, c1), "patch": patch})
    assert patterns, "failed to detect patterns"
    return patterns, pattern_ids


def _trace_path(coords: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in coords}
    neighbors = {
        rc: [n for n in ((rc[0] + dr, rc[1] + dc) for dr, dc in NEIGH4) if n in cells]
        for rc in cells
    }
    path = [start]
    prev = None
    cur = start
    while cur != end:
        nxts = [n for n in neighbors[cur] if n != prev]
        assert len(nxts) == 1, "line path must be simple"
        prev, cur = cur, nxts[0]
        path.append(cur)
    return path


def _cw_turns(path: list[tuple[int, int]]) -> int:
    dirs = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    cw = 0
    for d0, d1 in zip(dirs, dirs[1:]):
        if d0 == d1:
            continue
        i0 = DIR_INDEX[d0]
        i1 = DIR_INDEX[d1]
        if i1 == (i0 - 1) % 4:
            cw += 1
        elif i1 == (i0 + 1) % 4:
            cw -= 1
        else:
            raise AssertionError("line path cannot reverse")
    return cw % 4


def _slot_from_endpoint(endpoint: tuple[int, int], frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    rows = frame["rows"]
    cols = frame["cols"]
    er, ec = endpoint
    if er == r0 - 1:
        row = 0
        for j in range(len(cols) - 1):
            if cols[j] < ec < cols[j + 1]:
                return row, j
    if er == r1 + 1:
        row = len(rows) - 2
        for j in range(len(cols) - 1):
            if cols[j] < ec < cols[j + 1]:
                return row, j
    if ec == c0 - 1:
        col = 0
        for i in range(len(rows) - 1):
            if rows[i] < er < rows[i + 1]:
                return i, col
    if ec == c1 + 1:
        col = len(cols) - 2
        for i in range(len(rows) - 1):
            if rows[i] < er < rows[i + 1]:
                return i, col
    raise AssertionError("failed to map line endpoint to slot")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common(g)
    frame = _detect_frame(g, bg)
    line_color = _detect_line_color(g, bg, frame)
    patterns, pattern_ids = _extract_patterns(g, bg, frame["color"], line_color, frame["cell"])

    frame_mask = g == frame["color"]
    line_components = _components(g == line_color, connectivity=4)
    assert len(line_components) == len(patterns), "each pattern must have one line"

    r0, c0, r1, c1 = frame["bbox"]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    local_rows = [rr - r0 for rr in frame["rows"]]
    local_cols = [cc - c0 for cc in frame["cols"]]
    out[local_rows, :] = frame["color"]
    out[:, local_cols] = frame["color"]

    for comp in line_components:
        ok, endpoints = _is_path_component(comp)
        assert ok
        frame_flags = [_adjacent_to_frame(ep, frame_mask) for ep in endpoints]
        assert sum(frame_flags) == 1
        frame_ep = endpoints[0] if frame_flags[0] else endpoints[1]
        pattern_ep = endpoints[1] if frame_flags[0] else endpoints[0]
        adjacent = _adjacent_pattern_ids(pattern_ep, pattern_ids)
        assert len(adjacent) == 1
        pattern = patterns[next(iter(adjacent))]
        path = _trace_path(comp, pattern_ep, frame_ep)
        slot_r, slot_c = _slot_from_endpoint(frame_ep, frame)
        rotated = np.rot90(pattern["patch"], -_cw_turns(path))
        dest_r = frame["rows"][slot_r] - r0 + 1
        dest_c = frame["cols"][slot_c] - c0 + 1
        _paste(out, rotated, dest_r, dest_c)

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, rows: int, cols: int) -> None:
    h = rows * CELL + rows + 1
    w = cols * CELL + cols + 1
    grid[r0 : r0 + h, c0] = FRAME
    grid[r0 : r0 + h, c0 + w - 1] = FRAME
    grid[r0, c0 : c0 + w] = FRAME
    grid[r0 + h - 1, c0 : c0 + w] = FRAME
    for i in range(1, rows):
        rr = r0 + i * (CELL + 1)
        grid[rr, c0 : c0 + w] = FRAME
    for j in range(1, cols):
        cc = c0 + j * (CELL + 1)
        grid[r0 : r0 + h, cc] = FRAME


def _frame_output(rows: int, cols: int) -> np.ndarray:
    h = rows * CELL + rows + 1
    w = cols * CELL + cols + 1
    out = np.full((h, w), BG, dtype=int)
    _draw_frame(out, 0, 0, rows, cols)
    return out


def _entry_col(c0: int, col: int) -> int:
    return c0 + 2 + 4 * col


def _vertical_segment(r_start: int, r_end: int, c: int) -> list[tuple[int, int]]:
    step = 1 if r_end >= r_start else -1
    return [(r, c) for r in range(r_start, r_end + step, step)]


def _horizontal_segment(r: int, c_start: int, c_end: int) -> list[tuple[int, int]]:
    step = 1 if c_end >= c_start else -1
    return [(r, c) for c in range(c_start, c_end + step, step)]


def _merge_path(*segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for seg in segments:
        if not seg:
            continue
        if path and path[-1] == seg[0]:
            path.extend(seg[1:])
        else:
            path.extend(seg)
    return path


def _branch_turns(branch: str) -> int:
    if branch == "straight":
        return 0
    if branch == "left90":
        return 1
    if branch == "right90":
        return 3
    return 2


def _random_pattern() -> np.ndarray:
    for _ in range(200):
        colors = random.sample(PATTERN_COLORS, random.randint(2, 4))
        flat = [random.choice(colors) for _ in range(CELL * CELL)]
        patch = np.array(flat, dtype=int).reshape(CELL, CELL)
        rots = {np.rot90(patch, k).tobytes() for k in range(4)}
        if len(set(flat)) >= 2 and len(rots) == 4:
            return patch
    raise RuntimeError("failed to sample asymmetric pattern")


def _build_top_path(branch: str, pr: int, pc: int, frame_r0: int, entry_c: int) -> tuple[list[tuple[int, int]], int]:
    center = pc + 1
    endpoint_r = frame_r0 - 1
    if branch == "straight":
        start_r = pr + CELL
        return _vertical_segment(start_r, endpoint_r, center), 0
    if branch == "left90":
        start_r = pr + CELL
        path = _merge_path(
            _vertical_segment(start_r, endpoint_r, center),
            _horizontal_segment(endpoint_r, center, entry_c),
        )
        return path, 1
    if branch == "right90":
        start_r = pr + CELL
        path = _merge_path(
            _vertical_segment(start_r, endpoint_r, center),
            _horizontal_segment(endpoint_r, center, entry_c),
        )
        return path, 3
    detour = random.randint(1, 2)
    start_r = pr - 1
    turn_r = start_r - detour
    path = _merge_path(
        _vertical_segment(start_r, turn_r, center),
        _horizontal_segment(turn_r, center, entry_c),
        _vertical_segment(turn_r, endpoint_r, entry_c),
    )
    return path, 2


def _build_bottom_path(branch: str, pr: int, pc: int, frame_r1: int, entry_c: int) -> tuple[list[tuple[int, int]], int]:
    center = pc + 1
    endpoint_r = frame_r1 + 1
    if branch == "straight":
        start_r = pr - 1
        return _vertical_segment(start_r, endpoint_r, center), 0
    if branch in ("left90", "right90"):
        start_r = pr - 1
        path = _merge_path(
            _vertical_segment(start_r, endpoint_r, center),
            _horizontal_segment(endpoint_r, center, entry_c),
        )
        return path, _branch_turns(branch)
    detour = random.randint(1, 2)
    start_r = pr + CELL
    turn_r = start_r + detour
    path = _merge_path(
        _vertical_segment(start_r, turn_r, center),
        _horizontal_segment(turn_r, center, entry_c),
        _vertical_segment(turn_r, endpoint_r, entry_c),
    )
    return path, 2


def _paint_line(grid: np.ndarray, path: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    for r, c in path:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BG:
            return False
        seen.add((r, c))
    for r, c in seen:
        grid[r, c] = LINE
    return True


def _branch_offset(branch: str, top_row: bool) -> int:
    if branch == "straight":
        return 0
    if branch == "left90":
        return -1 if top_row else 1
    if branch == "right90":
        return 1 if top_row else -1
    if branch == "left180":
        return 1 if top_row else -1
    return -1 if top_row else 1


def _canonical_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    cols = random.randint(2, 4)
    rows = 2
    frame_h = rows * CELL + rows + 1
    frame_w = cols * CELL + cols + 1

    branch_choices = [random.choice(BRANCHES) for _ in range(rows * cols)]
    if not any(b == "straight" for b in branch_choices):
        return None
    if not any(b.startswith("left") for b in branch_choices):
        return None
    if not any(b.startswith("right") for b in branch_choices):
        return None

    top_levels = [4 + 4 * i for i in range(cols)]
    bottom_levels = [4 * i for i in range(cols)]
    random.shuffle(top_levels)
    random.shuffle(bottom_levels)

    frame_r0 = 4 * cols + random.randint(6, 8)
    frame_c0 = random.randint(3, 7)
    frame_r1 = frame_r0 + frame_h - 1
    bottom_base = frame_r1 + random.randint(4, 6)
    height = bottom_base + max(bottom_levels) + CELL + 4
    width = frame_c0 + frame_w + random.randint(3, 7)

    inp = np.full((height, width), BG, dtype=int)
    _draw_frame(inp, frame_r0, frame_c0, rows, cols)
    out = _frame_output(rows, cols)

    top_branches = branch_choices[:cols]
    bottom_branches = branch_choices[cols:]

    for col in range(cols):
        entry_c = _entry_col(frame_c0, col)
        patch_top = _random_pattern()
        patch_bottom = _random_pattern()

        top_branch = top_branches[col]
        top_offset = _branch_offset(top_branch, top_row=True)
        top_center = entry_c + top_offset
        top_pc = top_center - 1
        top_pr = top_levels[col]
        if top_pc < 0 or top_pc + CELL > width or top_pr < 0 or top_pr + CELL > frame_r0 - 1:
            return None
        if np.any(inp[top_pr : top_pr + CELL, top_pc : top_pc + CELL] != BG):
            return None
        _paste(inp, patch_top, top_pr, top_pc)
        top_path, _ = _build_top_path(top_branch, top_pr, top_pc, frame_r0, entry_c)
        if not _paint_line(inp, top_path):
            return None
        _paste(out, np.rot90(patch_top, -_branch_turns(top_branch)), 1, 1 + 4 * col)

        bottom_branch = bottom_branches[col]
        bottom_offset = _branch_offset(bottom_branch, top_row=False)
        bottom_center = entry_c + bottom_offset
        bottom_pc = bottom_center - 1
        bottom_pr = bottom_base + bottom_levels[col]
        if bottom_pc < 0 or bottom_pc + CELL > width or bottom_pr <= frame_r1 + 1 or bottom_pr + CELL > height:
            return None
        if np.any(inp[bottom_pr : bottom_pr + CELL, bottom_pc : bottom_pc + CELL] != BG):
            return None
        _paste(inp, patch_bottom, bottom_pr, bottom_pc)
        bottom_path, _ = _build_bottom_path(bottom_branch, bottom_pr, bottom_pc, frame_r1, entry_c)
        if not _paint_line(inp, bottom_path):
            return None
        _paste(out, np.rot90(patch_bottom, -_branch_turns(bottom_branch)), 5, 1 + 4 * col)

    return inp, out


def _rotate_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randint(0, 3)
    return np.rot90(inp, k).copy(), np.rot90(out, k).copy()


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _canonical_attempt()
        if built is None:
            continue
        inp, expected = _rotate_pair(*built)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate input for task 4")
